use std::{
    any::Any,
    cmp,
    collections::HashMap,
    fmt,
    os::fd::OwnedFd,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use calloop::LoopHandle;
use portable_atomic::AtomicF64;
use rustix::fs::OFlags;
use smallvec::SmallVec;
use tracing::{debug, trace, warn};
#[cfg(feature = "wayland_frontend")]
use wayland_server::{DisplayHandle, Resource};
use x11rb::{
    CURRENT_TIME,
    connection::Connection as _,
    errors::{ConnectionError, ReplyOrIdError},
    protocol::{
        sync::{Alarm, ConnectionExt as _, Counter, CreateAlarmAux, Int64, TESTTYPE, VALUETYPE},
        xfixes::SelectionNotifyEvent,
        xproto::{
            Atom, AtomEnum, ClientMessageData, ClientMessageEvent, ConfigureWindowAux, ConnectionExt,
            CreateWindowAux, EventMask, PropMode, Screen, StackMode, Window as X11Window, WindowClass,
        },
    },
    rust_connection::RustConnection,
    wrapper::ConnectionExt as _,
};

use crate::{
    backend::input::InputTime,
    input::{
        Seat, SeatHandler,
        dnd::{DnDGrab, DndAction, DndFocus, DndGrabHandler, OfferData, Source, SourceMetadata},
        pointer::Focus,
    },
    utils::{IsAlive, Logical, Point, Serial},
    wayland::seat::WaylandFocus,
    xwayland::{
        X11Surface, XWaylandClientData, XwmHandler,
        xwm::{Atoms, OwnedX11Window, XwmId, atom_from_mime, mime_from_atom, selection::XWmSelection},
    },
};

// TODO: We should have a central source for this
const BTN_LEFT: u32 = 0x110;
const DND_VERSION: u32 = 5;
const MIN_DND_VERSION: u32 = 2;
/// How long a drop on an X11 window waits for the target's XdndFinished (and first, when a position is
/// still unanswered, for its XdndStatus). A target that has not finished by then is taken for hung: the
/// offer ends and its source is cancelled.
///
/// The XDND spec gives no figure, only that a source "must be prepared to time out in case the target
/// malfunctions". Targets finish within milliseconds of reading the data; this leaves room for a large
/// transfer or a target that asks the user first, while an offer left active keeps every new X11 drag
/// from taking XdndSelection (see `xfixes_selection_notify`).
const DROP_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug)]
pub struct XWmDnd {
    pub selection: XWmSelection,
    pub active_offer: Option<XwmActiveOffer>,
    pub active_drag: Option<XwmActiveDrag>,
    pub xdnd_active: Arc<AtomicBool>,
    /// The alarm, on the X server's SERVERTIME counter, that ends the active offer `DROP_TIMEOUT` after
    /// its drop
    drop_alarm: Option<Alarm>,
}

impl XWmDnd {
    pub fn new(conn: &Arc<RustConnection>, screen: &Screen, atoms: &Atoms) -> Result<Self, ReplyOrIdError> {
        let selection = XWmSelection::new(conn, screen, atoms, atoms.XdndSelection)?;

        conn.change_property32(
            PropMode::REPLACE,
            *selection.window,
            atoms.XdndAware,
            AtomEnum::ATOM,
            &[DND_VERSION],
        )?;
        conn.flush()?;

        Ok(XWmDnd {
            selection,
            active_offer: None,
            active_drag: None,
            xdnd_active: Arc::new(AtomicBool::new(false)),
            drop_alarm: None,
        })
    }

    pub fn update_screen(&self, screen: &Screen) -> Result<(), ReplyOrIdError> {
        if let Some(active_drag) = self.active_drag.as_ref() {
            let mut drag_state = active_drag.state.lock().unwrap();

            if drag_state.mapped {
                self.selection.conn.configure_window(
                    *active_drag.target,
                    &ConfigureWindowAux::new()
                        .width(screen.width_in_pixels as u32)
                        .height(screen.height_in_pixels as u32),
                )?;
            } else {
                drag_state.pending_configure = Some((screen.width_in_pixels, screen.height_in_pixels));
            }
        }

        Ok(())
    }

    pub fn window_destroyed<D>(&mut self, window: &X11Window, loop_handle: &LoopHandle<'_, D>) -> bool {
        let mut res = self.selection.window_destroyed(window, loop_handle);
        if let Some(active_drag) = self.active_drag.as_ref() {
            if active_drag.owner == *window
                || active_drag
                    .state
                    .lock()
                    .unwrap()
                    .source
                    .as_ref()
                    .is_some_and(|w| w == window)
            {
                self.active_drag.take();
                self.xdnd_active.store(false, Ordering::Release);
                res = true;
            }
        }
        if self.active_offer.as_ref().is_some_and(|offer| {
            let state = offer.state.lock().unwrap();
            state.target == *window || state.proxy == Some(*window)
        }) {
            // The target went away (its client crashed or was killed) and will never send XdndFinished.
            // Left as it is, the offer would stay active for good: every new X11 drag would lose
            // XdndSelection to us (see `xfixes_selection_notify`), and the source would never hear back.
            self.end_offer();
        }
        res
    }

    /// Ends the active offer, whose target will not send XdndFinished: a grab still over the target
    /// can no longer drop on it, and the source of a drop already made is cancelled.
    pub fn end_offer(&mut self) {
        self.disarm_drop_timeout();
        if let Some(offer) = self.active_offer.take() {
            let mut state = offer.state.lock().unwrap();
            // A drop the target took was performed, as far as its source knows, which now waits for the
            // end. One it did not take was cancelled by the grab already (its offer was disabled).
            let performed = state.dropped && state.active;
            state.active = false;
            std::mem::drop(state);
            if performed {
                offer.source.cancel();
            }
        }
    }

    /// Ends the active offer, a drop still waiting for its target, which is still there (`end_offer`). A
    /// target that was never sent the drop, as it had not answered the last position, is told the drag
    /// left (XdndLeave), so that it does not keep a drag it believes is still over it. One that was sent
    /// the drop is told nothing more: after XdndDrop, only the target speaks (XdndFinished).
    fn end_waiting_drop(&mut self) {
        if let Some(offer) = self.active_offer.as_ref() {
            let state = offer.state.lock().unwrap();
            if state.pos_pending {
                let data = [*self.selection.window, 0, 0, 0, 0];
                trace!("Sending XdndLeave: {:?}", data);
                if let Err(err) = self.selection.conn.send_event(
                    false,
                    state.proxy.unwrap_or(state.target),
                    EventMask::NO_EVENT,
                    ClientMessageEvent::new(32, state.target, self.selection.atoms.XdndLeave, data),
                ) {
                    warn!("Failed to send DND_LEAVE event: {:?}", err);
                }
                let _ = self.selection.conn.flush();
            }
        }
        self.end_offer();
    }

    /// Arms the timeout of the active offer's drop (`DROP_TIMEOUT`): an alarm on the X server's SERVERTIME
    /// counter, relative to its value now (the server reads a system counter when it sets an alarm on
    /// it), which goes off once and is reported as an AlarmNotify (`handle_alarm`). Without the SYNC
    /// extension (`servertime` is `None`), a drop has no timeout.
    fn arm_drop_timeout(&mut self, servertime: Option<Counter>) {
        self.disarm_drop_timeout();
        let Some(servertime) = servertime else {
            return;
        };
        let conn = &self.selection.conn;
        let alarm = match conn.generate_id() {
            Ok(alarm) => alarm,
            Err(err) => {
                warn!(?err, "Failed to arm the timeout of an XDND drop");
                return;
            }
        };
        let aux = CreateAlarmAux::new()
            .counter(servertime)
            .value_type(VALUETYPE::RELATIVE)
            .value(Int64 {
                hi: 0,
                lo: DROP_TIMEOUT.as_millis() as u32,
            })
            .test_type(TESTTYPE::POSITIVE_COMPARISON)
            .delta(Int64 { hi: 0, lo: 0 })
            .events(1);
        if let Err(err) = conn.sync_create_alarm(alarm, &aux).and_then(|_| conn.flush()) {
            warn!(?err, "Failed to arm the timeout of an XDND drop");
            return;
        }
        self.drop_alarm = Some(alarm);
    }

    fn disarm_drop_timeout(&mut self) {
        if let Some(alarm) = self.drop_alarm.take() {
            // fails only when the X connection is gone, and the alarm with it
            let _ = self.selection.conn.sync_destroy_alarm(alarm);
            let _ = self.selection.conn.flush();
        }
    }

    /// The alarm `alarm` went off: when it is the timeout of the active offer's drop, the target did not
    /// finish the drop in time, and the drop ends.
    pub fn handle_alarm(&mut self, alarm: Alarm) {
        if self.drop_alarm != Some(alarm) {
            return;
        }
        // Only the drop of the active offer arms the alarm (`DndFocus::drop`), each one anew, so an active
        // offer that was dropped is the alarm's own. Any other is a second drag's, at the same time: the
        // alarm's drop lost the offer to it (that drag's `DndFocus::leave` clears the active offer,
        // whichever drag it is), and the alarm ends nothing.
        if self
            .active_offer
            .as_ref()
            .is_some_and(|offer| offer.state.lock().unwrap().dropped)
        {
            warn!(
                "XDND drop target did not finish the drop within {:?}, cancelling it",
                DROP_TIMEOUT
            );
            self.end_waiting_drop();
        } else {
            self.disarm_drop_timeout();
        }
    }

    pub fn has_window(&self, window: &X11Window) -> bool {
        if let Some(drag) = self.active_drag.as_ref() {
            if drag.target == *window {
                return true;
            }
        }

        self.selection.has_window(window)
    }

    pub fn xfixes_selection_notify<D>(
        dh: &DisplayHandle,
        data: &mut D,
        id: XwmId,
        event: SelectionNotifyEvent,
    ) -> Result<(), ReplyOrIdError>
    where
        D: XwmHandler + SeatHandler + DndGrabHandler + 'static,
        <D as SeatHandler>::PointerFocus: DndFocus<D>,
        <D as SeatHandler>::TouchFocus: DndFocus<D>,
    {
        let xwm = data.xwm_state(id);

        xwm.dnd.selection.owner = event.owner;
        if xwm.dnd.selection.owner == *xwm.dnd.selection.window {
            xwm.dnd.selection.timestamp = event.timestamp;
            return Ok(());
        }

        if xwm.dnd.active_offer.is_some() {
            // rough X11 client tries to take over the selection, take it back.
            xwm.conn
                .set_selection_owner(*xwm.dnd.selection.window, xwm.atoms.XdndSelection, CURRENT_TIME)?;
            return Ok(());
        }

        if let Some(active_drag) = xwm.dnd.active_drag.as_ref() {
            if active_drag.owner == xwm.dnd.selection.owner
                && active_drag.state.lock().unwrap().x11 != X11State::Finished
            {
                warn!("Got another selection notify for an already active grab. Ignoring...");
                return Ok(());
            } else {
                // new drag, ours is presumably outdated/stale now
                debug!("Dropping stale Xwm drag");

                let active_drag = xwm.dnd.active_drag.take().unwrap();
                active_drag.state.lock().unwrap().x11 = X11State::Finished;
                xwm.dnd.xdnd_active.store(false, Ordering::Release);
            }
        }

        if event.owner == x11rb::NONE {
            trace!("XDND selection went away");
            xwm.dnd.active_drag.take();
            xwm.dnd.xdnd_active.store(false, Ordering::Release);
            return Ok(());
        }

        trace!("New XDND selection from {}", event.owner);
        // figure out, if we have a grab
        //
        // Only a grab that started on a surface of this XWayland can become an X11 drag. Any X11 client may
        // take XdndSelection at any time; otherwise it would replace whatever grab holds the left button
        // (a Wayland client's drag, a compositor's move) and offer its own data to the surface under the pointer.
        let ptr_grab = data
            .seat_state()
            .seats
            .iter()
            .flat_map(|seat| {
                seat.get_pointer().and_then(|ptr| {
                    ptr.with_grab(|serial, grab| (seat.clone(), serial, grab.start_data().clone()))
                        .filter(|(_, _, start_data)| {
                            start_data.button == BTN_LEFT
                                && started_on_xwayland(start_data.focus.as_ref(), id)
                        })
                })
            })
            .max_by(|(_, s1, _), (_, s2, _)| s1.partial_cmp(s2).unwrap_or(cmp::Ordering::Equal));

        let touch_grab = data
            .seat_state()
            .seats
            .iter()
            .flat_map(|seat| {
                seat.get_touch().and_then(|touch| {
                    touch
                        .with_grab(|serial, grab| (seat.clone(), serial, grab.start_data().clone()))
                        .filter(|(_, _, start_data)| started_on_xwayland(start_data.focus.as_ref(), id))
                })
            })
            .max_by(|(_, s1, _), (_, s2, _)| s1.partial_cmp(s2).unwrap_or(cmp::Ordering::Equal));

        if ptr_grab.is_none() && touch_grab.is_none() {
            return Ok(());
        }

        // create our drop proxy
        let xwm = data.xwm_state(id);
        let window = xwm.conn.generate_id()?;
        xwm.conn.create_window(
            xwm.screen.root_depth,
            window,
            xwm.screen.root,
            0,
            0,
            xwm.screen.width_in_pixels,
            xwm.screen.height_in_pixels,
            0,
            WindowClass::INPUT_OUTPUT,
            xwm.screen.root_visual,
            &CreateWindowAux::new().event_mask(EventMask::PROPERTY_CHANGE),
        )?;
        xwm.conn.change_property32(
            PropMode::REPLACE,
            window,
            xwm.atoms.XdndAware,
            AtomEnum::ATOM,
            &[DND_VERSION],
        )?;
        xwm.conn.change_property8(
            PropMode::REPLACE,
            window,
            xwm.atoms.WM_NAME,
            xwm.atoms.UTF8_STRING,
            "Smithay XDND proxy".as_bytes(),
        )?;
        xwm.conn.map_window(window)?;
        xwm.conn
            .configure_window(window, &ConfigureWindowAux::new().stack_mode(StackMode::ABOVE))?;
        xwm.conn.flush()?;

        let window = OwnedX11Window::new(window, &xwm.conn);

        // create our dnd source
        let state = Arc::new(Mutex::new(XwmSourceState {
            mapped: true,
            x11: X11State::Active,
            wayland: WlState::Active,
            source: None,
            pending_configure: None,
            metadata: SourceMetadata {
                mime_types: vec![],
                dnd_actions: SmallVec::new_const(),
            },
            active_action: DndAction::None,
            last_timestamp: None,
            version: 0,
        }));
        let source = XwmDndSource {
            xwm: id,
            conn: Arc::downgrade(&xwm.conn),
            atoms: xwm.atoms,
            target: window.clone(),
            state: state.clone(),
            pending_transfers: xwm.dnd.selection.pending_transfers.clone(),
            xdnd_active: xwm.dnd.xdnd_active.clone(),
        };
        xwm.dnd.active_drag = Some(XwmActiveDrag {
            target: window,
            owner: xwm.dnd.selection.owner,
            state,
            pending_transfers: xwm.dnd.selection.pending_transfers.clone(),
        });
        xwm.dnd.xdnd_active.store(true, Ordering::Release);

        // create the DndGrab
        match (ptr_grab, touch_grab) {
            (Some((seat, s1, start_data)), Some((_, s2, _))) if s1 >= s2 => {
                // pointer_grab
                let grab = DnDGrab::new_pointer(dh, start_data, source, seat.clone());
                seat.get_pointer().unwrap().set_grab(data, grab, s1, Focus::Keep);
            }
            (Some((seat, serial, start_data)), None) => {
                // pointer grab
                let grab = DnDGrab::new_pointer(dh, start_data, source, seat.clone());
                seat.get_pointer()
                    .unwrap()
                    .set_grab(data, grab, serial, Focus::Keep);
            }
            (_, Some((seat, serial, start_data))) => {
                // touch_grab
                let grab = DnDGrab::new_touch(dh, start_data, source, seat.clone());
                seat.get_touch().unwrap().set_grab(data, grab, serial);
            }
            (None, None) => unreachable!(),
        };

        Ok(())
    }

    pub fn handle_status(
        &mut self,
        window: &X11Surface,
        status_msg: ClientMessageData,
        atoms: &Atoms,
        client_scale: Arc<AtomicF64>,
    ) {
        if let Some(offer) = self.active_offer.as_ref() {
            let data = status_msg.as_data32();
            let mut offer_state = offer.state.lock().unwrap();

            if offer_state.target == data[0] {
                let mut pos_update = false;
                offer_state.client_accepts = (data[1] & 1) != 0;

                if !offer_state.dropped {
                    let old_action = offer_state.preferred_action;
                    let new_action = DndAction::from_x(data[4], atoms);
                    if offer_state.supported_actions.contains(&new_action) {
                        offer_state.preferred_action = new_action;
                    } else if offer_state.supported_actions.contains(&DndAction::Copy) {
                        offer_state.preferred_action = DndAction::Copy;
                    } else {
                        offer_state.preferred_action = DndAction::None;
                    }

                    offer.source.choose_action(offer_state.preferred_action);
                    if old_action != offer_state.preferred_action {
                        pos_update = true;
                    }
                }

                // TODO: rectangle in data[2] and data[3] for optimizations

                offer_state.pos_pending = false;
                if let Some(pos) = offer_state
                    .pos_cached
                    .take()
                    .or(pos_update.then_some(offer_state.last_pos))
                {
                    let location = (window.last_configure().loc + pos.to_i32_round())
                        .to_client_precise_round::<_, i32>(client_scale.load(Ordering::Acquire));

                    let data = [
                        *self.selection.window,
                        0,
                        ((location.x << 16) | location.y) as u32,
                        CURRENT_TIME,
                        offer_state.preferred_action.to_x(atoms),
                    ];

                    if let Err(err) = self.selection.conn.send_event(
                        false,
                        window.window_id(),
                        EventMask::NO_EVENT,
                        ClientMessageEvent::new(32, window.window_id(), atoms.XdndPosition, data),
                    ) {
                        warn!("Failed to send DND_POSITION event: {:?}", err);
                    }
                    let _ = self.selection.conn.flush();
                    offer_state.pos_pending = true;
                } else if offer_state.dropped {
                    // finish drop

                    offer.source.drop_performed();

                    let data = [*self.selection.window, 0, CURRENT_TIME, 0, 0];
                    if let Err(err) = self.selection.conn.send_event(
                        false,
                        window.window_id(),
                        EventMask::NO_EVENT,
                        ClientMessageEvent::new(32, window.window_id(), self.selection.atoms.XdndDrop, data),
                    ) {
                        warn!("Failed to send DND_DROP event: {:?}", err);
                    }
                    let _ = self.selection.conn.flush();
                }
            }
        }
    }

    pub fn handle_finished(&mut self, status_msg: ClientMessageData) {
        if let Some(offer) = self.active_offer.as_ref() {
            let offer_state = offer.state.lock().unwrap();
            let data = status_msg.as_data32();
            trace!("Got XDND finished msg: {:?}", data);

            if data[0] != offer_state.target {
                return;
            }

            if !offer_state.dropped {
                return;
            }

            offer.source.finished();
            std::mem::drop(offer_state);
            self.active_offer.take();
            self.disarm_drop_timeout();
        }
    }

    /// Whether a drop on the X11 window the drag is offered to is in progress: that window was sent XdndDrop for a
    /// drop the grab took, accepted the drag in its last answer (XdndStatus), and asks for the drag's data until it
    /// finishes (`handle_finished`) or the drop ends without that (`end_offer`: the window went away, `DROP_TIMEOUT`
    /// is up, the next drag came over an X11 window, or the X connection closed). Each of these takes the active offer.
    pub fn drop_in_progress(&self) -> bool {
        self.active_offer.as_ref().is_some_and(|offer| {
            let state = offer.state.lock().unwrap();
            // XdndDrop also goes to a window that refused the drag; the grab then disables the offer and cancels the
            // source. A drop waits for the answer to a pending XdndPosition, and `handle_status` sends XdndDrop then
            // whatever that answer says: the grab took the drop on the answer before it, so the offer stays active
            // when this one refuses, and only `client_accepts` tells.
            state.active && state.dropped && !state.pos_pending && state.client_accepts
        })
    }

    pub fn transfer(&mut self, mime_type: impl AsRef<str>, send_fd: OwnedFd) {
        if let Some(offer) = self.active_offer.as_ref() {
            offer.source.send(mime_type.as_ref(), send_fd);
        }
    }

    pub fn handle_enter(&mut self, enter_msg: ClientMessageData) -> Result<(), ReplyOrIdError> {
        if let Some(drag) = self.active_drag.as_mut() {
            let mut drag_state = drag.state.lock().unwrap();
            let data = enter_msg.as_data32();

            trace!("Got XDND enter msg: {:?}", data);
            if drag_state.source.is_some_and(|w| w != data[0]) {
                debug!(
                    "Received XdndEnter from unknown source (got: {}, state: {:?}), ignoring..",
                    data[0], drag_state.source
                );
                return Ok(());
            }
            drag_state.source = Some(data[0]);
            let source = data[0];

            let version = data[1] >> 24;
            if version > DND_VERSION {
                warn!("New unsupported XDND version: {}", data[1]);
                return Ok(());
            }

            // get types
            let mimes = if (data[1] & 1) == 0 {
                (2..5)
                    .flat_map(|i| {
                        mime_from_atom(data[i], &self.selection.conn, &self.selection.atoms).transpose()
                    })
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                let reply = self
                    .selection
                    .conn
                    .get_property(
                        false,
                        source,
                        self.selection.atoms.XdndTypeList,
                        AtomEnum::ANY,
                        0,
                        0x1fffffff,
                    )?
                    .reply()?;
                let Some(values) = reply.value32() else {
                    return Ok(());
                };

                values
                    .flat_map(|atom| {
                        mime_from_atom(atom, &self.selection.conn, &self.selection.atoms).transpose()
                    })
                    .collect::<Result<Vec<_>, _>>()?
            };

            drag_state.x11 = X11State::Active;
            drag_state.metadata.mime_types = mimes;
            drag_state.version = version;
        }

        Ok(())
    }

    pub fn handle_position(&mut self, pos_msg: ClientMessageData) -> Result<(), ReplyOrIdError> {
        if let Some(drag) = self.active_drag.as_ref() {
            let mut drag_state = drag.state.lock().unwrap();

            let data = pos_msg.as_data32();
            if drag_state.source.is_none_or(|w| w != data[0]) {
                debug!(
                    "Received XdndPosition from unknown source (got: {}, expected: {:?}), ignoring..",
                    data[0], drag_state.source
                );
                return Ok(());
            }
            let source = data[0];

            let mut actions = SmallVec::from_elem(DndAction::Copy, 1); // copy is always supported on XDND
            if drag_state.version > 1 {
                #[allow(clippy::single_match)]
                match DndAction::from_x(data[4], &self.selection.atoms) {
                    DndAction::Move => actions.push(DndAction::Move),
                    _ => {} // We don't support Ask or Private atm,
                }
            }

            drag_state.last_timestamp = Some(data[3]);
            drag_state.metadata.dnd_actions = actions;

            let mut flags = 1 << 1;
            if !matches!(&drag_state.wayland, WlState::Cancelled)
                && drag_state.active_action != DndAction::None
            {
                flags |= 1 << 0;
            }
            let data = [
                *drag.target,
                flags,
                0,
                0,
                drag_state.active_action.to_x(&self.selection.atoms),
            ];
            self.selection.conn.send_event(
                false,
                source,
                EventMask::NO_EVENT,
                ClientMessageEvent::new(32, source, self.selection.atoms.XdndStatus, data),
            )?;
            self.selection.conn.flush()?;

            if matches!(&drag_state.wayland, &WlState::GotFd(_, _)) {
                let mut wl_state = WlState::Active;
                std::mem::swap(&mut drag_state.wayland, &mut wl_state);

                let WlState::GotFd(mime_type, fd) = wl_state else {
                    unreachable!()
                };

                let Some(atom) = atom_from_mime(&mime_type, &self.selection.conn, &self.selection.atoms)?
                else {
                    warn!("Unable to determine atom for mime_type ({})", mime_type);
                    return Ok(());
                };

                drag.pending_transfers
                    .lock()
                    .unwrap()
                    .insert(*drag.target, (drag.target.clone(), fd));

                self.selection.conn.convert_selection(
                    drag.owner,
                    self.selection.atoms.XdndSelection,
                    atom,
                    self.selection.atoms._WL_SELECTION,
                    drag_state.last_timestamp.unwrap(),
                )?;
                self.selection.conn.flush()?;
            }
        }
        Ok(())
    }

    pub fn handle_leave(&mut self, leave_msg: ClientMessageData) -> Result<(), ReplyOrIdError> {
        if let Some(drag) = self.active_drag.as_mut() {
            let mut state = drag.state.lock().unwrap();

            let data = leave_msg.as_data32();
            trace!("Got XDND leave msg: {:?}", data);

            if state.source.is_none_or(|w| w != data[0]) {
                debug!(
                    "Received XdndLeave from unknown source (got: {}, expected: {:?}), ignoring..",
                    data[0], state.source
                );
                return Ok(());
            }
            let _ = state.source.take();

            state.x11 = X11State::Left;

            if matches!(&state.wayland, &WlState::GotFd(_, _) | &WlState::Dropped) {
                state.wayland = WlState::Finished;
            }
            if matches!(&state.wayland, &WlState::Cancelled | &WlState::Finished) {
                std::mem::drop(state);
                self.active_drag.take();
                self.xdnd_active.store(false, Ordering::Release);
            }
        }
        Ok(())
    }

    pub fn handle_drop(&mut self, drop_msg: ClientMessageData) -> Result<(), ReplyOrIdError> {
        if let Some(drag) = self.active_drag.take() {
            let mut state = drag.state.lock().unwrap();
            self.xdnd_active.store(false, Ordering::Release);

            let data = drop_msg.as_data32();
            trace!("Got XDND drop msg: {:?}", data);

            if state.source.is_none_or(|w| w != data[0]) {
                debug!(
                    "Received XdndDrop from unknown source (got: {}, expected: {:?}), ignoring..",
                    data[0], state.source
                );
                return Ok(());
            }
            let source = data[0];

            let conn = &self.selection.conn;
            let atoms = &self.selection.atoms;

            let failed = || -> Result<(), ReplyOrIdError> {
                drag.pending_transfers.lock().unwrap().remove(&drag.target);

                let data = [*drag.target, 0, 0, 0, 0];
                conn.send_event(
                    false,
                    source,
                    EventMask::NO_EVENT,
                    ClientMessageEvent::new(32, source, atoms.XdndFinished, data),
                )?;
                conn.flush()?;

                Ok(())
            };

            if matches!(&state.wayland, &WlState::Cancelled | &WlState::Finished) {
                failed()?;
                return Ok(());
            }

            if state.version >= 1 {
                state.last_timestamp = Some(data[2]);
            }

            if matches!(&state.wayland, &WlState::GotFd(_, _)) {
                let mut wl_state = WlState::Dropped;
                std::mem::swap(&mut state.wayland, &mut wl_state);

                let WlState::GotFd(mime_type, fd) = wl_state else {
                    unreachable!()
                };

                drag.pending_transfers
                    .lock()
                    .unwrap()
                    .insert(*drag.target, (drag.target.clone(), fd));

                let Some(atom) = atom_from_mime(&mime_type, &self.selection.conn, &self.selection.atoms)?
                else {
                    warn!("Unable to determine atom for mime_type ({})", mime_type);

                    failed()?;
                    return Ok(());
                };

                conn.convert_selection(
                    drag.owner,
                    atoms.XdndSelection,
                    atom,
                    atoms._WL_SELECTION,
                    state.last_timestamp.unwrap_or(x11rb::CURRENT_TIME),
                )?;
                conn.flush()?;
            }

            state.x11 = X11State::Dropped;
        }
        Ok(())
    }
}

impl DndAction {
    fn from_x(atom: Atom, atoms: &Atoms) -> DndAction {
        match atom {
            x if x == atoms.XdndActionCopy => DndAction::Copy,
            x if x == atoms.XdndActionMove => DndAction::Move,
            x if x == atoms.XdndActionAsk => DndAction::Ask,
            _ => DndAction::None,
        }
    }

    fn to_x(self, atoms: &Atoms) -> Atom {
        match self {
            DndAction::Copy => atoms.XdndActionCopy,
            DndAction::Move => atoms.XdndActionMove,
            DndAction::Ask => atoms.XdndActionAsk,
            DndAction::None => AtomEnum::NONE.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum X11State {
    Active,
    Left,
    Dropped,
    Finished,
}

#[derive(Debug)]
enum WlState {
    Active,
    Cancelled,
    Dropped,
    GotFd(String, OwnedFd),
    Finished,
}

#[derive(Debug)]
struct XwmSourceState {
    x11: X11State,
    wayland: WlState,
    source: Option<X11Window>,

    mapped: bool,
    pending_configure: Option<(u16, u16)>,

    active_action: DndAction,
    last_timestamp: Option<u32>,
    version: u32,

    metadata: SourceMetadata,
}

#[derive(Debug)]
pub struct XwmActiveDrag {
    target: OwnedX11Window,
    owner: X11Window,

    state: Arc<Mutex<XwmSourceState>>,
    pending_transfers: Arc<Mutex<HashMap<X11Window, (OwnedX11Window, OwnedFd)>>>,
}

#[derive(Debug)]
pub struct XwmDndSource {
    xwm: XwmId,
    conn: Weak<RustConnection>,
    atoms: Atoms,

    target: OwnedX11Window,

    state: Arc<Mutex<XwmSourceState>>,
    pending_transfers: Arc<Mutex<HashMap<X11Window, (OwnedX11Window, OwnedFd)>>>,
    xdnd_active: Arc<AtomicBool>,
}

impl XwmDndSource {
    /// The grab ended while the X11 client had left the proxy (it sent XdndLeave, e.g. because the drag was cancelled
    /// with Escape): no XdndDrop will come for this drag any more, so it is over on the X11 side.
    ///
    /// Without this the drag stays active until the next XdndSelection change: override-redirect windows are skipped by
    /// `X11Surface::surface_under` (menus let the pointer through), and the same client taking XdndSelection again for its
    /// next drag is ignored.
    fn grab_ended(&self, state: &mut XwmSourceState) {
        if state.x11 == X11State::Left {
            state.x11 = X11State::Finished;
            self.xdnd_active.store(false, Ordering::Release);
        }
    }
}

impl Drop for XwmDndSource {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.upgrade() {
            let mut state = self.state.lock().unwrap();
            state.wayland = WlState::Finished;
            if state.mapped {
                let _ = conn.unmap_window(*self.target);
                let _ = conn.flush();
            }
        }
    }
}

impl IsAlive for XwmDndSource {
    fn alive(&self) -> bool {
        let state = self.state.lock().unwrap();
        !matches!(&state.wayland, &WlState::Cancelled)
            || (matches!(&state.wayland, &WlState::Finished) && state.x11 != X11State::Finished)
    }
}

impl Source for XwmDndSource {
    fn metadata(&self) -> Option<crate::input::dnd::SourceMetadata> {
        Some(self.state.lock().unwrap().metadata.clone())
    }

    fn choose_action(&self, action: DndAction) {
        self.state.lock().unwrap().active_action = action;
    }

    fn send(&self, mime_type: &str, fd: OwnedFd) {
        let mut state = self.state.lock().unwrap();
        if !state.metadata.mime_types.iter().any(|m| m == mime_type) {
            return;
        }

        if let Err(err) = rustix::fs::fcntl_setfl(&fd, OFlags::WRONLY | OFlags::NONBLOCK) {
            warn!(?err, "Failed to restrict file descriptor");
        }

        if let Some(conn) = self.conn.upgrade() {
            trace!("XDND Transfer from xwayland -> wayland");

            if let Some(timestamp) = state.last_timestamp {
                let Ok(Some(atom)) = atom_from_mime(mime_type, &conn, &self.atoms) else {
                    warn!("Unable to determine atom for mime_type ({})", mime_type);
                    return;
                };

                self.pending_transfers
                    .lock()
                    .unwrap()
                    .insert(*self.target, (self.target.clone(), fd));

                if let Err(err) = conn.convert_selection(
                    *self.target,
                    self.atoms.XdndSelection,
                    atom,
                    self.atoms._WL_SELECTION,
                    timestamp,
                ) {
                    warn!("Failed to request selection for Dnd drop: {:?}", err);
                    self.pending_transfers.lock().unwrap().remove(&self.target);
                    return;
                }
            } else {
                state.wayland = WlState::GotFd(mime_type.to_string(), fd);
            }

            let _ = conn.flush();
        }
    }

    fn drop_performed(&self) {
        let mut state = self.state.lock().unwrap();
        state.wayland = WlState::Dropped;
        self.grab_ended(&mut state);
    }

    fn cancel(&self) {
        let mut state = self.state.lock().unwrap();
        state.wayland = WlState::Cancelled;
        self.grab_ended(&mut state);

        if state.x11 == X11State::Dropped {
            if let (Some(conn), Some(source)) = (self.conn.upgrade(), state.source) {
                trace!("XDND cancelled, but already dropped, sending XDNDFinished");

                let data = [*self.target, 0, 0, 0, 0];
                if let Err(err) = conn.send_event(
                    false,
                    source,
                    EventMask::NO_EVENT,
                    ClientMessageEvent::new(32, source, self.atoms.XdndFinished, data),
                ) {
                    warn!("Failed to send XdndFinished: {:?}", err);
                }
                let _ = conn.flush();
            }
            state.x11 = X11State::Finished;
        }
    }

    fn finished(&self) {
        let mut state = self.state.lock().unwrap();

        if state.x11 == X11State::Dropped {
            if let (Some(conn), Some(source)) = (self.conn.upgrade(), state.source) {
                trace!("XDND done, sending XDNDFinished");

                let data = [
                    *self.target,
                    (1 << 0),
                    state.active_action.to_x(&self.atoms),
                    0,
                    0,
                ];
                if let Err(err) = conn.send_event(
                    false,
                    source,
                    EventMask::NO_EVENT,
                    ClientMessageEvent::new(32, source, self.atoms.XdndFinished, data),
                ) {
                    warn!("Failed to send XdndFinished: {:?}", err);
                }
                let _ = conn.flush();
            }
            state.x11 = X11State::Finished;
        }

        state.wayland = WlState::Finished;
    }

    fn is_client_local(&self, target: &dyn Any) -> bool {
        target.downcast_ref::<XwmId>().is_some_and(|id| *id == self.xwm)
    }
}

#[derive(Debug)]
struct XwmOfferState {
    target: X11Window,
    proxy: Option<X11Window>,

    preferred_action: DndAction,
    supported_actions: SmallVec<[DndAction; 3]>,

    pos_pending: bool,
    last_pos: Point<f64, Logical>,
    pos_cached: Option<Point<f64, Logical>>,

    client_accepts: bool,
    active: bool,
    dropped: bool,
}

pub struct XwmActiveOffer {
    state: Arc<Mutex<XwmOfferState>>,
    source: Arc<dyn Source>,
}

impl fmt::Debug for XwmActiveOffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XwmActiveOffer")
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

/// Type implementing [`OfferData`](crate::input::dnd::OfferData) for XDND based DnD sources
#[derive(Debug)]
pub struct XwmOfferData<S: Source> {
    state: Arc<Mutex<XwmOfferState>>,
    source: Arc<S>,
}

/// Whether a grab's focus is a surface of the XWayland instance `id`.
fn started_on_xwayland<F: WaylandFocus>(focus: Option<&(F, Point<f64, Logical>)>, id: XwmId) -> bool {
    focus
        .and_then(|(focus, _)| focus.wl_surface())
        .and_then(|surface| surface.client())
        .is_some_and(|client| {
            client
                .get_data::<XWaylandClientData>()
                .and_then(|data| data.user_data().get::<XwmId>())
                == Some(&id)
        })
}

fn get_proxy_window(
    window: X11Window,
    conn: &impl ConnectionExt,
    atoms: &Atoms,
) -> Result<Option<X11Window>, ConnectionError> {
    if let Some(prop) = conn
        .get_property(false, window, atoms.XdndProxy, AtomEnum::WINDOW, 0, 1)?
        .reply_unchecked()?
    {
        if let Some(proxy) = prop.value32().and_then(|mut val| val.next()) {
            if let Some(prop) = conn
                .get_property(false, proxy, atoms.XdndProxy, AtomEnum::WINDOW, 0, 1)?
                .reply_unchecked()?
            {
                if let Some(verify) = prop.value32().and_then(|mut val| val.next()) {
                    if proxy == verify {
                        return Ok(Some(proxy));
                    }
                }
            }
        }
    }

    Ok(None)
}

impl<D: XwmHandler + SeatHandler> DndFocus<D> for X11Surface {
    type OfferData<S>
        = XwmOfferData<S>
    where
        S: Source;

    fn enter<S: Source>(
        &self,
        data: &mut D,
        _dh: &DisplayHandle,
        source: Arc<S>,
        seat: &Seat<D>,
        location: Point<f64, Logical>,
        _serial: &Serial,
    ) -> Option<Self::OfferData<S>> {
        let xwm_id = self.xwm_id()?;
        let xwm = data.xwm_state(xwm_id);

        if source.is_client_local(&xwm_id) {
            if let Some(active_drag) = xwm.dnd.active_drag.as_mut() {
                trace!("XDND grab entered X11Surface, unmapping proxy");
                let mut drag_state = active_drag.state.lock().unwrap();
                if drag_state.mapped {
                    xwm.conn
                        .unmap_window(*active_drag.target)
                        .inspect_err(|err| warn!("Unable to unmap proxy dnd window: {}", err))
                        .ok()?;
                    let _ = xwm.conn.flush();
                    drag_state.last_timestamp = None;
                    drag_state.mapped = false;
                }
            }

            None
        } else {
            trace!("Wayland dnd grab entered Xwayland surface, taking over XDND_SELECTION");
            if xwm.dnd.selection.owner != *xwm.dnd.selection.window {
                xwm.conn
                    .set_selection_owner(*xwm.dnd.selection.window, xwm.atoms.XdndSelection, CURRENT_TIME)
                    .inspect_err(|err| warn!("Failed to take X11 DND_SELECTION: {:?}", err))
                    .ok()?;
            }

            let proxy = get_proxy_window(self.window_id(), &xwm.conn, &self.atoms).ok()?;
            let proxy_or_window = proxy.unwrap_or(self.window_id());

            let prop = xwm
                .conn
                .get_property(false, proxy_or_window, self.atoms.XdndAware, AtomEnum::ANY, 0, 1)
                .ok()?
                .reply()
                .ok()?;
            if prop.type_ != AtomEnum::ATOM.into() {
                return None;
            }
            let client_ver = prop.value32()?.next()?;
            if client_ver < MIN_DND_VERSION {
                return None;
            }
            let version = client_ver.min(DND_VERSION);

            let metadata = source.metadata()?;
            let mime_types = metadata
                .mime_types
                .iter()
                .filter_map(|mime| atom_from_mime(mime, &xwm.conn, &xwm.atoms).transpose())
                .collect::<Result<Vec<_>, _>>()
                .inspect_err(|err| warn!("Failed to convert mime types: {:?}", err))
                .ok()?;

            // A drop still waiting for its target's XdndFinished is over: XdndSelection serves this offer
            // from here on, so that target could no longer get its own data. It ends before this offer's
            // XdndEnter, which may go to the same window.
            if xwm
                .dnd
                .active_offer
                .as_ref()
                .is_some_and(|active| active.state.lock().unwrap().dropped)
            {
                xwm.dnd.end_waiting_drop();
            }

            let mut enter_data = [
                *xwm.dnd.selection.window,
                version << 24,
                AtomEnum::NONE.into(),
                AtomEnum::NONE.into(),
                AtomEnum::NONE.into(),
            ];
            for (i, mime_type) in mime_types.iter().take(3).enumerate() {
                enter_data[i + 2] = *mime_type;
            }

            if mime_types.len() > 3 {
                enter_data[1] |= 1;
                xwm.conn
                    .change_property32(
                        PropMode::REPLACE,
                        *xwm.dnd.selection.window,
                        self.atoms.XdndTypeList,
                        AtomEnum::ATOM,
                        &mime_types,
                    )
                    .inspect_err(|err| warn!("Failed to update DND_TYPE_LIST: {:?}", err))
                    .ok()?;
            } else {
                xwm.conn
                    .delete_property(*xwm.dnd.selection.window, self.atoms.XdndTypeList)
                    .inspect_err(|err| warn!("Failed to update DND_TYPE_LIST: {:?}", err))
                    .ok()?;
            }
            xwm.dnd.selection.mime_types = metadata.mime_types;

            trace!("Sending XdndEnter: {:?}", enter_data);
            xwm.conn
                .send_event(
                    false,
                    proxy_or_window,
                    EventMask::NO_EVENT,
                    ClientMessageEvent::new(32, self.window_id(), self.atoms.XdndEnter, enter_data),
                )
                .inspect_err(|err| warn!("Failed to send DND_ENTER event: {:?}", err))
                .ok()?;

            let mut offer = XwmOfferData {
                state: Arc::new(Mutex::new(XwmOfferState {
                    target: self.window_id(),
                    proxy,

                    preferred_action: if metadata.dnd_actions.contains(&DndAction::Copy) {
                        DndAction::Copy
                    } else {
                        DndAction::None
                    },
                    supported_actions: metadata.dnd_actions,

                    last_pos: location,
                    pos_pending: false,
                    pos_cached: None,

                    client_accepts: false,
                    active: true,
                    dropped: false,
                })),
                source,
            };
            xwm.dnd.active_offer = Some(XwmActiveOffer {
                state: offer.state.clone(),
                source: offer.source.clone(),
            });

            DndFocus::motion(self, data, Some(&mut offer), seat, location, InputTime::now());
            Some(offer)
        }
    }

    fn motion<S: Source>(
        &self,
        data: &mut D,
        offer: Option<&mut XwmOfferData<S>>,
        _seat: &Seat<D>,
        location: Point<f64, Logical>,
        _time: InputTime,
    ) {
        let Some(offer) = offer else { return };
        let mut state = offer.state.lock().unwrap();
        if state.dropped {
            return;
        }

        let preferred = {
            if state.pos_pending {
                state.pos_cached = Some(location);
                return;
            }
            state.pos_pending = true;
            state.preferred_action
        };

        let Some(xwm_id) = self.xwm_id() else { return };
        let xwm = data.xwm_state(xwm_id);
        let location = (self.last_configure().loc + location.to_i32_round())
            .to_client_precise_round::<_, i32>(xwm.client_scale.load(Ordering::Acquire));

        let data = [
            *xwm.dnd.selection.window,
            0,
            ((location.x << 16) | location.y) as u32,
            CURRENT_TIME,
            preferred.to_x(&xwm.atoms),
        ];

        if let Err(err) = xwm.conn.send_event(
            false,
            state.proxy.unwrap_or(self.window_id()),
            EventMask::NO_EVENT,
            ClientMessageEvent::new(32, self.window_id(), self.atoms.XdndPosition, data),
        ) {
            warn!("Failed to send DND_POSITION event: {:?}", err);
        }
        let _ = xwm.conn.flush();
    }

    fn leave<S: Source>(&self, data: &mut D, offer: Option<&mut XwmOfferData<S>>, _seat: &Seat<D>) {
        let Some(xwm_id) = self.xwm_id() else { return };
        let Some(offer) = offer else {
            // remap the proxy
            trace!("XDND grab left X11Surface, remapping proxy");

            let xwm = data.xwm_state(xwm_id);
            if let Some(active_drag) = xwm.dnd.active_drag.as_mut() {
                let mut drag_state = active_drag.state.lock().unwrap();
                if !drag_state.mapped {
                    if let Err(err) = xwm.conn.map_window(*active_drag.target) {
                        warn!("Unable to map proxy dnd window: {}", err);
                        return;
                    }

                    let mut configure = ConfigureWindowAux::new().stack_mode(StackMode::ABOVE);
                    if let Some((w, h)) = drag_state.pending_configure.take() {
                        configure = configure.width(Some(w as u32)).height(Some(h as u32));
                    }
                    if let Err(err) = xwm.conn.configure_window(*active_drag.target, &configure) {
                        warn!("Unable to configure proxy dnd window: {}", err);
                        return;
                    }
                    drag_state.mapped = true;
                }
            }

            return;
        };

        let state = offer.state.lock().unwrap();
        if !state.dropped {
            let xwm = data.xwm_state(xwm_id);
            let data = [*xwm.dnd.selection.window, 0, 0, 0, 0];
            trace!("Sending XdndLeave: {:?}", data);

            if let Err(err) = xwm.conn.send_event(
                false,
                state.proxy.unwrap_or(self.window_id()),
                EventMask::NO_EVENT,
                ClientMessageEvent::new(32, self.window_id(), self.atoms.XdndLeave, data),
            ) {
                warn!("Failed to send DND_LEAVE event: {:?}", err);
            }
            let _ = xwm.conn.flush();

            xwm.dnd.active_offer = None;
        }
    }

    fn drop<S: Source>(&self, data: &mut D, offer: Option<&mut XwmOfferData<S>>, _seat: &Seat<D>) {
        let Some(xwm_id) = self.xwm_id() else { return };
        let xwm = data.xwm_state(xwm_id);

        let Some(offer) = offer else {
            // the x11 source was presumably dropped on an x11 window, so we are done here.
            if let Some(active_drag) = xwm.dnd.active_drag.take() {
                active_drag.state.lock().unwrap().x11 = X11State::Finished;
                xwm.dnd.xdnd_active.store(false, Ordering::Release);
            }
            return;
        };

        let mut state = offer.state.lock().unwrap();
        state.dropped = true;
        // The target has `DROP_TIMEOUT` to finish the drop, and first to answer a pending position. Only
        // the active offer's drop is waited for (`handle_finished`): an offer no longer active (its target
        // went away, or a second drag's offer took its place) has nothing to time out.
        if xwm
            .dnd
            .active_offer
            .as_ref()
            .is_some_and(|active| Arc::ptr_eq(&active.state, &offer.state))
        {
            xwm.dnd.arm_drop_timeout(xwm.servertime_counter);
        }
        if state.pos_pending {
            return;
        }

        offer.source.drop_performed();

        let data = [*xwm.dnd.selection.window, 0, CURRENT_TIME, 0, 0];
        trace!("Sending XdndDrop: {:?}", data);
        if let Err(err) = xwm.conn.send_event(
            false,
            state.proxy.unwrap_or(self.window_id()),
            EventMask::NO_EVENT,
            ClientMessageEvent::new(32, self.window_id(), self.atoms.XdndDrop, data),
        ) {
            warn!("Failed to send DND_DROP event: {:?}", err);
        }
        let _ = xwm.conn.flush();
    }
}

impl<S: Source> OfferData for XwmOfferData<S> {
    fn disable(&self) {
        self.state.lock().unwrap().active = false;
    }

    fn drop(&self) {
        self.state.lock().unwrap().dropped = true;
    }

    fn validated(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.active && state.preferred_action != DndAction::None && state.client_accepts
    }
}
