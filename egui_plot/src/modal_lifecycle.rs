use egui::{
    Color32, Context, Event, FocusDirection, Frame, Id, Key, LayerId, Modal, Modifiers, Order,
    Popup, RichText, Ui,
};

const MASKED_PANE_INPUT_KEY: &str = "egui_plot_masked_pane_input";
const OPEN_MODAL_LAYERS_KEY: &str = "egui_plot_open_modal_layers";

/// Returns the active top modal layer, ignoring a [`ModalLifecycle`] layer that closed on a
/// strictly earlier pass while `egui::Memory::top_modal_layer` awaits `end_pass` refresh.
#[must_use]
pub fn active_top_modal_layer(ctx: &Context) -> Option<LayerId> {
    let now_pass = ctx.cumulative_pass_nr();
    if let Some(layer) = ctx.memory(|m| m.top_modal_layer()) {
        let state_id = layer.id.with("modal_lifecycle_state");
        let closed_earlier = ctx.data(|d| {
            d.get_temp::<ModalLifecycleState>(state_id)
                .is_some_and(|s| {
                    !s.was_open
                        && s.last_closed_pass
                            .is_none_or(|closed_p| now_pass > closed_p)
                })
        });
        if !closed_earlier {
            return Some(layer);
        }
    }
    ctx.data(|d| {
        d.get_temp::<Vec<LayerId>>(Id::new(OPEN_MODAL_LAYERS_KEY))
            .and_then(|layers| layers.last().copied())
    })
}

#[derive(Clone, Default)]
struct MaskedPaneInput {
    events: Vec<Event>,
    keys_down: Vec<Key>,
}

fn is_pane_scoped_event(ev: &Event) -> bool {
    matches!(
        ev,
        Event::Key { .. }
            | Event::Text(_)
            | Event::Ime(_)
            | Event::Copy
            | Event::Cut
            | Event::Paste(_)
    )
}

fn drain_pane_scoped_input(ctx: &Context) -> MaskedPaneInput {
    ctx.input_mut(|i| {
        let mut kept = Vec::with_capacity(i.events.len());
        let mut stashed = Vec::new();
        for ev in i.events.drain(..) {
            if is_pane_scoped_event(&ev) {
                stashed.push(ev);
            } else {
                kept.push(ev);
            }
        }
        i.events = kept;
        let keys_down = i.keys_down.drain().collect();
        MaskedPaneInput {
            events: stashed,
            keys_down,
        }
    })
}

/// Runs `f` while temporarily masking keyboard, text, IME, and clipboard events when `allow_input`
/// is `false`, stashing them so the topmost [`Order::Foreground`] [`ModalLifecycle`] can unmask them.
pub fn with_masked_pane_input<R>(ctx: &Context, allow_input: bool, f: impl FnOnce() -> R) -> R {
    if allow_input {
        return f();
    }
    let has_pane_input =
        ctx.input(|i| !i.keys_down.is_empty() || i.events.iter().any(is_pane_scoped_event));
    if !has_pane_input {
        return f();
    }

    let stash_id = Id::new(MASKED_PANE_INPUT_KEY);
    let stashed = drain_pane_scoped_input(ctx);
    let prev_stash = ctx.data_mut(|d| {
        let prev = d.get_temp::<MaskedPaneInput>(stash_id);
        d.insert_temp(stash_id, stashed);
        prev
    });

    let out = f();

    let remaining = ctx.data_mut(|d| {
        let rem = d.get_temp::<MaskedPaneInput>(stash_id);
        if let Some(prev) = prev_stash {
            d.insert_temp(stash_id, prev);
        } else {
            d.remove::<MaskedPaneInput>(stash_id);
        }
        rem
    });
    if let Some(rem) = remaining
        && (!rem.events.is_empty() || !rem.keys_down.is_empty())
    {
        ctx.input_mut(|i| {
            i.events.extend(rem.events);
            i.keys_down.extend(rem.keys_down);
        });
    }
    out
}

fn with_unmasked_top_modal_input<R>(ctx: &Context, modal_id: Id, f: impl FnOnce() -> R) -> R {
    let modal_layer = LayerId::new(Order::Foreground, modal_id);
    if active_top_modal_layer(ctx) != Some(modal_layer) {
        return f();
    }
    let stash_id = Id::new(MASKED_PANE_INPUT_KEY);
    let Some(stashed) = ctx.data_mut(|d| {
        let cur = d.get_temp::<MaskedPaneInput>(stash_id);
        d.remove::<MaskedPaneInput>(stash_id);
        cur
    }) else {
        return f();
    };

    ctx.input_mut(|i| {
        i.events.extend(stashed.events);
        i.keys_down.extend(stashed.keys_down);
    });
    let out = f();
    let remaining = drain_pane_scoped_input(ctx);
    ctx.data_mut(|d| d.insert_temp(stash_id, remaining));
    out
}

/// Per-frame lifecycle handle passed to [`ModalLifecycle::show`] content closures.
pub struct ModalFrameContext {
    /// True on the frame the modal transitions from closed to open.
    pub just_opened: bool,
    /// True when this modal is the active topmost modal layer and no popup is open.
    pub is_top_modal: bool,
    close_requested: bool,
}

impl ModalFrameContext {
    /// Request that the modal close at the end of the current frame and restore focus.
    pub fn close(&mut self) {
        self.close_requested = true;
    }
}

#[derive(Clone, Default)]
struct ModalLifecycleState {
    was_open: bool,
    initial_focus_applied: bool,
    last_open_pass: Option<u64>,
    last_closed_pass: Option<u64>,
    prior_focus: Option<Id>,
    pending_focus_restore: Option<Id>,
}

/// Reusable wrapper around [`egui::Modal`] providing initial focus placement,
/// `Tab` focus containment, `Key::Escape` and backdrop dismissal, and focus
/// restoration to the trigger control on close.
pub struct ModalLifecycle {
    id: Id,
    title: Option<RichText>,
    default_width: Option<f32>,
    max_width: Option<f32>,
    max_height: Option<f32>,
    backdrop_color: Option<Color32>,
    frame: Option<Frame>,
}

impl ModalLifecycle {
    /// Create a new modal lifecycle controller keyed by a unique `egui::Id`.
    pub fn new(id: impl Into<Id>) -> Self {
        Self {
            id: id.into(),
            title: None,
            default_width: None,
            max_width: None,
            max_height: None,
            backdrop_color: None,
            frame: None,
        }
    }

    /// Render an optional heading and separator at the top of the modal body.
    pub fn title(mut self, title: impl Into<RichText>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Configure the minimum width of the modal content container.
    pub fn default_width(mut self, width: f32) -> Self {
        self.default_width = Some(width);
        self
    }

    /// Configure the maximum width of the modal content container.
    pub fn max_width(mut self, width: f32) -> Self {
        self.max_width = Some(width);
        self
    }

    /// Configure the maximum height of the modal content container.
    pub fn max_height(mut self, height: f32) -> Self {
        self.max_height = Some(height);
        self
    }

    /// Customize the modal backdrop scrim color.
    pub fn backdrop_color(mut self, color: Color32) -> Self {
        self.backdrop_color = Some(color);
        self
    }

    /// Customize the modal container [`egui::Frame`].
    pub fn frame(mut self, frame: Frame) -> Self {
        self.frame = Some(frame);
        self
    }

    /// Synchronize modal state and render the modal when `*open` is `true`.
    ///
    /// When the modal closes (via `Key::Escape`, backdrop click, [`ModalFrameContext::close`],
    /// or setting `*open = false`), keyboard focus returns to the widget that held focus
    /// prior to opening.
    pub fn show<R>(
        self,
        ctx: &Context,
        open: &mut bool,
        add_contents: impl FnOnce(&mut Ui, &mut ModalFrameContext) -> R,
    ) -> Option<R> {
        let state_id = self.id.with("modal_lifecycle_state");
        if !*open {
            Self::restore_if_previously_open(ctx, self.id, state_id, false);
            return None;
        }

        let prev_top_layer = active_top_modal_layer(ctx);
        let pass_nr = ctx.cumulative_pass_nr();
        let (just_opened, initial_focus_applied, prior_focus) =
            Self::record_open_frame(ctx, self.id, state_id, pass_nr);

        let modal_id = self.id;
        let frame = self.frame.unwrap_or_else(|| Frame::window(&ctx.style()));
        let mut modal = Modal::new(modal_id).frame(frame);
        if let Some(color) = self.backdrop_color {
            modal = modal.backdrop_color(color);
        }

        let modal_layer = LayerId::new(Order::Foreground, modal_id);
        if just_opened || !initial_focus_applied {
            ctx.move_to_top(modal_layer);
            if just_opened && prev_top_layer.is_some_and(|l| l != modal_layer) {
                ctx.request_discard("promote stacked modal layer");
            }
        }
        let cur_top_layer = active_top_modal_layer(ctx);
        let is_top_or_first = cur_top_layer == Some(modal_layer) || cur_top_layer.is_none();

        let (output, should_dismiss) = with_unmasked_top_modal_input(ctx, modal_id, || {
            let can_receive_keys = is_top_or_first && !Popup::is_any_open(ctx);
            let modal_res = modal.show(ctx, |ui| {
                self.render_modal_body(
                    ui,
                    state_id,
                    just_opened,
                    initial_focus_applied,
                    prior_focus,
                    can_receive_keys,
                    add_contents,
                )
            });
            let (output, close_requested) = modal_res.inner;
            let is_top_modal = modal_res.is_top_modal || is_top_or_first;
            let backdrop_dismissed = !just_opened && modal_res.backdrop_response.clicked();
            let ui_closed = modal_res.response.should_close();
            let escape_dismissed = is_top_modal
                && !modal_res.any_popup_open
                && ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape));
            let dismiss = close_requested || backdrop_dismissed || ui_closed || escape_dismissed;
            if close_requested {
                ctx.input_mut(|i| {
                    let _ = i.consume_key(Modifiers::NONE, Key::Enter);
                    let _ = i.consume_key(Modifiers::NONE, Key::Space);
                });
            }
            (output, dismiss)
        });

        if !*open || should_dismiss {
            *open = false;
            Self::restore_if_previously_open(ctx, modal_id, state_id, true);
            ctx.request_repaint();
        }

        Some(output)
    }

    fn render_modal_body<R>(
        self,
        ui: &mut Ui,
        state_id: Id,
        just_opened: bool,
        initial_focus_applied: bool,
        prior_focus: Option<Id>,
        is_top_modal: bool,
        add_contents: impl FnOnce(&mut Ui, &mut ModalFrameContext) -> R,
    ) -> (R, bool) {
        let apply_initial_focus = !initial_focus_applied && !ui.is_sizing_pass();
        if apply_initial_focus {
            if let Some(prev) = prior_focus {
                ui.memory_mut(|m| m.surrender_focus(prev));
            }
            ui.memory_mut(|m| m.move_focus(FocusDirection::Next));
        } else if just_opened && let Some(prev) = prior_focus {
            ui.memory_mut(|m| m.surrender_focus(prev));
        }

        if let Some(w) = self.default_width {
            ui.set_min_width(w);
        }
        if let Some(max_w) = self.max_width {
            ui.set_max_width(max_w);
        }
        if let Some(max_h) = self.max_height {
            ui.set_max_height(max_h);
        }
        if let Some(heading) = self.title {
            ui.heading(heading);
            ui.separator();
        }

        let mut frame_ctx = ModalFrameContext {
            just_opened: apply_initial_focus,
            is_top_modal,
            close_requested: false,
        };
        let output = add_contents(ui, &mut frame_ctx);

        if apply_initial_focus {
            ui.memory_mut(|m| m.move_focus(FocusDirection::None));
            Self::mark_initial_focus_applied(ui.ctx(), state_id);
            ui.ctx().request_repaint();
        }

        (output, frame_ctx.close_requested)
    }

    fn record_open_frame(
        ctx: &Context,
        modal_id: Id,
        state_id: Id,
        pass_nr: u64,
    ) -> (bool, bool, Option<Id>) {
        let mut state = ctx
            .data(|d| d.get_temp::<ModalLifecycleState>(state_id))
            .unwrap_or_default();
        let continuous = state.was_open
            && state
                .last_open_pass
                .is_some_and(|last| pass_nr <= last.saturating_add(1));
        let just_opened = !continuous;
        if just_opened {
            let prior_focus = ctx.memory(|m| m.focused()).or(state.pending_focus_restore);
            state = ModalLifecycleState {
                was_open: true,
                initial_focus_applied: false,
                last_open_pass: Some(pass_nr),
                last_closed_pass: None,
                prior_focus,
                pending_focus_restore: None,
            };
        } else {
            state.last_open_pass = Some(pass_nr);
        }
        let initial_focus_applied = state.initial_focus_applied;
        let prior_focus = state.prior_focus;
        let layer = LayerId::new(Order::Foreground, modal_id);
        ctx.data_mut(|d| {
            d.insert_temp(state_id, state);
            let mut layers = d
                .get_temp::<Vec<LayerId>>(Id::new(OPEN_MODAL_LAYERS_KEY))
                .unwrap_or_default();
            if just_opened {
                layers.retain(|&l| l != layer);
                layers.push(layer);
            } else if !layers.contains(&layer) {
                layers.push(layer);
            }
            d.insert_temp(Id::new(OPEN_MODAL_LAYERS_KEY), layers);
        });
        (just_opened, initial_focus_applied, prior_focus)
    }

    fn mark_initial_focus_applied(ctx: &Context, state_id: Id) {
        ctx.data_mut(|d| {
            let mut state = d
                .get_temp::<ModalLifecycleState>(state_id)
                .unwrap_or_default();
            state.initial_focus_applied = true;
            d.insert_temp(state_id, state);
        });
    }

    fn restore_if_previously_open(
        ctx: &Context,
        modal_id: Id,
        state_id: Id,
        rendered_this_pass: bool,
    ) {
        let layer = LayerId::new(Order::Foreground, modal_id);
        let pass_nr = ctx.cumulative_pass_nr();
        let mut state = ctx.data_mut(|d| {
            if let Some(mut layers) = d.get_temp::<Vec<LayerId>>(Id::new(OPEN_MODAL_LAYERS_KEY)) {
                layers.retain(|&l| l != layer);
                d.insert_temp(Id::new(OPEN_MODAL_LAYERS_KEY), layers);
            }
            d.get_temp::<ModalLifecycleState>(state_id)
                .unwrap_or_default()
        });
        if state.was_open {
            state.was_open = false;
            state.initial_focus_applied = false;
            state.last_open_pass = None;
            state.last_closed_pass = Some(pass_nr);
            let prior = state.prior_focus.take();
            state.pending_focus_restore = prior;
            ctx.data_mut(|d| d.insert_temp(state_id, state));
            if rendered_this_pass {
                if let Some(prev_id) = prior {
                    ctx.memory_mut(|m| m.request_focus(prev_id));
                }
                return;
            }
        }
        Self::flush_pending_focus_restore(ctx, layer, state_id);
    }

    fn flush_pending_focus_restore(ctx: &Context, layer: LayerId, state_id: Id) {
        let Some(prev_id) = ctx.data(|d| {
            d.get_temp::<ModalLifecycleState>(state_id)
                .and_then(|s| s.pending_focus_restore)
        }) else {
            return;
        };
        let stale_modal_top = ctx.memory(|m| m.top_modal_layer()) == Some(layer);
        let currently_unfocused = ctx.memory(|m| m.focused().is_none());
        if stale_modal_top && ctx.current_pass_index() == 0 && currently_unfocused {
            ctx.request_discard("restore focus after modal layer clear");
            return;
        }
        ctx.data_mut(|d| {
            if let Some(mut s) = d.get_temp::<ModalLifecycleState>(state_id) {
                s.pending_focus_restore = None;
                d.insert_temp(state_id, s);
            }
        });
        if currently_unfocused {
            ctx.memory_mut(|m| m.request_focus(prev_id));
        }
    }
}
