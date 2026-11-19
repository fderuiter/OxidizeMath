use egui::{Color32, Context, FocusDirection, Frame, Id, Key, Modal, Modifiers, RichText, Ui};

/// Per-frame lifecycle handle passed to [`ModalLifecycle::show`] content closures.
pub struct ModalFrameContext {
    /// True on the frame the modal transitions from closed to open.
    pub just_opened: bool,
    close_requested: bool,
}

impl ModalFrameContext {
    /// Request that the modal close at the end of the current frame and restore focus.
    pub fn close(&mut self) {
        self.close_requested = true;
    }

    /// Returns whether closure dismissal has been requested during this frame.
    pub fn is_close_requested(&self) -> bool {
        self.close_requested
    }
}

#[derive(Clone, Default)]
struct ModalLifecycleState {
    was_open: bool,
    initial_focus_applied: bool,
    last_open_pass: Option<u64>,
    prior_focus: Option<Id>,
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
            Self::restore_if_previously_open(ctx, state_id);
            return None;
        }

        let pass_nr = ctx.cumulative_pass_nr();
        let (just_opened, initial_focus_applied, prior_focus) =
            Self::record_open_frame(ctx, state_id, pass_nr);

        let frame = self.frame.unwrap_or_else(|| Frame::window(&ctx.style()));
        let mut modal = Modal::new(self.id).frame(frame);
        if let Some(color) = self.backdrop_color {
            modal = modal.backdrop_color(color);
        }

        let default_width = self.default_width;
        let max_width = self.max_width;
        let max_height = self.max_height;
        let title = self.title;

        let modal_res = modal.show(ctx, |ui| {
            let apply_initial_focus = !initial_focus_applied && !ui.is_sizing_pass();
            if apply_initial_focus {
                if let Some(prev) = prior_focus {
                    ui.memory_mut(|m| m.surrender_focus(prev));
                }
                ui.memory_mut(|m| m.move_focus(FocusDirection::Next));
            } else if just_opened && let Some(prev) = prior_focus {
                ui.memory_mut(|m| m.surrender_focus(prev));
            }

            if let Some(w) = default_width {
                ui.set_min_width(w);
            }
            if let Some(max_w) = max_width {
                ui.set_max_width(max_w);
            }
            if let Some(max_h) = max_height {
                ui.set_max_height(max_h);
            }

            if let Some(heading) = title {
                ui.heading(heading);
                ui.separator();
            }

            let mut frame_ctx = ModalFrameContext {
                just_opened: apply_initial_focus,
                close_requested: false,
            };
            let output = add_contents(ui, &mut frame_ctx);

            if apply_initial_focus {
                ui.memory_mut(|m| m.move_focus(FocusDirection::None));
                Self::mark_initial_focus_applied(ui.ctx(), state_id);
                ui.ctx().request_repaint();
            }

            (output, frame_ctx.close_requested)
        });

        let (output, close_requested) = modal_res.inner;
        let is_top_or_first_modal =
            modal_res.is_top_modal || ctx.memory(|m| m.top_modal_layer().is_none());
        let backdrop_dismissed = !just_opened && modal_res.backdrop_response.clicked();
        let ui_closed = modal_res.response.should_close();
        let escape_dismissed = is_top_or_first_modal
            && !modal_res.any_popup_open
            && ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape));

        if !*open || close_requested || backdrop_dismissed || ui_closed || escape_dismissed {
            *open = false;
            Self::restore_if_previously_open(ctx, state_id);
            ctx.request_repaint();
        }

        Some(output)
    }

    fn record_open_frame(ctx: &Context, state_id: Id, pass_nr: u64) -> (bool, bool, Option<Id>) {
        let mut state = ctx
            .data(|d| d.get_temp::<ModalLifecycleState>(state_id))
            .unwrap_or_default();
        let continuous = state.was_open
            && state
                .last_open_pass
                .is_some_and(|last| pass_nr <= last.saturating_add(1));
        let just_opened = !continuous;
        if just_opened {
            let prior_focus = ctx.memory(|m| m.focused());
            state = ModalLifecycleState {
                was_open: true,
                initial_focus_applied: false,
                last_open_pass: Some(pass_nr),
                prior_focus,
            };
        } else {
            state.last_open_pass = Some(pass_nr);
        }
        let initial_focus_applied = state.initial_focus_applied;
        let prior_focus = state.prior_focus;
        ctx.data_mut(|d| d.insert_temp(state_id, state));
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

    fn restore_if_previously_open(ctx: &Context, state_id: Id) {
        let maybe_prior = ctx.data_mut(|d| {
            let mut state = d
                .get_temp::<ModalLifecycleState>(state_id)
                .unwrap_or_default();
            if !state.was_open {
                return None;
            }
            state.was_open = false;
            state.initial_focus_applied = false;
            state.last_open_pass = None;
            let prior = state.prior_focus.take();
            d.insert_temp(state_id, state);
            prior
        });
        if let Some(prev_id) = maybe_prior {
            ctx.memory_mut(|m| m.request_focus(prev_id));
        }
    }
}
