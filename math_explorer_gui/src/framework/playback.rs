#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Unified accessible playback transport widget for continuous and episodic simulations.

use crate::async_sim::SimWorkerState;
use eframe::egui;

#[path = "playback_controls.rs"]
mod playback_controls;
use playback_controls::button_min_size;

/// Discrete user action emitted by [`PlaybackTransport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackAction {
    /// Start or resume simulation playback.
    Play,
    /// Pause continuous simulation playback.
    Pause,
    /// Rewind the simulation by one recorded timeline frame (`Comma`).
    StepBack,
    /// Advance the paused simulation by a single deterministic step (`Period`).
    Step,
    /// Scrub directly to a zero-based timeline frame index.
    Scrub(usize),
    /// Reset simulation state to initial baseline conditions.
    Reset,
}

/// Reusable horizontal transport bar providing Play/Pause, opt-in StepBack (`Comma`),
/// Step (`Period`), Reset, and opt-in timeline scrubber controls with `WidgetInfo` roles,
/// `AccessibleHoverText` tooltips, polite ARIA announcements, and 44x44 touch targets.
pub struct PlaybackTransport<'a> {
    pub(super) id_salt: &'a str,
    pub(super) tool_name: &'a str,
    pub(super) is_playing: bool,
    pub(super) is_in_thread: bool,
    pub(super) can_toggle: bool,
    pub(super) step_back_enabled: Option<bool>,
    pub(super) can_step: bool,
    pub(super) can_reset: bool,
    pub(super) scrubber_range: Option<(usize, usize)>,
    pub(super) play_label: &'a str,
    pub(super) pause_label: &'a str,
    pub(super) step_back_label: &'a str,
    pub(super) step_label: &'a str,
    pub(super) reset_label: &'a str,
}

impl<'a> PlaybackTransport<'a> {
    /// Constructs a transport bar synchronized with a background [`SimWorkerState`].
    #[must_use]
    pub fn from_worker_state(id_salt: &'a str, tool_name: &'a str, state: SimWorkerState) -> Self {
        let is_playing = state.is_running();
        let can_toggle = matches!(state, SimWorkerState::Ready | SimWorkerState::Running);
        let can_step = state.is_ready();
        let can_reset = matches!(
            state,
            SimWorkerState::Ready | SimWorkerState::Running | SimWorkerState::Stepping
        );
        Self {
            id_salt,
            tool_name,
            is_playing,
            is_in_thread: false,
            can_toggle,
            step_back_enabled: None,
            can_step,
            can_reset,
            scrubber_range: None,
            play_label: "▶ Play",
            pause_label: "⏸ Pause",
            step_back_label: "⏮ Back",
            step_label: "⏭ Step",
            reset_label: "↺ Reset",
        }
    }

    /// Constructs a transport bar for an in-thread simulation.
    #[must_use]
    pub fn in_thread(id_salt: &'a str, tool_name: &'a str, is_playing: bool) -> Self {
        Self {
            id_salt,
            tool_name,
            is_playing,
            is_in_thread: true,
            can_toggle: true,
            step_back_enabled: None,
            can_step: !is_playing,
            can_reset: true,
            scrubber_range: None,
            play_label: "▶ Play",
            pause_label: "⏸ Pause",
            step_back_label: "⏮ Back",
            step_label: "⏭ Step",
            reset_label: "↺ Reset",
        }
    }

    /// Customizes the visible Play button label (e.g. `"▶ Train"`).
    #[must_use]
    pub fn play_label(mut self, label: &'a str) -> Self {
        self.play_label = label;
        self
    }

    /// Customizes the visible StepBack button label.
    #[must_use]
    pub fn step_back_label(mut self, label: &'a str) -> Self {
        self.step_back_label = label;
        self
    }

    /// Customizes the visible Step button label (e.g. `"⏭ Step Episode"`).
    #[must_use]
    pub fn step_label(mut self, label: &'a str) -> Self {
        self.step_label = label;
        self
    }

    /// Customizes the visible Reset button label (e.g. `"↻ Reset (90° Pulse)"`).
    #[must_use]
    pub fn reset_label(mut self, label: &'a str) -> Self {
        self.reset_label = label;
        self
    }

    /// Opts into displaying the StepBack button (`Comma` shortcut) with the given enabled state.
    #[must_use]
    pub fn can_step_back(mut self, enabled: bool) -> Self {
        self.step_back_enabled = Some(enabled);
        self
    }

    /// Opts into displaying a synchronized timeline frame scrubber over `0..len`.
    #[must_use]
    pub fn timeline_scrubber(mut self, current: usize, len: usize) -> Self {
        self.scrubber_range = Some((current, len));
        self
    }

    /// Overrides whether the Reset control is enabled (e.g. to respawn a faulted worker).
    #[must_use]
    pub fn can_reset(mut self, enabled: bool) -> Self {
        self.can_reset = enabled;
        self
    }

    /// Renders the transport bar and returns any triggered [`PlaybackAction`].
    pub fn show(self, ui: &mut egui::Ui) -> Option<PlaybackAction> {
        let base_id = ui.make_persistent_id(self.id_salt);
        let min_size = button_min_size(ui.ctx());
        let ui_enabled = ui.is_enabled();
        let mut action = None;

        ui.horizontal_wrapped(|ui| {
            action = self.render_button_row(ui, base_id, ui_enabled, min_size);
        });

        if let Some((current, len)) = self.scrubber_range {
            if let Some(scrubbed) =
                self.render_scrubber(ui, base_id, ui_enabled, min_size.y, current, len)
            {
                if action.is_none() {
                    action = Some(scrubbed);
                }
            }
        }

        let mut should_announce = true;
        if ui_enabled {
            self.register_shortcut_metadata(ui.ctx());
            let (resolved, announce) = self.resolve_transport_action(ui, base_id, action);
            action = resolved;
            should_announce = announce;
        } else if let Some(triggered) = action {
            should_announce = crate::app::viewport::compare_sync::emit_playback_broadcast(
                ui.ctx(),
                base_id,
                triggered,
                None,
            );
        }

        if let Some(triggered) = action {
            self.announce_action(ui.ctx(), base_id, ui_enabled, triggered, should_announce);
            ui.ctx().request_repaint();
        }

        action
    }

    fn is_action_applicable(&self, action: PlaybackAction) -> bool {
        match action {
            PlaybackAction::Play => self.can_toggle && !self.is_playing,
            PlaybackAction::Pause => self.can_toggle && self.is_playing,
            PlaybackAction::StepBack => self.step_back_enabled == Some(true),
            PlaybackAction::Step => self.can_step,
            PlaybackAction::Scrub(idx) => self
                .scrubber_range
                .is_some_and(|(_, len)| len > 1 && idx < len),
            PlaybackAction::Reset => self.can_reset,
        }
    }

    fn resolve_transport_action(
        &self,
        ui: &egui::Ui,
        base_id: egui::Id,
        clicked_action: Option<PlaybackAction>,
    ) -> (Option<PlaybackAction>, bool) {
        use crate::app::viewport::compare_sync::{
            claim_pending_sync_key, emit_playback_broadcast, poll_playback_broadcast,
        };
        let ctx = ui.ctx();
        if let Some((bcast_action, announce)) = poll_playback_broadcast(ctx, base_id) {
            if self.is_action_applicable(bcast_action) {
                return (Some(bcast_action), announce);
            }
        }
        if let Some(clicked) = clicked_action {
            let announce = emit_playback_broadcast(ctx, base_id, clicked, None);
            return (Some(clicked), announce);
        }
        if let Some((sync_key_action, key)) =
            claim_pending_sync_key(ctx, self.can_toggle, self.is_playing, self.can_step)
        {
            let announce = emit_playback_broadcast(ctx, base_id, sync_key_action, Some(key));
            return (Some(sync_key_action), announce);
        }
        if let Some(local_key_action) = self.check_keyboard_shortcuts(ui) {
            let key = match local_key_action {
                PlaybackAction::Play | PlaybackAction::Pause => Some(egui::Key::Space),
                PlaybackAction::StepBack => Some(egui::Key::Comma),
                PlaybackAction::Step => Some(egui::Key::Period),
                PlaybackAction::Scrub(_) | PlaybackAction::Reset => None,
            };
            let announce = emit_playback_broadcast(ctx, base_id, local_key_action, key);
            return (Some(local_key_action), announce);
        }
        (None, false)
    }

    fn register_shortcut_metadata(&self, ctx: &egui::Context) {
        use egui_plot::commands::{CommandRegistryData, CommandTrigger};
        if self.can_toggle {
            CommandRegistryData::register(
                ctx,
                "Play / Pause",
                "Toggle simulation playback or pause",
                CommandTrigger::Key(egui::Key::Space),
                true,
                "Simulation Playback",
            );
        }
        if self.step_back_enabled == Some(true) {
            CommandRegistryData::register(
                ctx,
                "Step Backward",
                "Rewind simulation by one recorded timeline frame",
                CommandTrigger::Key(egui::Key::Comma),
                true,
                "Simulation Playback",
            );
        }
        if self.can_step {
            CommandRegistryData::register(
                ctx,
                "Step Forward",
                "Advance paused simulation by one deterministic step",
                CommandTrigger::Key(egui::Key::Period),
                true,
                "Simulation Playback",
            );
        }
    }

    fn check_keyboard_shortcuts(&self, ui: &egui::Ui) -> Option<PlaybackAction> {
        if !super::is_pane_shortcut_allowed(ui) {
            return None;
        }
        if self.can_toggle
            && super::is_pane_activation_shortcut_allowed(ui)
            && ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Space))
        {
            return Some(if self.is_playing {
                PlaybackAction::Pause
            } else {
                PlaybackAction::Play
            });
        }
        if self.step_back_enabled == Some(true)
            && ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Comma))
        {
            return Some(PlaybackAction::StepBack);
        }
        if self.can_step
            && ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Period))
        {
            return Some(PlaybackAction::Step);
        }
        None
    }
}
