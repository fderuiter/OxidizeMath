//! Unified accessible playback transport widget for continuous and episodic simulations.

use crate::accessibility::{announce_status, AccessibleHoverText};
use crate::async_sim::SimWorkerState;
use crate::framework::InputMode;
use eframe::egui;

/// Discrete user action emitted by [`PlaybackTransport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackAction {
    /// Start or resume simulation playback.
    Play,
    /// Pause continuous simulation playback.
    Pause,
    /// Advance the paused simulation by a single deterministic step.
    Step,
    /// Reset simulation state to initial baseline conditions.
    Reset,
}

/// Reusable horizontal transport bar providing Play/Pause, Step, and Reset controls
/// with `WidgetInfo` roles, `AccessibleHoverText` tooltips, polite ARIA announcements,
/// keyboard shortcuts (`Space` and `Period`), and 44x44 touch targets.
pub struct PlaybackTransport<'a> {
    id_salt: &'a str,
    tool_name: &'a str,
    is_playing: bool,
    is_in_thread: bool,
    can_toggle: bool,
    can_step: bool,
    can_reset: bool,
    play_label: &'a str,
    pause_label: &'a str,
    step_label: &'a str,
    reset_label: &'a str,
    audio_settings: Option<&'a mut crate::audio::AcousticSettings>,
}

/// Scoped `egui` temp-data bridge synchronizing [`PlaybackTransport`] audio controls with [`super::SimulationFramework`].
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ScopedAudioUiBridge {
    pub(crate) muted: bool,
    pub(crate) toggled_mute: bool,
    pub(crate) pause_triggered: bool,
    pub(crate) settings_changed: bool,
    pub(crate) settings: crate::audio::AcousticSettings,
}

pub(crate) fn scoped_audio_bridge_id() -> egui::Id {
    egui::Id::new("SCOPED_FRAMEWORK_AUDIO_BRIDGE")
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
            can_step,
            can_reset,
            play_label: "▶ Play",
            pause_label: "⏸ Pause",
            step_label: "⏭ Step",
            reset_label: "↺ Reset",
            audio_settings: None,
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
            can_step: !is_playing,
            can_reset: true,
            play_label: "▶ Play",
            pause_label: "⏸ Pause",
            step_label: "⏭ Step",
            reset_label: "↺ Reset",
            audio_settings: None,
        }
    }

    /// Customizes the visible Play button label (e.g. `"▶ Train"`).
    #[must_use]
    pub fn play_label(mut self, label: &'a str) -> Self {
        self.play_label = label;
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

    /// Overrides whether the Reset control is enabled (e.g. to respawn a faulted worker).
    #[must_use]
    pub fn can_reset(mut self, enabled: bool) -> Self {
        self.can_reset = enabled;
        self
    }

    /// Attaches default-muted sonification controls (Mute/Unmute toggle and Gain slider)
    /// bound to the tool's persisted [`crate::audio::AcousticSettings`].
    #[must_use]
    pub fn with_sonification(mut self, settings: &'a mut crate::audio::AcousticSettings) -> Self {
        self.audio_settings = Some(settings);
        self
    }

    /// Renders the transport bar and returns any triggered [`PlaybackAction`].
    pub fn show(mut self, ui: &mut egui::Ui) -> Option<PlaybackAction> {
        let base_id = ui.make_persistent_id(self.id_salt);
        let min_size = button_min_size(ui.ctx());
        let ui_enabled = ui.is_enabled();
        let mut action = None;

        ui.horizontal_wrapped(|ui| {
            action = self.render_core_buttons(ui, base_id, ui_enabled, min_size);
            if let Some(settings) = self.audio_settings.take() {
                super::playback_audio::render_audio_controls(
                    ui,
                    base_id,
                    self.tool_name,
                    ui_enabled,
                    min_size,
                    settings,
                );
            }
        });

        if ui_enabled {
            self.register_shortcut_metadata(ui.ctx());
            if action.is_none() {
                action = self.check_keyboard_shortcuts(ui);
            }
        }

        if let Some(triggered) = action {
            if triggered == PlaybackAction::Pause {
                super::playback_audio::notify_pause_triggered(ui.ctx());
            }
            self.announce_action(ui.ctx(), base_id, ui_enabled, triggered);
            ui.ctx().request_repaint();
        }

        action
    }

    fn render_core_buttons(
        &self,
        ui: &mut egui::Ui,
        base_id: egui::Id,
        ui_enabled: bool,
        min_size: egui::Vec2,
    ) -> Option<PlaybackAction> {
        let mut action = None;
        if self.render_toggle_button(ui, base_id, min_size) {
            action = Some(if self.is_playing {
                PlaybackAction::Pause
            } else {
                PlaybackAction::Play
            });
        }

        let step_a11y = format!(
            "Step {} simulation forward one step (Period)",
            self.tool_name
        );
        if render_action_button(
            ui,
            base_id.with("step_button_info"),
            self.step_label,
            &step_a11y,
            ui_enabled && self.can_step,
            min_size,
        ) && action.is_none()
        {
            action = Some(PlaybackAction::Step);
        }

        let reset_a11y = format!("Reset {} simulation", self.tool_name);
        if render_action_button(
            ui,
            base_id.with("reset_button_info"),
            self.reset_label,
            &reset_a11y,
            ui_enabled && self.can_reset,
            min_size,
        ) && action.is_none()
        {
            action = Some(PlaybackAction::Reset);
        }
        action
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

    fn render_toggle_button(
        &self,
        ui: &mut egui::Ui,
        base_id: egui::Id,
        min_size: egui::Vec2,
    ) -> bool {
        let enabled = ui.is_enabled() && self.can_toggle;
        let text = if self.is_playing {
            self.pause_label
        } else {
            self.play_label
        };
        let a11y_label = if self.is_playing {
            format!("Pause {} simulation (Space)", self.tool_name)
        } else {
            format!("Play {} simulation (Space)", self.tool_name)
        };
        let info = egui::WidgetInfo::selected(
            egui::WidgetType::Button,
            enabled,
            self.is_playing,
            &a11y_label,
        );
        let resp = ui.add_enabled(enabled, egui::Button::new(text).min_size(min_size));
        resp.widget_info({
            let info = info.clone();
            move || info.clone()
        });
        ui.ctx().data_mut(|d| {
            d.insert_temp(base_id.with("play_button_info"), info);
        });
        resp.accessible_hover_text(&a11y_label).clicked()
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
        if self.can_step
            && ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Period))
        {
            return Some(PlaybackAction::Step);
        }
        None
    }

    fn announce_action(
        &self,
        ctx: &egui::Context,
        base_id: egui::Id,
        ui_enabled: bool,
        action: PlaybackAction,
    ) {
        let announcement = match action {
            PlaybackAction::Play => format!("Playing {} simulation", self.tool_name),
            PlaybackAction::Pause => format!("Paused {} simulation", self.tool_name),
            PlaybackAction::Step if self.is_in_thread => {
                format!("Stepped {} simulation — step completed", self.tool_name)
            }
            PlaybackAction::Step => format!("Stepping {} simulation", self.tool_name),
            PlaybackAction::Reset => format!("Reset {} simulation", self.tool_name),
        };
        announce_status(&announcement);
        ctx.data_mut(|d| {
            d.insert_temp(egui::Id::new("aria_live_message"), announcement);
            d.insert_temp(egui::Id::new("aria_live_dispatched"), true);
            if matches!(action, PlaybackAction::Play | PlaybackAction::Pause) {
                let next_playing = matches!(action, PlaybackAction::Play);
                let next_label = if next_playing {
                    format!("Pause {} simulation (Space)", self.tool_name)
                } else {
                    format!("Play {} simulation (Space)", self.tool_name)
                };
                d.insert_temp(
                    base_id.with("play_button_info"),
                    egui::WidgetInfo::selected(
                        egui::WidgetType::Button,
                        ui_enabled && self.can_toggle,
                        next_playing,
                        next_label,
                    ),
                );
            }
        });
    }
}

fn render_action_button(
    ui: &mut egui::Ui,
    info_id: egui::Id,
    label: &str,
    a11y_label: &str,
    enabled: bool,
    min_size: egui::Vec2,
) -> bool {
    let info = egui::WidgetInfo::labeled(egui::WidgetType::Button, enabled, a11y_label);
    let resp = ui.add_enabled(enabled, egui::Button::new(label).min_size(min_size));
    resp.widget_info({
        let info = info.clone();
        move || info.clone()
    });
    ui.ctx().data_mut(|d| {
        d.insert_temp(info_id, info);
    });
    resp.accessible_hover_text(a11y_label).clicked()
}

fn button_min_size(ctx: &egui::Context) -> egui::Vec2 {
    let is_touch = ctx.data(|d| {
        let mode = d
            .get_temp::<InputMode>(egui::Id::new("INPUT_MODE"))
            .unwrap_or(InputMode::Mouse);
        let touch_flag = d
            .get_temp::<bool>(egui::Id::new("INPUT_MODE_TOUCH"))
            .unwrap_or(false);
        mode == InputMode::Touch || touch_flag
    });
    if is_touch {
        egui::vec2(44.0, 44.0)
    } else {
        egui::Vec2::ZERO
    }
}
