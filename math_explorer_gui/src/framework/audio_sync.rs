//! Scoped audio state synchronization, default-muted controls, and divergence silence guards
//! for [`crate::framework::SimulationFramework`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::playback::{scoped_audio_bridge_id, ScopedAudioUiBridge};
use super::SimulationFramework;
use crate::audio::AcousticSettings;
use eframe::egui;

impl SimulationFramework {
    /// Returns `true` if live sonification audio is muted (`true` by default on boot and load).
    #[must_use]
    pub const fn is_audio_muted(&self) -> bool {
        self.audio_controller.is_muted()
    }

    /// Explicitly unmutes audio via user gesture and announces the state change via ARIA.
    pub fn unmute_audio_with_gesture(&mut self) {
        let tool_name = self.active_tool.as_ref().map_or("simulation", |t| t.name());
        self.audio_controller.unmute_from_user_gesture(tool_name);
    }

    /// Mutes audio via user gesture and ramps output gain to zero within `20 ms`.
    pub fn mute_audio(&mut self) {
        let tool_name = self.active_tool.as_ref().map_or("simulation", |t| t.name());
        self.audio_controller.mute_by_user(tool_name);
    }

    /// Updates persisted non-mute acoustic settings on both the active tool and audio controller.
    pub fn set_acoustic_settings(&mut self, settings: AcousticSettings) {
        let clean = settings.sanitized();
        if let Some(ref mut tool) = self.active_tool {
            tool.set_acoustic_settings(clean);
        }
        self.audio_controller.set_settings(clean);
        crate::audio::set_user_master_volume(clean.gain);
        self.bump_framework_revision();
    }

    pub(super) fn begin_frame_audio_bridge(&mut self, ctx: &egui::Context) {
        let settings = self
            .active_tool
            .as_ref()
            .map_or_else(AcousticSettings::default, |t| t.acoustic_settings());
        self.audio_controller.set_settings(settings);
        let bridge = ScopedAudioUiBridge {
            muted: self.audio_controller.is_muted(),
            toggled_mute: false,
            pause_triggered: false,
            settings_changed: false,
            settings,
        };
        ctx.data_mut(|d| {
            d.insert_temp(scoped_audio_bridge_id(), bridge);
        });
    }

    pub(super) fn finish_frame_audio_bridge(&mut self, ctx: &egui::Context) {
        let bridge_opt =
            ctx.data_mut(|d| d.remove_temp::<ScopedAudioUiBridge>(scoped_audio_bridge_id()));
        let tool_name = self.active_tool.as_ref().map_or("simulation", |t| t.name());

        if let Some(bridge) = bridge_opt {
            if bridge.settings_changed {
                if let Some(ref mut tool) = self.active_tool {
                    tool.set_acoustic_settings(bridge.settings);
                }
                self.audio_controller.set_settings(bridge.settings);
                self.bump_framework_revision();
            }
            if bridge.toggled_mute {
                if bridge.muted {
                    self.audio_controller.mute_by_user(tool_name);
                } else {
                    self.audio_controller.unmute_from_user_gesture(tool_name);
                }
            }
            if bridge.pause_triggered {
                self.audio_controller.silence_on_pause();
            }
        }

        let tool_diverged = self
            .active_tool
            .as_ref()
            .is_some_and(|t| !t.has_finite_parameters());
        if self.is_diverged || tool_diverged {
            self.audio_controller.auto_mute_on_divergence(tool_name);
            return;
        }

        let next_frame = self
            .active_tool
            .as_ref()
            .and_then(|t| t.sonification_frame());
        if let Some(ref tool) = self.active_tool {
            self.audio_controller.set_settings(tool.acoustic_settings());
        }
        self.audio_controller.update_frame(next_frame);
    }
}
