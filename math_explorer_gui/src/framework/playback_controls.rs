#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Button row, timeline scrubber slider, and ARIA live announcement helpers for [`super::PlaybackTransport`].

use super::{PlaybackAction, PlaybackTransport};
use crate::accessibility::{announce_status, AccessibleHoverText};
use crate::framework::InputMode;
use eframe::egui;

impl PlaybackTransport<'_> {
    pub(super) fn render_button_row(
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

        if let Some(can_back) = self.step_back_enabled {
            let back_a11y = format!(
                "Step {} simulation backward one frame (Comma)",
                self.tool_name
            );
            if render_action_button(
                ui,
                base_id.with("step_back_button_info"),
                self.step_back_label,
                &back_a11y,
                ui_enabled && can_back,
                min_size,
            ) && action.is_none()
            {
                action = Some(PlaybackAction::StepBack);
            }
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

    pub(super) fn render_scrubber(
        &self,
        ui: &mut egui::Ui,
        base_id: egui::Id,
        ui_enabled: bool,
        min_height: f32,
        current: usize,
        len: usize,
    ) -> Option<PlaybackAction> {
        let max_idx = len.saturating_sub(1);
        let mut frame_idx = current.min(max_idx);
        let enabled = ui_enabled && len > 1;
        let a11y_label = format!(
            "Scrub {} timeline frame ({} of {})",
            self.tool_name, frame_idx, max_idx
        );
        let info = egui::WidgetInfo::slider(enabled, frame_idx as f64, &a11y_label);
        let resp = ui.scope(|ui| {
            if min_height > 0.0 {
                ui.spacing_mut().interact_size.y = ui.spacing().interact_size.y.max(min_height);
            }
            let slider = egui::Slider::new(&mut frame_idx, 0..=max_idx).text("Frame");
            ui.add_enabled(enabled, slider)
        });
        let slider_resp = resp.inner;
        slider_resp.widget_info({
            let info = info.clone();
            move || info.clone()
        });
        ui.ctx().data_mut(|d| {
            d.insert_temp(base_id.with("timeline_scrubber_info"), info);
        });
        let changed = slider_resp.accessible_hover_text(&a11y_label).changed();
        if changed && frame_idx != current {
            Some(PlaybackAction::Scrub(frame_idx))
        } else {
            None
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

    fn format_announcement(&self, action: PlaybackAction) -> String {
        match action {
            PlaybackAction::Play => format!("Playing {} simulation", self.tool_name),
            PlaybackAction::Pause => format!("Paused {} simulation", self.tool_name),
            PlaybackAction::StepBack if self.is_in_thread => {
                format!(
                    "Stepped backward {} simulation — frame restored",
                    self.tool_name
                )
            }
            PlaybackAction::StepBack => format!("Stepping backward {} simulation", self.tool_name),
            PlaybackAction::Step if self.is_in_thread => {
                format!("Stepped {} simulation — step completed", self.tool_name)
            }
            PlaybackAction::Step => format!("Stepping {} simulation", self.tool_name),
            PlaybackAction::Scrub(frame) => {
                format!("Scrubbed {} timeline to frame {}", self.tool_name, frame)
            }
            PlaybackAction::Reset => format!("Reset {} simulation", self.tool_name),
        }
    }

    pub(super) fn announce_action(
        &self,
        ctx: &egui::Context,
        base_id: egui::Id,
        ui_enabled: bool,
        action: PlaybackAction,
        should_announce: bool,
    ) {
        if should_announce {
            let announcement = self.format_announcement(action);
            announce_status(&announcement);
            ctx.data_mut(|d| {
                d.insert_temp(egui::Id::new("aria_live_message"), announcement);
                d.insert_temp(egui::Id::new("aria_live_dispatched"), true);
            });
        }
        if matches!(
            action,
            PlaybackAction::Play
                | PlaybackAction::Pause
                | PlaybackAction::StepBack
                | PlaybackAction::Scrub(_)
        ) {
            ctx.data_mut(|d| {
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
            });
        }
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

pub(super) fn button_min_size(ctx: &egui::Context) -> egui::Vec2 {
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
