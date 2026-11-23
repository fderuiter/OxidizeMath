//! Accessibility, shortcut registration, and button sizing helpers for [`super::PlaybackTransport`].

use super::PlaybackAction;
use crate::accessibility::AccessibleHoverText;
use crate::async_sim::TimelineMetadata;
use crate::framework::InputMode;
use eframe::egui;

pub(super) fn register_shortcut_metadata(
    ctx: &egui::Context,
    can_toggle: bool,
    can_step: bool,
    can_step_back: bool,
) {
    use egui_plot::commands::{CommandRegistryData, CommandTrigger};
    if can_toggle {
        CommandRegistryData::register(
            ctx,
            "Play / Pause",
            "Toggle simulation playback or pause",
            CommandTrigger::Key(egui::Key::Space),
            true,
            "Simulation Playback",
        );
    }
    if can_step {
        CommandRegistryData::register(
            ctx,
            "Step Forward",
            "Advance paused simulation by one deterministic step",
            CommandTrigger::Key(egui::Key::Period),
            true,
            "Simulation Playback",
        );
    }
    if can_step_back {
        CommandRegistryData::register(
            ctx,
            "Step Back",
            "Rewind paused simulation by one deterministic step",
            CommandTrigger::Key(egui::Key::Comma),
            true,
            "Simulation Playback",
        );
        CommandRegistryData::register(
            ctx,
            "Previous Keyframe",
            "Jump backward to the previous checkpoint keyframe",
            CommandTrigger::Key(egui::Key::OpenBracket),
            true,
            "Simulation Playback",
        );
    }
}

pub(super) fn format_announcement(
    tool_name: &str,
    is_in_thread: bool,
    timeline: Option<TimelineMetadata>,
    action: PlaybackAction,
) -> String {
    match action {
        PlaybackAction::Play => format!("Playing {tool_name} simulation"),
        PlaybackAction::Pause => format!("Paused {tool_name} simulation"),
        PlaybackAction::Step if is_in_thread => {
            format!("Stepped {tool_name} simulation — step completed")
        }
        PlaybackAction::Step => format!("Stepping {tool_name} simulation"),
        PlaybackAction::StepBack => {
            let target = timeline.map_or(0, |m| m.current_step.saturating_sub(1).max(m.min_step));
            format!("Stepping {tool_name} simulation backward to step {target}")
        }
        PlaybackAction::ScrubTo(step) => {
            format!("Scrubbed {tool_name} simulation to step {step}")
        }
        PlaybackAction::Reset => format!("Reset {tool_name} simulation"),
    }
}

pub(super) fn render_action_button(
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
