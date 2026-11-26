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

pub(super) fn take_dispatched_playback_action(
    ctx: &egui::Context,
    is_playing: bool,
    can_toggle: bool,
    can_step: bool,
    can_step_back: bool,
    can_reset: bool,
    timeline: Option<TimelineMetadata>,
) -> Option<PlaybackAction> {
    use egui_plot::commands::CommandRegistryData;
    if !CommandRegistryData::has_pending_dispatch(ctx, "Simulation Playback") {
        return None;
    }
    if can_toggle
        && CommandRegistryData::take_dispatched(ctx, "Play / Pause", "Simulation Playback")
    {
        return Some(if is_playing {
            PlaybackAction::Pause
        } else {
            PlaybackAction::Play
        });
    }
    if can_step && CommandRegistryData::take_dispatched(ctx, "Step Forward", "Simulation Playback")
    {
        return Some(PlaybackAction::Step);
    }
    if can_step_back
        && CommandRegistryData::take_dispatched(ctx, "Step Back", "Simulation Playback")
    {
        return Some(PlaybackAction::StepBack);
    }
    if can_step_back {
        if let Some(meta) = timeline {
            if CommandRegistryData::take_dispatched(ctx, "Previous Keyframe", "Simulation Playback")
            {
                return Some(PlaybackAction::ScrubTo(meta.previous_keyframe_step()));
            }
        }
    }
    if can_reset
        && CommandRegistryData::take_dispatched(ctx, "Reset Simulation", "Simulation Playback")
    {
        return Some(PlaybackAction::Reset);
    }
    None
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

pub(super) fn render_step_and_reset_buttons(
    ui: &mut egui::Ui,
    base_id: egui::Id,
    tool_name: &str,
    labels: (&str, &str, &str),
    flags: (bool, bool, bool),
    min_size: egui::Vec2,
) -> Option<PlaybackAction> {
    let mut action = None;
    let specs = [
        (
            "step_back_button_info",
            labels.0,
            format!("Step {tool_name} simulation backward one step (Comma)"),
            flags.0,
            PlaybackAction::StepBack,
        ),
        (
            "step_button_info",
            labels.1,
            format!("Step {tool_name} simulation forward one step (Period)"),
            flags.1,
            PlaybackAction::Step,
        ),
        (
            "reset_button_info",
            labels.2,
            format!("Reset {tool_name} simulation"),
            flags.2,
            PlaybackAction::Reset,
        ),
    ];
    for (salt, label, a11y, enabled, act) in &specs {
        if render_action_button(ui, base_id.with(*salt), label, a11y, *enabled, min_size)
            && action.is_none()
        {
            action = Some(*act);
        }
    }
    action
}

pub(super) fn render_timeline_scrubber(
    ui: &mut egui::Ui,
    base_id: egui::Id,
    tool_name: &str,
    can_step: bool,
    meta: TimelineMetadata,
    min_size: egui::Vec2,
) -> Option<PlaybackAction> {
    let enabled = ui.is_enabled() && can_step && meta.max_step > meta.min_step;
    let mut step_val = meta.current_step.clamp(meta.min_step, meta.max_step);
    let a11y_label = format!(
        "Scrub {tool_name} timeline step ({}..={})",
        meta.min_step, meta.max_step
    );
    let info = egui::WidgetInfo::slider(enabled, step_val as f64, &a11y_label);

    let resp = ui
        .scope(|ui| {
            if min_size.y >= 44.0 {
                ui.spacing_mut().interact_size.x = ui.spacing().interact_size.x.max(44.0);
                ui.spacing_mut().interact_size.y = ui.spacing().interact_size.y.max(44.0);
            }
            ui.add_enabled(
                enabled,
                egui::Slider::new(&mut step_val, meta.min_step..=meta.max_step)
                    .text(format!("Step [{}..{}]", meta.min_step, meta.max_step)),
            )
        })
        .inner;

    resp.widget_info({
        let info = info.clone();
        move || info.clone()
    });
    ui.ctx().data_mut(|d| {
        d.insert_temp(base_id.with("timeline_scrubber_info"), info);
    });
    let changed = resp.accessible_hover_text(&a11y_label).changed();
    (changed && step_val != meta.current_step).then_some(PlaybackAction::ScrubTo(step_val))
}
