//! Timeline scrubber slider, step-back button, and `Fork Branch` controls for [`super::PlaybackTransport`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::playback::{button_min_size, render_action_button, PlaybackAction, PlaybackTransport};
use super::timeline_branch::{
    announce_branch_cleared, announce_branch_forked, TimelineBranchAction, TimelineTransportState,
};
use crate::accessibility::{announce_live_status, AccessibleHoverText};
use crate::app::palette::SemanticPalette;
use eframe::egui;

impl<'a> PlaybackTransport<'a> {
    /// Renders the transport bar augmented with step-back, `Fork Branch` toggle, and timeline scrubber.
    pub fn show_with_timeline(
        self,
        ui: &mut egui::Ui,
        timeline: TimelineTransportState,
    ) -> (Option<PlaybackAction>, Option<TimelineBranchAction>) {
        let base_id = ui.make_persistent_id(self.id_salt);
        let min_size = button_min_size(ui.ctx());
        let ui_enabled = ui.is_enabled();
        let tool_name = self.tool_name;
        let can_step_back = !self.is_playing && timeline.can_step_back();

        let mut branch_action = None;
        if ui_enabled {
            register_timeline_shortcuts(ui.ctx(), can_step_back);
            if check_step_back_shortcut(ui, can_step_back) {
                branch_action = Some(TimelineBranchAction::StepBack);
            }
        }

        let play_action = self.show(ui);

        ui.horizontal_wrapped(|ui| {
            let step_back_a11y =
                format!("Step {tool_name} simulation backward one checkpoint (Comma)");
            if render_action_button(
                ui,
                base_id.with("step_back_button_info"),
                "⏮ Back",
                &step_back_a11y,
                ui_enabled && can_step_back,
                min_size,
            ) && branch_action.is_none()
            {
                branch_action = Some(TimelineBranchAction::StepBack);
            }

            if render_fork_toggle_button(ui, base_id, tool_name, timeline, ui_enabled, min_size)
                && branch_action.is_none()
            {
                branch_action = Some(TimelineBranchAction::ToggleForkBranch);
            }
        });

        if let Some(scrubbed) =
            render_timeline_scrubber(ui, base_id, tool_name, timeline, ui_enabled)
        {
            if branch_action.is_none() {
                branch_action = Some(TimelineBranchAction::ScrubTo(scrubbed));
            }
        }

        if let Some(triggered) = branch_action {
            dispatch_timeline_announcement(ui.ctx(), tool_name, timeline, triggered);
            ui.ctx().request_repaint();
        }

        (play_action, branch_action)
    }
}

fn register_timeline_shortcuts(ctx: &egui::Context, can_step_back: bool) {
    use egui_plot::commands::{CommandRegistryData, CommandTrigger};
    if can_step_back {
        CommandRegistryData::register(
            ctx,
            "Step Backward",
            "Rewind simulation to the previous timeline checkpoint",
            CommandTrigger::Key(egui::Key::Comma),
            true,
            "Simulation Playback",
        );
    }
}

fn check_step_back_shortcut(ui: &egui::Ui, can_step_back: bool) -> bool {
    can_step_back
        && super::is_pane_shortcut_allowed(ui)
        && ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Comma))
}

fn render_fork_toggle_button(
    ui: &mut egui::Ui,
    base_id: egui::Id,
    tool_name: &str,
    timeline: TimelineTransportState,
    ui_enabled: bool,
    min_size: egui::Vec2,
) -> bool {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    let text = if timeline.fork_active {
        egui::RichText::new("⑂ Forked Twin")
            .strong()
            .color(palette.accent)
    } else {
        egui::RichText::new("⑂ Fork Branch")
    };
    let a11y_label = if timeline.fork_active {
        format!("Clear shadow twin branch in {tool_name}")
    } else {
        format!(
            "Fork shadow twin branch in {tool_name} at step {}",
            timeline.current_step
        )
    };
    let info = egui::WidgetInfo::selected(
        egui::WidgetType::Button,
        ui_enabled,
        timeline.fork_active,
        &a11y_label,
    );
    let mut btn = egui::Button::new(text).min_size(min_size);
    if timeline.fork_active {
        btn = btn.stroke(egui::Stroke::new(1.5_f32, palette.accent));
    }
    let resp = ui.add_enabled(ui_enabled, btn);
    resp.widget_info({
        let info = info.clone();
        move || info.clone()
    });
    ui.ctx().data_mut(|d| {
        d.insert_temp(base_id.with("fork_branch_button_info"), info);
    });
    resp.accessible_hover_text(&a11y_label).clicked()
}

fn render_timeline_scrubber(
    ui: &mut egui::Ui,
    base_id: egui::Id,
    tool_name: &str,
    timeline: TimelineTransportState,
    ui_enabled: bool,
) -> Option<usize> {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    let mut step_val = timeline
        .current_step
        .clamp(timeline.min_step, timeline.max_step.max(timeline.min_step));
    let upper = timeline.max_step.max(timeline.min_step);
    let enabled = ui_enabled && timeline.can_scrub();
    let unit = timeline.time_unit;
    let mut scrubbed_to = None;

    ui.horizontal_wrapped(|ui| {
        ui.label(
            egui::RichText::new("Timeline:")
                .small()
                .color(palette.muted_text),
        );
        let a11y = format!(
            "Scrub {tool_name} timeline step ({} to {})",
            timeline.min_step, upper
        );
        let slider = egui::Slider::new(&mut step_val, timeline.min_step..=upper)
            .text(format!("t = {:.2}{unit}", timeline.current_time));
        let resp = ui.add_enabled(enabled, slider);
        resp.widget_info({
            let label = a11y.clone();
            let val = step_val as f64;
            move || egui::WidgetInfo::slider(enabled, val, &label)
        });
        ui.ctx().data_mut(|d| {
            d.insert_temp(base_id.with("timeline_scrubber_step"), step_val);
        });
        if resp.accessible_hover_text(&a11y).changed() && step_val != timeline.current_step {
            scrubbed_to = Some(step_val);
        }
    });

    scrubbed_to
}

fn dispatch_timeline_announcement(
    ctx: &egui::Context,
    tool_name: &str,
    timeline: TimelineTransportState,
    action: TimelineBranchAction,
) {
    match action {
        TimelineBranchAction::StepBack => {
            announce_live_status(ctx, format!("Stepped {tool_name} backward one checkpoint"));
        }
        TimelineBranchAction::ScrubTo(step) => {
            announce_live_status(ctx, format!("Scrubbed {tool_name} timeline to step {step}"));
        }
        TimelineBranchAction::ToggleForkBranch => {
            if timeline.fork_active {
                announce_branch_cleared(ctx, tool_name);
            } else {
                announce_branch_forked(
                    ctx,
                    tool_name,
                    timeline.current_step,
                    timeline.current_time,
                    timeline.time_unit,
                );
            }
        }
        TimelineBranchAction::ClearBranch => {
            announce_branch_cleared(ctx, tool_name);
        }
    }
}
