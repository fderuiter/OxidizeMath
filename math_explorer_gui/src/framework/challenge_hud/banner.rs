//! Top-banner UI rendering for guided stories, playable AST challenges, and custom AST sandbox.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::ChallengeHudMode;
use crate::accessibility::AccessibleHoverText;
use crate::framework::{scoped_panel_id, SimulationFramework};
use crate::starter_gallery::{ChallengeMetadata, PresetStoryMetadata};
use eframe::egui;

enum HudBannerAction {
    None,
    StepStory(usize),
    ToggleHint,
    ResetChallenge,
    ToggleSandbox,
    CloseHud,
}

impl SimulationFramework {
    pub(in crate::framework) fn render_challenge_hud_panel(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        id_source: &str,
    ) {
        if matches!(self.challenge_hud.mode, ChallengeHudMode::Inactive)
            && !self.challenge_hud.sandbox.is_open
        {
            return;
        }
        self.flush_hud_announcements(ctx);
        let observables = self
            .active_tool
            .as_ref()
            .map(|t| t.observables())
            .unwrap_or_default();
        let panel_id = scoped_panel_id(ui, format!("{id_source}_challenge_hud_banner"));
        let mut action = HudBannerAction::None;
        egui::TopBottomPanel::top(panel_id).show_inside(ui, |ui| {
            action = match &self.challenge_hud.mode {
                ChallengeHudMode::Story {
                    story, step_index, ..
                } => draw_story_banner(ui, story, *step_index, &observables),
                ChallengeHudMode::Challenge {
                    challenge,
                    solved,
                    currently_satisfied,
                    show_hint,
                    ..
                } => draw_challenge_banner(
                    ui,
                    challenge,
                    (*solved, *currently_satisfied, *show_hint, self.is_diverged),
                    &observables,
                ),
                ChallengeHudMode::Inactive => HudBannerAction::None,
            };
            if self.challenge_hud.sandbox.is_open {
                ui.separator();
                self.draw_sandbox_row(ui, &observables);
            }
        });
        self.apply_hud_banner_action(ctx, action);
    }

    fn draw_sandbox_row(&mut self, ui: &mut egui::Ui, observables: &[(&'static str, f64)]) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let mut edited = false;
        ui.horizontal_wrapped(|ui| {
            ui.strong("🧪 AST Sandbox:");
            edited = ui
                .add(
                    egui::TextEdit::singleline(&mut self.challenge_hud.sandbox.formula)
                        .hint_text("e.g. expected_growth_rate < 0 && stake_multiplier > 2")
                        .desired_width(280.0),
                )
                .changed();
            match &self.challenge_hud.sandbox.last_eval {
                Ok(true) => ui.colored_label(palette.success, "✅ Predicate TRUE"),
                Ok(false) => ui.colored_label(ui.visuals().weak_text_color(), "⏳ Predicate FALSE"),
                Err(err) => ui.colored_label(palette.error, format!("⚠️ {err}")),
            };
        });
        if edited {
            let next = self.challenge_hud.sandbox.formula.clone();
            self.challenge_hud.sandbox.set_formula(&next);
            self.eval_sandbox_with_observables(observables, Some(&ui.ctx().clone()));
        }
    }

    fn apply_hud_banner_action(&mut self, ctx: &egui::Context, action: HudBannerAction) {
        match action {
            HudBannerAction::None => {}
            HudBannerAction::StepStory(idx) => {
                let _ = self.step_story(idx);
                self.flush_hud_announcements(ctx);
                ctx.request_repaint();
            }
            HudBannerAction::ToggleHint => {
                if let ChallengeHudMode::Challenge {
                    ref mut show_hint, ..
                } = self.challenge_hud.mode
                {
                    *show_hint = !*show_hint;
                }
            }
            HudBannerAction::ResetChallenge => {
                let _ = self.reset_active_challenge();
                ctx.request_repaint();
            }
            HudBannerAction::ToggleSandbox => {
                self.challenge_hud.sandbox.is_open = !self.challenge_hud.sandbox.is_open;
            }
            HudBannerAction::CloseHud => {
                self.clear_challenge_hud();
                ctx.request_repaint();
            }
        }
    }
}

fn draw_story_banner(
    ui: &mut egui::Ui,
    story: &'static PresetStoryMetadata,
    step_index: usize,
    observables: &[(&'static str, f64)],
) -> HudBannerAction {
    let mut action = HudBannerAction::None;
    let total = story.steps.len();
    let Some(step) = story.steps.get(step_index) else {
        return action;
    };
    ui.horizontal_wrapped(|ui| {
        ui.strong(format!(
            "{} {} — Step {}/{total}: {}",
            story.icon,
            story.title,
            step_index + 1,
            step.title
        ));
        if ui
            .add_enabled(step_index > 0, egui::Button::new("◀ Prev"))
            .accessible_hover_text("Previous story step")
            .clicked()
        {
            action = HudBannerAction::StepStory(step_index.saturating_sub(1));
        }
        if ui
            .add_enabled(step_index + 1 < total, egui::Button::new("Next ▶"))
            .accessible_hover_text("Next story step")
            .clicked()
        {
            action = HudBannerAction::StepStory(step_index + 1);
        }
        if ui
            .button("🧪 Sandbox")
            .accessible_hover_text("Toggle custom AST sandbox")
            .clicked()
        {
            action = HudBannerAction::ToggleSandbox;
        }
        if ui
            .button("✖ Exit Story")
            .accessible_hover_text("Close story banner")
            .clicked()
        {
            action = HudBannerAction::CloseHud;
        }
    });
    ui.label(step.narrative);
    draw_observable_strip(ui, observables, step.focus_observables);
    action
}

fn draw_challenge_banner(
    ui: &mut egui::Ui,
    challenge: &'static ChallengeMetadata,
    flags: (bool, bool, bool, bool),
    observables: &[(&'static str, f64)],
) -> HudBannerAction {
    let (solved, currently_satisfied, show_hint, is_diverged) = flags;
    let mut action = HudBannerAction::None;
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    ui.horizontal_wrapped(|ui| {
        ui.strong(format!("{} Challenge: {}", challenge.icon, challenge.title));
        let (color, badge) = if is_diverged {
            (palette.warning, "⏸ Paused (solver diverged)")
        } else if solved && currently_satisfied {
            (palette.success, "✅ Solved!")
        } else if solved {
            (palette.success, "✅ Solved (goal explored)")
        } else {
            (palette.accent, "🎯 In Progress")
        };
        ui.colored_label(color, badge);
        if ui.button("💡 Hint").clicked() {
            action = HudBannerAction::ToggleHint;
        }
        if ui.button("↻ Reset Challenge").clicked() {
            action = HudBannerAction::ResetChallenge;
        }
        if ui.button("🧪 Sandbox").clicked() {
            action = HudBannerAction::ToggleSandbox;
        }
        if ui.button("✖ Exit Challenge").clicked() {
            action = HudBannerAction::CloseHud;
        }
    });
    ui.label(challenge.goal_summary);
    ui.small(format!("Target AST: {}", challenge.predicate_expr));
    if show_hint {
        ui.colored_label(palette.accent, format!("Hint: {}", challenge.hint));
    }
    draw_observable_strip(ui, observables, challenge.focus_observables);
    action
}

fn draw_observable_strip(
    ui: &mut egui::Ui,
    observables: &[(&'static str, f64)],
    focus: &[&'static str],
) {
    let mut chips: Vec<(&'static str, f64)> = focus
        .iter()
        .filter_map(|&wanted| observables.iter().copied().find(|(k, _)| *k == wanted))
        .collect();
    if chips.is_empty() {
        chips.extend(observables.iter().copied().take(4));
    }
    if chips.is_empty() {
        return;
    }
    ui.horizontal_wrapped(|ui| {
        ui.small("Live Observables:");
        for (name, val) in chips {
            let text = if val.abs() >= 100.0 || val == 0.0 {
                format!("{name} = {val:.2}")
            } else {
                format!("{name} = {val:.4}")
            };
            ui.code(text);
        }
    });
}
