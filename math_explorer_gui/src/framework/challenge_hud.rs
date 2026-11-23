//! WCAG 2.2 AA Inverse Challenge HUD banner rendered inside [`super::SimulationFramework`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::challenge::ChallengeEvaluation;
use super::{scoped_panel_id, AccessibleHoverText, SimulationFramework};
use crate::app::SemanticPalette;
use crate::starter_gallery::ChallengeMetadata;
use eframe::egui;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HudAction {
    ResetBaseline,
    CopyShareLink,
    ExitChallenge,
}

impl SimulationFramework {
    pub(super) fn show_challenge_hud(&mut self, ui: &mut egui::Ui, id_source: &str) {
        let Some(meta) = self.active_challenge else {
            return;
        };
        let eval = self
            .active_tool
            .as_ref()
            .and_then(|tool| tool.evaluate_challenge())
            .unwrap_or_else(|| {
                ChallengeEvaluation::new(0.0, false, "Residual", 0.0, "Evaluating challenge state")
            });

        let ctx = ui.ctx().clone();
        if eval.passed && !self.challenge_was_passed {
            self.challenge_was_passed = true;
            crate::accessibility::announce_live_status(
                &ctx,
                format!(
                    "Challenge passed: {}! Score {:.1} out of 100.",
                    meta.title, eval.score
                ),
            );
        } else if !eval.passed && self.challenge_was_passed {
            self.challenge_was_passed = false;
        }

        let hud_id = scoped_panel_id(ui, format!("{id_source}_challenge_hud"));
        let mut triggered_action = None;
        egui::TopBottomPanel::top(hud_id).show_inside(ui, |ui| {
            triggered_action = render_hud_panel_contents(ui, meta, &eval);
        });

        match triggered_action {
            Some(HudAction::ResetBaseline) => {
                let _ = self.reset_active_challenge_baseline(Some(&ctx));
                ctx.request_repaint();
            }
            Some(HudAction::CopyShareLink) => {
                self.copy_active_share_link(&ctx);
                ctx.request_repaint();
            }
            Some(HudAction::ExitChallenge) => {
                self.exit_active_challenge(Some(&ctx));
                ctx.request_repaint();
            }
            None => {}
        }
    }
}

fn render_hud_panel_contents(
    ui: &mut egui::Ui,
    meta: &ChallengeMetadata,
    eval: &ChallengeEvaluation,
) -> Option<HudAction> {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    let border_color = if eval.passed {
        palette.success
    } else {
        palette.graph_edge
    };
    let mut action = None;
    egui::Frame::NONE
        .fill(ui.visuals().panel_fill)
        .stroke(egui::Stroke::new(1.5_f32, border_color))
        .corner_radius(6.0)
        .inner_margin(egui::Margin::symmetric(10, 6))
        .show(ui, |ui| {
            action = render_hud_top_row(ui, meta, eval, &palette);
            ui.add_space(4.0);
            render_hud_constraints_row(ui, meta, eval, &palette);
        });
    action
}

fn render_hud_top_row(
    ui: &mut egui::Ui,
    meta: &ChallengeMetadata,
    eval: &ChallengeEvaluation,
    palette: &SemanticPalette,
) -> Option<HudAction> {
    let mut action = None;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 8.0;
        ui.label(
            egui::RichText::new(format!("{} {}", meta.icon, meta.title))
                .strong()
                .color(ui.visuals().text_color()),
        );
        render_pass_badge(ui, eval.passed, palette);
        render_score_meter(ui, eval.score, eval.passed, palette);
        ui.label(
            egui::RichText::new(format!(
                "{}: {:.4}",
                eval.residual_label, eval.residual_value
            ))
            .monospace()
            .color(palette.info),
        );

        if ui
            .button("↻ Baseline Reset")
            .accessible_hover_text("Reset tunable parameters to unsolved challenge baseline")
            .clicked()
        {
            action = Some(HudAction::ResetBaseline);
        }
        if ui
            .button("🔗 Share Challenge")
            .accessible_hover_text("Copy stateless #tab=...&challenge=<id> URL to clipboard")
            .clicked()
        {
            action = Some(HudAction::CopyShareLink);
        }
        if ui
            .button("✕ Exit")
            .accessible_hover_text("Exit inverse challenge mode and unlock all parameters")
            .clicked()
        {
            action = Some(HudAction::ExitChallenge);
        }
    });
    action
}

fn render_pass_badge(ui: &mut egui::Ui, passed: bool, palette: &SemanticPalette) {
    let (label, color) = if passed {
        ("✓ PASS", palette.success)
    } else {
        ("◎ IN PROGRESS", palette.warning)
    };
    egui::Frame::NONE
        .fill(ui.visuals().extreme_bg_color)
        .stroke(egui::Stroke::new(1.5_f32, color))
        .corner_radius(4.0)
        .inner_margin(egui::Margin::symmetric(6, 2))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(label).strong().small().color(color));
        });
}

fn render_score_meter(ui: &mut egui::Ui, score: f64, passed: bool, palette: &SemanticPalette) {
    let fill_color = if passed {
        palette.success
    } else {
        palette.accent
    };
    let (rect, response) = ui.allocate_exact_size(egui::vec2(108.0, 18.0), egui::Sense::hover());
    crate::accessibility::register_canvas_accessibility(
        ui,
        &response,
        egui::accesskit::Role::Meter,
        "Challenge Score",
        Some(&format!("{score:.1} out of 100")),
    );
    let painter = ui.painter();
    painter.rect(
        rect,
        4.0,
        ui.visuals().extreme_bg_color,
        egui::Stroke::new(1.0_f32, palette.graph_edge),
        egui::StrokeKind::Inside,
    );
    let frac = (score as f32 / 100.0).clamp(0.0, 1.0);
    if frac > 0.0 {
        let inner = rect.shrink(2.0);
        let bar_rect =
            egui::Rect::from_min_size(inner.min, egui::vec2(inner.width() * frac, inner.height()));
        painter.rect_filled(bar_rect, 2.0, fill_color);
    }
    ui.label(
        egui::RichText::new(format!("{score:.1}/100"))
            .strong()
            .monospace()
            .color(ui.visuals().text_color()),
    );
}

fn render_hud_constraints_row(
    ui: &mut egui::Ui,
    meta: &ChallengeMetadata,
    eval: &ChallengeEvaluation,
    palette: &SemanticPalette,
) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 10.0;
        ui.label(
            egui::RichText::new(format!("🔒 Locked: {}", meta.locked_inputs))
                .small()
                .color(palette.muted_text),
        );
        ui.label(
            egui::RichText::new(format!("🎛 Budget: {}", meta.tunable_budget))
                .small()
                .color(palette.muted_text),
        );
        ui.label(
            egui::RichText::new(format!("🎯 Target: {}", meta.target_summary))
                .small()
                .color(ui.visuals().text_color()),
        );
        if !eval.detail.is_empty() {
            ui.label(
                egui::RichText::new(format!("• {}", eval.detail))
                    .small()
                    .color(palette.info),
            );
        }
    });
}

// [cite:favorite_child]
