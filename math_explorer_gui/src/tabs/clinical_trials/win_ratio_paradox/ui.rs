//! User interface controls and KPI paradox summary banner for `WinRatioParadoxLab`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::challenge::CHALLENGE_WIN_RATIO_REVERSAL;
use super::plots::{render_kaplan_meier_comparison_plot, render_regime_map_plot};
use super::WinRatioParadoxLab;
use crate::app::SemanticPalette;
use eframe::egui;

/// Renders the complete `WinRatioParadoxLab` interactive view.
pub fn show_win_ratio_paradox_lab(lab: &mut WinRatioParadoxLab, ui: &mut egui::Ui) {
    let palette = SemanticPalette::from_visuals(ui.visuals());

    ui.heading("Win Ratio Composite-Endpoint Paradox Lab");
    ui.label(
        "Contrast pooled time-to-first-event Hazard Ratios against Pocock's prioritized death-before-hospitalization Win Ratio.",
    );
    ui.add_space(4.0);

    render_kpi_cards(lab, ui, &palette);
    ui.separator();

    render_parameter_controls(lab, ui, &palette);
    ui.separator();

    ui.columns(2, |cols| {
        cols[0].strong("2D Severity-Reversal Regime Map (Click to Teleport)");
        cols[0].small(
            "Highlight: Pooled HR > 1.0 (naive penalty) & Prioritized WR > 1.0 (true benefit)",
        );
        render_regime_map_plot(lab, &mut cols[0], &palette);

        cols[1].strong("Two-Arm Kaplan-Meier: Fatal Survival vs Pooled First Event");
        cols[1]
            .small("Early non-fatal hospitalizations collapse Treatment's dashed composite curve");
        render_kaplan_meier_comparison_plot(lab, &mut cols[1], &palette);
    });
}

fn render_kpi_cards(lab: &WinRatioParadoxLab, ui: &mut egui::Ui, palette: &SemanticPalette) {
    let m = &lab.metrics;
    ui.horizontal_wrapped(|ui| {
        let hr_color = if m.pooled_hazard_ratio > 1.0 {
            palette.error
        } else {
            palette.success
        };
        ui.group(|ui| {
            ui.label("Pooled First-Event HR");
            ui.colored_label(hr_color, format!("HR = {:.4}", m.pooled_hazard_ratio));
            ui.small(if m.pooled_hazard_ratio > 1.0 {
                "Naive composite penalizes treatment"
            } else {
                "Naive composite favors treatment"
            });
        });

        let wr_color = if m.win_ratio > 1.0 {
            palette.success
        } else {
            palette.warning
        };
        ui.group(|ui| {
            ui.label("Prioritized Win Ratio");
            ui.colored_label(
                wr_color,
                format!("WR = {:.4} (p = {:.2e})", m.win_ratio, m.win_ratio_p_value),
            );
            ui.small(format!(
                "95% CI [{:.2}, {:.2}] | Matched WR = {:.2}",
                m.win_ratio_ci.0, m.win_ratio_ci.1, m.matched_win_ratio
            ));
        });

        ui.group(|ui| {
            ui.label("Fatal Survival t-Test");
            ui.colored_label(
                palette.info,
                format!("t = {:.4} (p = {:.2e})", m.t_statistic, m.t_p_value),
            );
            ui.small(format!(
                "Wins={} (T1:{}, T2:{}) | Losses={} | Ties={}",
                m.wins, m.tier1_wins, m.tier2_wins, m.losses, m.ties
            ));
        });
    });

    if m.is_severity_reversal {
        ui.colored_label(
            palette.warning,
            format!(
                "⚡ Composite Severity Reversal Active: Pooled time-to-first-event reports HR = {:.4} against treatment, yet hierarchical death-first Win Ratio = {:.4} (t = {:.4}) confirms substantial survival benefit.",
                m.pooled_hazard_ratio, m.win_ratio, m.t_statistic
            ),
        );
    }
}

fn render_parameter_controls(
    lab: &mut WinRatioParadoxLab,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    let locked = lab.active_challenge == Some(CHALLENGE_WIN_RATIO_REVERSAL);
    let mut changed = false;

    ui.columns(2, |cols| {
        changed |= cols[0]
            .add(
                egui::Slider::new(&mut lab.config.mortality_reduction, 0.0..=0.90)
                    .text("Mortality Reduction"),
            )
            .changed();
        changed |= cols[0]
            .add(
                egui::Slider::new(&mut lab.config.early_hosp_rate, 0.0..=0.95)
                    .text("Early Non-Fatal Hosp Rate"),
            )
            .changed();
        changed |= cols[0]
            .add(
                egui::Slider::new(&mut lab.config.early_hosp_mean_months, 1.0..=24.0)
                    .text("Mean Early Hosp Onset (mo)"),
            )
            .changed();

        cols[1].add_enabled_ui(!locked, |ui| {
            changed |= ui
                .add(
                    egui::Slider::new(&mut lab.config.sample_size_per_arm, 10..=40)
                        .text("Subjects / Arm"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut lab.config.control_mortality_rate, 0.20..=0.85)
                        .text("Control Mortality Rate"),
                )
                .changed();
            changed |= ui
                .checkbox(
                    &mut lab.config.matched_pairing,
                    "1:1 Matched Pairing (vs Unmatched All-Pairs)",
                )
                .changed();
        });
    });

    render_action_buttons(lab, ui, palette, locked);

    if changed {
        lab.recalculate();
    }
}

fn render_action_buttons(
    lab: &mut WinRatioParadoxLab,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
    locked: bool,
) {
    ui.horizontal(|ui| {
        if ui.button("Starter Reversal Preset").clicked() {
            lab.reset_starter_cohort();
        }
        ui.add_enabled_ui(!locked, |ui| {
            if ui.button("Resample Cohort (Seeded RNG)").clicked() {
                lab.resample_seeded_cohort();
            }
        });
        if locked {
            ui.colored_label(
                palette.info,
                "🔒 Challenge locks active on N, horizon, control mortality & pairing",
            );
        }
    });
}

// [cite:win_ratio]
