//! User interface controls and mispricing/Kelly KPI cards for `CopulaTailRiskLab`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::challenge::CHALLENGE_COPULA_TAIL_KELLY;
use super::plots::{render_sgp_correlation_curves, render_tail_regime_heatmap};
use super::CopulaTailRiskLab;
use crate::app::SemanticPalette;
use eframe::egui;

/// Renders the complete `CopulaTailRiskLab` interactive workbench.
pub fn show_copula_tail_risk_lab(lab: &mut CopulaTailRiskLab, ui: &mut egui::Ui) {
    let palette = SemanticPalette::from_visuals(ui.visuals());

    ui.heading("Copula Tail-Risk, Same-Game Parlay & Kelly Mispricing Lab");
    ui.label(
        "Explore how correlation swings SGP joint probabilities and why Student-t tail clustering amplifies extreme crashes over Gaussian copulas.",
    );
    ui.add_space(4.0);

    render_kpi_cards(lab, ui, &palette);
    ui.separator();

    render_controls(lab, ui, &palette);
    ui.separator();

    ui.columns(2, |cols| {
        cols[0].strong("2D (ρ, u) Student-t / Gaussian Tail-Amplification Heatmap");
        cols[0].small("Click anywhere on (ρ, u) to inspect conditional crash ratio P_t / P_Gauss");
        render_tail_regime_heatmap(lab, &mut cols[0], &palette);

        cols[1].strong("Same-Game Parlay Joint Probability vs Correlation ρ");
        cols[1].small("Contrasts naive p₁·p₂ multiplication against Gaussian & Student-t copulas");
        render_sgp_correlation_curves(lab, &mut cols[1], &palette);
    });
}

fn render_kpi_cards(lab: &CopulaTailRiskLab, ui: &mut egui::Ui, palette: &SemanticPalette) {
    let m = &lab.metrics;
    ui.horizontal_wrapped(|ui| {
        ui.group(|ui| {
            ui.label("SGP Joint Probability");
            ui.colored_label(
                palette.info,
                format!(
                    "Gaussian {:.2}% | Student-t {:.2}%",
                    m.sgp_gaussian_prob * 100.0,
                    m.student_t_joint_prob * 100.0
                ),
            );
            ui.small(format!(
                "Naive {:.2}% | ρ=-0.50: {:.2}% | ρ=+0.70: {:.2}%",
                m.naive_independent_prob * 100.0,
                m.sgp_neg50_prob * 100.0,
                m.sgp_pos70_prob * 100.0
            ));
        });
        render_tail_and_kelly_cards(lab, ui, palette);
    });

    if m.is_overbet_ruin {
        ui.colored_label(
            palette.error,
            "⚠ Kelly Overbetting / Correlation Trap: Wager exceeds safe correlated edge, driving expected logarithmic bankroll growth negative!",
        );
    }
}

fn render_tail_and_kelly_cards(
    lab: &CopulaTailRiskLab,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    let m = &lab.metrics;
    let tail_col = if m.tail_multiplier >= 3.0 {
        palette.warning
    } else {
        palette.success
    };
    ui.group(|ui| {
        ui.label(format!(
            "Upper-Tail Crash @ u = {:.3}",
            lab.config.tail_quantile_u
        ));
        ui.colored_label(
            tail_col,
            format!("{:.2}x Student-t Multiplier", m.tail_multiplier),
        );
        ui.small(format!(
            "Student-t {:.2}% vs Gaussian {:.2}% | λ_U = {:.3}",
            m.student_t_cond_tail * 100.0,
            m.gaussian_cond_tail * 100.0,
            m.asymptotic_lambda_u
        ));
    });

    let g_col = if m.true_expected_growth > 0.0 {
        palette.success
    } else if m.true_expected_growth < 0.0 {
        palette.error
    } else {
        palette.muted_text
    };
    ui.group(|ui| {
        ui.label("Correlated Kelly Bankroll Sizing");
        ui.colored_label(
            g_col,
            format!(
                "Stake {:.2}% | E[Δln W] = {:+.4}",
                m.applied_stake * 100.0,
                m.true_expected_growth
            ),
        );
        ui.small(format!(
            "Optimal f* = {:.2}% | Naive f = {:.2}%",
            m.copula_kelly_fraction * 100.0,
            m.naive_kelly_fraction * 100.0
        ));
    });
}

fn render_controls(lab: &mut CopulaTailRiskLab, ui: &mut egui::Ui, palette: &SemanticPalette) {
    let locked = lab.active_challenge == Some(CHALLENGE_COPULA_TAIL_KELLY);
    let mut changed = false;

    ui.columns(2, |cols| {
        changed |= cols[0]
            .add(
                egui::Slider::new(&mut lab.config.correlation_rho, -0.95..=0.95)
                    .text("Correlation ρ"),
            )
            .changed();
        changed |= cols[0]
            .add(
                egui::Slider::new(&mut lab.config.stake_multiplier, 0.10..=2.50)
                    .text("Kelly Stake Multiplier"),
            )
            .changed();
        cols[0].add_enabled_ui(!locked, |ui| {
            changed |= ui
                .add(
                    egui::Slider::new(&mut lab.config.tail_quantile_u, 0.85..=0.9995)
                        .text("Tail Quantile u"),
                )
                .changed();
        });
        changed |= render_locked_copula_sliders(lab, &mut cols[1], locked);
    });

    ui.horizontal(|ui| {
        if ui.button("Starter SGP + 5.83x Tail Preset").clicked() {
            lab.reset_starter_preset();
        }
        if locked {
            ui.colored_label(
                palette.info,
                "🔒 Challenge locks active on marginals, odds, ν = 3.0 & quantile u = 0.999",
            );
        }
    });

    if changed {
        lab.recalculate();
    }
}

fn render_locked_copula_sliders(
    lab: &mut CopulaTailRiskLab,
    ui: &mut egui::Ui,
    locked: bool,
) -> bool {
    let mut changed = false;
    ui.add_enabled_ui(!locked, |ui| {
        changed |= ui
            .add(
                egui::Slider::new(&mut lab.config.leg1_prob, 0.02..=0.95)
                    .text("Leg 1 Marginal Prob"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut lab.config.leg2_prob, 0.02..=0.95)
                    .text("Leg 2 Marginal Prob"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut lab.config.degrees_of_freedom, 1.5..=30.0)
                    .text("Student-t ν (DoF)"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut lab.config.offered_decimal_odds, 1.2..=100.0)
                    .text("Offered Decimal Odds"),
            )
            .changed();
    });
    changed
}

// [cite:clinical_trials_statistics]
