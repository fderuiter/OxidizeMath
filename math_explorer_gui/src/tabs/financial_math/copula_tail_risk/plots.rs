//! 2D `(rho, u)` tail-multiplier heatmap and SGP correlation curves for `CopulaTailRiskLab`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{CopulaTailRiskLab, COPULA_GRID_DIM};
use crate::app::{SemanticPalette, ThemeMode};
use eframe::egui;
use egui_plot::{HLine, Line, LineStyle, Plot, PlotImage, PlotPoint, Points, VLine};

/// Renders the 2D `(rho, u)` Student-t / Gaussian tail-amplification heatmap with click-to-teleport.
pub fn render_tail_regime_heatmap(
    lab: &mut CopulaTailRiskLab,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    let tex_id = ensure_copula_texture(lab, ui);
    let cur_rho = lab.config.correlation_rho;
    let cur_u = lab.config.tail_quantile_u;
    let mut teleport: Option<[f64; 2]> = None;

    Plot::new("copula_tail_regime_heatmap_plot")
        .height(270.0)
        .allow_scroll(false)
        .x_axis_label("Latent Correlation ρ")
        .y_axis_label("Extreme Quantile u")
        .include_x(-0.85)
        .include_x(0.90)
        .include_y(0.85)
        .include_y(0.9995)
        .show(ui, |plot_ui| {
            if let Some(tid) = tex_id {
                plot_ui.image(PlotImage::new(
                    "Student-t / Gaussian Tail Ratio",
                    tid,
                    PlotPoint::new(0.025, 0.9245),
                    egui::vec2(1.75, 0.149),
                ));
            }
            plot_ui.vline(
                VLine::new("Selected ρ", cur_rho)
                    .color(palette.accent)
                    .style(LineStyle::Dashed { length: 6.0_f32 }),
            );
            plot_ui.hline(
                HLine::new("Selected Quantile u", cur_u)
                    .color(palette.accent)
                    .style(LineStyle::Dashed { length: 6.0_f32 }),
            );
            plot_ui.points(
                Points::new("Active (ρ, u)", vec![[cur_rho, cur_u]])
                    .radius(6.0_f32)
                    .color(palette.accent),
            );
            lab.callout_manager.render_on_plot(plot_ui);
            if plot_ui.response().clicked() {
                if let Some(ptr) = plot_ui.pointer_coordinate() {
                    teleport = Some([ptr.x.clamp(-0.95, 0.95), ptr.y.clamp(0.85, 0.9995)]);
                }
            }
        });

    if let Some([n_rho, n_u]) = teleport {
        lab.config.correlation_rho = n_rho;
        if lab.active_challenge.is_none() {
            lab.config.tail_quantile_u = n_u;
        }
        lab.recalculate();
    }
}

/// Renders SGP joint probability curves across correlation $\rho \in [-0.90, 0.90]$.
pub fn render_sgp_correlation_curves(
    lab: &CopulaTailRiskLab,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    let fair_break_even = 1.0 / lab.config.offered_decimal_odds.max(1.01);
    Plot::new("copula_sgp_correlation_curves_plot")
        .height(270.0)
        .allow_scroll(false)
        .x_axis_label("Latent Correlation ρ")
        .y_axis_label("Joint Parlay Probability P(A ∩ B)")
        .include_x(-0.90)
        .include_x(0.90)
        .include_y(0.0)
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("Gaussian Copula SGP", lab.curve_gaussian.clone())
                    .color(palette.info)
                    .width(2.6_f32),
            );
            plot_ui.line(
                Line::new("Student-t Copula SGP", lab.curve_student_t.clone())
                    .color(palette.success)
                    .width(2.4_f32),
            );
            plot_ui.line(
                Line::new("Naive Independence (p₁·p₂)", lab.curve_naive.clone())
                    .color(palette.muted_text)
                    .style(LineStyle::Dashed { length: 6.0_f32 })
                    .width(2.0_f32),
            );
            plot_ui.hline(
                HLine::new("Sportsbook Break-Even Prob (1/Odds)", fair_break_even)
                    .color(palette.warning)
                    .style(LineStyle::Dotted { spacing: 5.0_f32 }),
            );
            plot_ui.points(
                Points::new(
                    "Active SGP Point",
                    vec![[lab.config.correlation_rho, lab.metrics.sgp_gaussian_prob]],
                )
                .radius(5.5_f32)
                .color(palette.accent),
            );
        });
}

// [cite:clinical_trials_statistics]

fn ensure_copula_texture(lab: &mut CopulaTailRiskLab, ui: &egui::Ui) -> Option<egui::TextureId> {
    let high_contrast = ThemeMode::from_visuals(ui.visuals()) == ThemeMode::HighContrast;
    if lab.texture_dirty || lab.regime_texture.is_none() {
        lab.populate_regime_heatmap(high_contrast);
        let img = egui::ColorImage::new(
            [COPULA_GRID_DIM, COPULA_GRID_DIM],
            lab.regime_pixels.clone(),
        );
        let opts = egui::TextureOptions::NEAREST;
        if let Some(tex) = &mut lab.regime_texture {
            tex.set(img, opts);
        } else {
            lab.regime_texture = Some(ui.ctx().load_texture("copula_tail_regime_tex", img, opts));
        }
        lab.texture_dirty = false;
    }
    lab.regime_texture.as_ref().map(egui::TextureHandle::id)
}
