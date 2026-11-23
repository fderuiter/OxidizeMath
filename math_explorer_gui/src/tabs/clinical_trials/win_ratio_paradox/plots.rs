//! 2D regime map and Kaplan-Meier plot views for `WinRatioParadoxLab`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{WinRatioParadoxLab, REGIME_GRID_DIM};
use crate::app::{SemanticPalette, ThemeMode};
use eframe::egui;
use egui_plot::{HLine, Line, LineStyle, Plot, PlotImage, PlotPoint, Points, VLine};

/// Renders the 2D `(mortality_reduction, early_hosp_rate)` reversal regime map with click-to-teleport.
pub fn render_regime_map_plot(
    lab: &mut WinRatioParadoxLab,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    let tex_id = ensure_regime_texture(lab, ui);
    let cur_x = lab.config.mortality_reduction;
    let cur_y = lab.config.early_hosp_rate;
    let mut clicked_pos: Option<[f64; 2]> = None;

    Plot::new("win_ratio_regime_map_plot")
        .height(270.0)
        .allow_scroll(false)
        .x_axis_label("Mortality Reduction (1 - RR_death)")
        .y_axis_label("Early Non-Fatal Hospitalization Rate")
        .include_x(0.0)
        .include_x(0.90)
        .include_y(0.0)
        .include_y(0.95)
        .show(ui, |plot_ui| {
            if let Some(tid) = tex_id {
                let image = PlotImage::new(
                    "Severity Reversal Regime",
                    tid,
                    PlotPoint::new(0.45, 0.475),
                    egui::vec2(0.90, 0.95),
                );
                plot_ui.image(image);
            }
            plot_ui.vline(
                VLine::new("Current Mortality Reduction", cur_x)
                    .color(palette.accent)
                    .style(LineStyle::Dashed { length: 6.0_f32 }),
            );
            plot_ui.hline(
                HLine::new("Current Early Hosp Rate", cur_y)
                    .color(palette.accent)
                    .style(LineStyle::Dashed { length: 6.0_f32 }),
            );
            plot_ui.points(
                Points::new("Active Trial Operating Point", vec![[cur_x, cur_y]])
                    .radius(6.0_f32)
                    .color(palette.accent),
            );
            lab.callout_manager.render_on_plot(plot_ui);
            if plot_ui.response().clicked() {
                if let Some(ptr) = plot_ui.pointer_coordinate() {
                    clicked_pos = Some([ptr.x.clamp(0.0, 0.90), ptr.y.clamp(0.0, 0.95)]);
                }
            }
        });

    if let Some([nx, ny]) = clicked_pos {
        lab.config.mortality_reduction = nx;
        lab.config.early_hosp_rate = ny;
        lab.recalculate();
    }
}

/// Renders two-arm composite first-event vs fatal survival Kaplan-Meier step curves.
pub fn render_kaplan_meier_comparison_plot(
    lab: &WinRatioParadoxLab,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    Plot::new("win_ratio_km_comparison_plot")
        .height(270.0)
        .allow_scroll(false)
        .x_axis_label("Follow-Up Time (months)")
        .y_axis_label("Event-Free Probability S(t)")
        .include_x(0.0)
        .include_x(lab.config.horizon_months)
        .include_y(0.0)
        .include_y(1.05)
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("Treatment Fatal Survival KM", lab.km_fatal_treat.clone())
                    .color(palette.success)
                    .width(2.8_f32),
            );
            plot_ui.line(
                Line::new("Control Fatal Survival KM", lab.km_fatal_ctrl.clone())
                    .color(palette.error)
                    .width(2.4_f32),
            );
            plot_ui.line(
                Line::new("Treatment Pooled First-Event KM", lab.km_comp_treat.clone())
                    .color(palette.warning)
                    .style(LineStyle::Dashed { length: 7.0_f32 })
                    .width(2.4_f32),
            );
            plot_ui.line(
                Line::new("Control Pooled First-Event KM", lab.km_comp_ctrl.clone())
                    .color(palette.info)
                    .style(LineStyle::Dotted { spacing: 5.0_f32 })
                    .width(2.0_f32),
            );
        });
}

// [cite:win_ratio]

fn ensure_regime_texture(lab: &mut WinRatioParadoxLab, ui: &egui::Ui) -> Option<egui::TextureId> {
    let high_contrast = ThemeMode::from_visuals(ui.visuals()) == ThemeMode::HighContrast;
    if lab.texture_dirty || lab.regime_texture.is_none() {
        lab.populate_regime_map(high_contrast);
        let img = egui::ColorImage::new(
            [REGIME_GRID_DIM, REGIME_GRID_DIM],
            lab.regime_pixels.clone(),
        );
        let opts = egui::TextureOptions::NEAREST;
        if let Some(tex) = &mut lab.regime_texture {
            tex.set(img, opts);
        } else {
            lab.regime_texture = Some(ui.ctx().load_texture("win_ratio_regime_tex", img, opts));
        }
        lab.texture_dirty = false;
    }
    lab.regime_texture.as_ref().map(egui::TextureHandle::id)
}
