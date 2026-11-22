#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::OdeSolverTool;
use crate::presets::OdePreset;
use eframe::egui::{Color32, CornerRadius, Stroke, Vec2};
use egui_plot::{Callout, PlotUi};

fn callout_frame_styling(
    plot_ui: &PlotUi<'_>,
) -> (Color32, Stroke, Color32, crate::app::SemanticPalette) {
    let visuals = plot_ui.ctx().style().visuals.clone();
    let palette = crate::app::SemanticPalette::from_visuals(&visuals);
    let bg_fill = Color32::from_rgba_unmultiplied(
        visuals.window_fill().r(),
        visuals.window_fill().g(),
        visuals.window_fill().b(),
        230,
    );
    let frame_stroke = visuals.widgets.noninteractive.bg_stroke;
    let text_color = visuals.text_color();
    (bg_fill, frame_stroke, text_color, palette)
}

fn render_initial_condition_callout(
    plot_ui: &mut PlotUi<'_>,
    y0: f64,
    bg_fill: Color32,
    frame_stroke: Stroke,
    text_color: Color32,
    palette: &crate::app::SemanticPalette,
) {
    if !y0.is_finite() {
        return;
    }
    let callout_ic = Callout::new([0.0, y0], format!("Initial Condition: y(0) = {:.2}", y0))
        .offset(Vec2::new(35.0, -30.0))
        .leader_stroke(Stroke::new(1.5_f32, palette.series_color(0)))
        .frame(bg_fill, frame_stroke)
        .corner_radius(CornerRadius::same(4))
        .text_color(text_color);

    plot_ui.callout(callout_ic);
}

fn render_preset_asymptote_callout(
    plot_ui: &mut PlotUi<'_>,
    tool: &OdeSolverTool,
    bg_fill: Color32,
    frame_stroke: Stroke,
    text_color: Color32,
    palette: &crate::app::SemanticPalette,
) {
    match tool.preset {
        OdePreset::LogisticGrowth => {
            if tool.param_cap_k.is_finite() && tool.total_time.is_finite() {
                let callout_cap = Callout::new(
                    [tool.total_time, tool.param_cap_k],
                    format!("Carrying Capacity: K = {:.2}", tool.param_cap_k),
                )
                .offset(Vec2::new(-45.0, -25.0))
                .leader_stroke(Stroke::new(1.5_f32, palette.success))
                .frame(bg_fill, frame_stroke)
                .corner_radius(CornerRadius::same(4))
                .text_color(text_color);

                plot_ui.callout(callout_cap);
            }
        }
        OdePreset::Exponential => {
            if tool.param_k <= 0.0 && tool.total_time.is_finite() {
                let callout_ss = Callout::new([tool.total_time, 0.0], "Steady State: y = 0.00")
                    .offset(Vec2::new(-40.0, -25.0))
                    .leader_stroke(Stroke::new(1.5_f32, palette.series_color(1)))
                    .frame(bg_fill, frame_stroke)
                    .corner_radius(CornerRadius::same(4))
                    .text_color(text_color);

                plot_ui.callout(callout_ss);
            }
        }
        OdePreset::HarmonicOscillator => {
            if let Some(root_t) = harmonic_equilibrium_root(tool) {
                let callout_eq = Callout::new([root_t, 0.0], "Equilibrium: y = 0.00")
                    .offset(Vec2::new(30.0, 25.0))
                    .leader_stroke(Stroke::new(1.5_f32, palette.series_color(1)))
                    .frame(bg_fill, frame_stroke)
                    .corner_radius(CornerRadius::same(4))
                    .text_color(text_color);

                plot_ui.callout(callout_eq);
            }
        }
        OdePreset::LinearSystem2D | OdePreset::PlanarCustom | OdePreset::Custom => {}
    }
}

/// Locates an interpolated zero crossing $y(t) = 0$ of the simulated trajectory near the horizon midpoint.
pub(crate) fn harmonic_equilibrium_root(tool: &OdeSolverTool) -> Option<f64> {
    if !tool.total_time.is_finite() {
        return None;
    }
    let pts: Vec<egui_plot::PlotPoint> = tool
        .time_series
        .iter()
        .copied()
        .zip(tool.y_series.iter().copied())
        .filter(|(t, y)| t.is_finite() && y.is_finite())
        .map(|(t, y)| egui_plot::PlotPoint::new(t, y))
        .collect();
    let crossings = egui_plot::poi::interpolate_threshold_crossings(&pts, 0.0);
    let mid = 0.5 * tool.total_time;
    crossings
        .into_iter()
        .filter(|t| t.is_finite() && *t >= 0.0 && *t <= tool.total_time)
        .min_by(|a, b| (a - mid).abs().total_cmp(&(b - mid).abs()))
        .or_else(|| {
            (tool.ic_y0.abs() <= 1e-12 && tool.ic_v0.abs() <= 1e-12).then_some(0.0)
        })
}

/// Renders simulation callout annotations onto the ODE plot canvas.
pub fn render_ode_callouts(plot_ui: &mut PlotUi<'_>, tool: &OdeSolverTool) {
    let (bg_fill, frame_stroke, text_color, palette) = callout_frame_styling(plot_ui);
    render_initial_condition_callout(
        plot_ui,
        tool.ic_y0,
        bg_fill,
        frame_stroke,
        text_color,
        &palette,
    );
    render_preset_asymptote_callout(plot_ui, tool, bg_fill, frame_stroke, text_color, &palette);
}
