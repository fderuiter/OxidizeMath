#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::OdeSolverTool;
use crate::presets::OdePreset;
use eframe::egui::{Color32, CornerRadius, Stroke, Vec2};
use egui_plot::{Callout, PlotUi};

fn callout_frame_styling(plot_ui: &PlotUi<'_>) -> (Color32, Stroke, Color32) {
    let visuals = plot_ui.ctx().style().visuals.clone();
    let bg_fill = Color32::from_rgba_unmultiplied(
        visuals.window_fill().r(),
        visuals.window_fill().g(),
        visuals.window_fill().b(),
        230,
    );
    let frame_stroke = visuals.widgets.noninteractive.bg_stroke;
    let text_color = visuals.text_color();
    (bg_fill, frame_stroke, text_color)
}

fn render_initial_condition_callout(
    plot_ui: &mut PlotUi<'_>,
    y0: f64,
    bg_fill: Color32,
    frame_stroke: Stroke,
    text_color: Color32,
) {
    if !y0.is_finite() {
        return;
    }
    let callout_ic = Callout::new([0.0, y0], format!("Initial Condition: y(0) = {:.2}", y0))
        .offset(Vec2::new(35.0, -30.0))
        .leader_stroke(Stroke::new(1.5_f32, Color32::from_rgb(52, 152, 219)))
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
) {
    match tool.preset {
        OdePreset::LogisticGrowth => {
            if tool.param_cap_k.is_finite() && tool.total_time.is_finite() {
                let callout_cap = Callout::new(
                    [tool.total_time, tool.param_cap_k],
                    format!("Carrying Capacity: K = {:.2}", tool.param_cap_k),
                )
                .offset(Vec2::new(-45.0, -25.0))
                .leader_stroke(Stroke::new(1.5_f32, Color32::from_rgb(46, 204, 113)))
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
                    .leader_stroke(Stroke::new(1.5_f32, Color32::from_rgb(230, 126, 34)))
                    .frame(bg_fill, frame_stroke)
                    .corner_radius(CornerRadius::same(4))
                    .text_color(text_color);

                plot_ui.callout(callout_ss);
            }
        }
        OdePreset::HarmonicOscillator => {
            if tool.total_time.is_finite() {
                let callout_eq = Callout::new([tool.total_time * 0.5, 0.0], "Equilibrium: y = 0.00")
                    .offset(Vec2::new(30.0, 25.0))
                    .leader_stroke(Stroke::new(1.5_f32, Color32::from_rgb(230, 126, 34)))
                    .frame(bg_fill, frame_stroke)
                    .corner_radius(CornerRadius::same(4))
                    .text_color(text_color);

                plot_ui.callout(callout_eq);
            }
        }
    }
}

/// Renders simulation callout annotations onto the ODE plot canvas.
pub fn render_ode_callouts(plot_ui: &mut PlotUi<'_>, tool: &OdeSolverTool) {
    let (bg_fill, frame_stroke, text_color) = callout_frame_styling(plot_ui);
    render_initial_condition_callout(plot_ui, tool.ic_y0, bg_fill, frame_stroke, text_color);
    render_preset_asymptote_callout(plot_ui, tool, bg_fill, frame_stroke, text_color);
}
