#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::OdeSolverTool;
use crate::presets::OdePreset;
use eframe::egui::{Color32, CornerRadius, Stroke, Vec2};
use egui_plot::{Callout, PlotUi};
use std::collections::HashMap;

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
    if tool.series_cache.matches_all(
        &tool.time_series,
        &tool.y_series,
        &tool.v_series,
        tool.total_time,
        tool.ic_y0,
        tool.ic_v0,
    ) {
        return tool.series_cache.harmonic_root();
    }
    if !tool.total_time.is_finite() {
        return None;
    }
    let series = (&tool.time_series[..], &tool.y_series[..]);
    let mid = 0.5 * tool.total_time;
    math_explorer::poi::closest_threshold_crossing(&series, 0.0, mid)
        .filter(|t| t.is_finite() && *t >= 0.0 && *t <= tool.total_time)
        .or_else(|| (tool.ic_y0.abs() <= 1e-12 && tool.ic_v0.abs() <= 1e-12).then_some(0.0))
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

impl scientific_metadata::theory::TheoryDescribable for OdeSolverTool {
    fn maturity_tier(&self) -> scientific_metadata::theory::MaturityTier {
        scientific_metadata::theory::MaturityTier::Certified
    }
    fn theory_description(&self) -> String {
        match self.preset {
            OdePreset::Exponential => "First-order linear ordinary differential equation describing exponential growth or decay proportional to current quantity.".into(),
            OdePreset::HarmonicOscillator => "Second-order ordinary differential equation modeling simple harmonic motion under a Hooke's law restoring force.".into(),
            OdePreset::LogisticGrowth => "Nonlinear ordinary differential equation modeling density-dependent population growth with an environmental carrying capacity.".into(),
            OdePreset::LinearSystem2D => "Planar 2x2 linear dynamical system classified on the Poincaré trace-determinant bifurcation diagram.".into(),
            OdePreset::PlanarCustom => "Coupled 2D planar ordinary differential equations: x' = f(t, x, y), y' = g(t, x, y).".into(),
            OdePreset::Custom if self.custom_coupled => "Coupled two-state ordinary differential equation system: y' = f(t, y, v), v' = g(t, y, v)".into(),
            OdePreset::Custom => "Custom user-defined ordinary differential equation: y' = f(t, y)".into(),
        }
    }
    fn phonetic_description(&self) -> String {
        "Ordinary differential equations theory".into()
    }
    fn theory_citation(&self) -> String {
        match self.preset {
            OdePreset::Exponential
            | OdePreset::LinearSystem2D
            | OdePreset::PlanarCustom
            | OdePreset::Custom => {
                "Differential Equations and Dynamical Systems (Perko, 2001)".into()
            }
            OdePreset::HarmonicOscillator => {
                "Classical Mechanics (Goldstein et al., 2002); Perko (2001)".into()
            }
            OdePreset::LogisticGrowth => {
                "Elements of Mathematical Biology (Lotka, 1925); Verhulst (1838)".into()
            }
        }
    }
    fn theory_latex_equation(&self) -> Option<String> {
        let eq = match self.preset {
            OdePreset::Exponential => r"\frac{dy}{dt} = k y",
            OdePreset::HarmonicOscillator => r"\frac{d^2 y}{dt^2} = -k y",
            OdePreset::LogisticGrowth => r"\frac{dy}{dt} = r y (1 - \frac{y}{K})",
            OdePreset::LinearSystem2D => r"\dot{\mathbf{x}} = A \mathbf{x}",
            OdePreset::PlanarCustom | OdePreset::Custom => return None,
        };
        Some(eq.into())
    }
    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::from([
            (
                "Exponential".into(),
                "First-order linear ODE representing exponential growth or decay".into(),
            ),
            (
                "Harmonic Oscillator".into(),
                "Second-order linear ODE representing simple harmonic motion".into(),
            ),
            (
                "Logistic Growth".into(),
                "Nonlinear first-order ODE modeling population growth with carrying capacity"
                    .into(),
            ),
            (
                "Linear 2D".into(),
                "Planar 2x2 linear system with Poincaré stability regimes".into(),
            ),
            (
                "Planar Custom".into(),
                "Coupled 2D user-defined ODE phase-plane system".into(),
            ),
            (
                "Custom".into(),
                "Custom user-defined scalar or coupled ordinary differential equation".into(),
            ),
        ])
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "harmonic_oscillator",
        title: "Harmonic Oscillator",
        description: "y'' = -k·y: Second-order differential equation modeling undamped periodic spring motion.",
        category: "Differential Equations",
        icon: "〰️",
        domain: "analysis",
        tool_name: Some("OdeSolverTool"),
        preset_generator: || crate::starter_gallery::PRESET_HARMONIC_OSCILLATOR,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "logistic_growth",
        title: "Logistic Growth",
        description: "y' = r·y·(1 - y/K): Non-linear growth constrained by carrying capacity K.",
        category: "Population Dynamics",
        icon: "📈",
        domain: "analysis",
        tool_name: Some("OdeSolverTool"),
        preset_generator: || crate::starter_gallery::PRESET_LOGISTIC_GROWTH,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "exponential_decay",
        title: "Exponential Decay",
        description: "y' = k·y: First-order differential decay process with constant half-life.",
        category: "Calculus",
        icon: "📉",
        domain: "analysis",
        tool_name: Some("OdeSolverTool"),
        preset_generator: || crate::starter_gallery::PRESET_EXPONENTIAL_DECAY,
    }
}
