#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{bounded_drag_value, enforce_ordered_drag_range, InteractiveTool};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Polygon};
use math_explorer::pure_math::analysis::integration::{Integrator, Trapezoidal};

pub(crate) const RIEMANN_X_MIN_BOUND: f64 = -50.0;
pub(crate) const RIEMANN_X_MAX_BOUND: f64 = 50.0;
const RIEMANN_MIN_SPAN: f64 = 0.1;

#[derive(PartialEq, Eq, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) enum FunctionPreset {
    SinX,
    XSquared,
    EToX,
}

impl FunctionPreset {
    fn evaluate(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 0.0;
        }
        match self {
            FunctionPreset::SinX => x.sin(),
            FunctionPreset::XSquared => {
                let clamped = x.clamp(-1e150, 1e150);
                clamped * clamped
            }
            FunctionPreset::EToX => x.clamp(-700.0, 80.0).exp(),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            FunctionPreset::SinX => "f(x) = sin(x)",
            FunctionPreset::XSquared => "f(x) = x^2",
            FunctionPreset::EToX => "f(x) = e^x",
        }
    }
}

pub struct RiemannIntegrationTool {
    pub(crate) preset: FunctionPreset,
    pub(crate) min_x: f64,
    pub(crate) max_x: f64,
    pub(crate) steps: usize,
    pub(crate) result: f64,
    cached_trapezoid_polygons: Vec<Vec<[f64; 2]>>,
    cached_true_curve: Vec<[f64; 2]>,
}

impl Default for RiemannIntegrationTool {
    fn default() -> Self {
        let mut tool = Self {
            preset: FunctionPreset::SinX,
            min_x: 0.0,
            max_x: std::f64::consts::PI,
            steps: 10,
            result: 0.0,
            cached_trapezoid_polygons: Vec::new(),
            cached_true_curve: Vec::new(),
        };
        tool.recalculate();
        tool
    }
}

impl RiemannIntegrationTool {
    pub(crate) fn recalculate(&mut self) {
        if !self.min_x.is_finite()
            || !self.max_x.is_finite()
            || self.min_x >= self.max_x
            || self.steps == 0
        {
            self.result = 0.0;
            self.cached_trapezoid_polygons.clear();
            self.cached_true_curve.clear();
            return;
        }

        let clamped_min = self.min_x.clamp(-700.0, 80.0);
        let clamped_max = self.max_x.clamp(-700.0, 80.0);
        let integrator = Trapezoidal::new(self.steps);
        let f = |x: f64| self.preset.evaluate(x);
        let res = if matches!(self.preset, FunctionPreset::EToX)
            && (clamped_min != self.min_x || clamped_max != self.max_x)
        {
            if clamped_min < clamped_max {
                integrator.integrate(f, clamped_min, clamped_max, 1e-6).value
            } else {
                0.0
            }
        } else {
            integrator.integrate(f, self.min_x, self.max_x, 1e-6).value
        };
        self.result = if res.is_finite() { res } else { 0.0 };

        let span = self.max_x - self.min_x;
        if !span.is_finite() || span <= 0.0 {
            self.cached_trapezoid_polygons.clear();
            self.cached_true_curve.clear();
            return;
        }
        let plot_min = self.min_x - span * 0.1;
        let plot_max = self.max_x + span * 0.1;

        self.cached_true_curve = (0..=500)
            .map(|i| {
                let x = plot_min + (plot_max - plot_min) * (i as f64 / 500.0);
                [x, f(x)]
            })
            .filter(|pt| pt[0].is_finite() && pt[1].is_finite())
            .collect();

        let dx = span / (self.steps as f64);
        self.cached_trapezoid_polygons = (0..self.steps)
            .filter_map(|i| {
                let x0 = self.min_x + (i as f64) * dx;
                let x1 = x0 + dx;
                let y0 = f(x0);
                let y1 = f(x1);
                [x0, x1, y0, y1]
                    .iter()
                    .all(|v| v.is_finite())
                    .then(|| vec![[x0, 0.0], [x1, 0.0], [x1, y1], [x0, y0]])
            })
            .collect();
    }

    fn draw_preset_combo(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label("Function:");
        egui::ComboBox::from_id_salt("function_preset")
            .selected_text(self.preset.name())
            .show_ui(ui, |ui| {
                for (candidate, default_bounds) in [
                    (FunctionPreset::SinX, (0.0, std::f64::consts::PI)),
                    (FunctionPreset::XSquared, (-2.0, 2.0)),
                    (FunctionPreset::EToX, (-2.0, 2.0)),
                ] {
                    if ui
                        .selectable_value(&mut self.preset, candidate, candidate.name())
                        .changed()
                    {
                        self.min_x = default_bounds.0;
                        self.max_x = default_bounds.1;
                        changed = true;
                    }
                }
            });
        changed
    }

    fn draw_controls(&mut self, parent_ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "riemann_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Integration Controls");
            ui.separator();
            changed |= self.draw_preset_combo(ui);

            ui.add_space(10.0);
            ui.label("Integration Interval [min, max]");
            let mut min_changed = false;
            let mut max_changed = false;
            ui.horizontal(|ui| {
                min_changed = ui
                    .add(bounded_drag_value(
                        &mut self.min_x,
                        RIEMANN_X_MIN_BOUND..=RIEMANN_X_MAX_BOUND,
                        0.1,
                    ))
                    .changed();
                ui.label("to");
                max_changed = ui
                    .add(bounded_drag_value(
                        &mut self.max_x,
                        RIEMANN_X_MIN_BOUND..=RIEMANN_X_MAX_BOUND,
                        0.1,
                    ))
                    .changed();
            });
            enforce_ordered_drag_range(
                &mut self.min_x,
                &mut self.max_x,
                RIEMANN_X_MIN_BOUND,
                RIEMANN_X_MAX_BOUND,
                RIEMANN_MIN_SPAN,
                min_changed,
                max_changed,
            );
            changed |= min_changed || max_changed;

            ui.add_space(10.0);
            if ui
                .add(
                    egui::Slider::new(&mut self.steps, 1..=200)
                        .logarithmic(true)
                        .text("Partitions (N)"),
                )
                .changed()
            {
                changed = true;
            }

            ui.separator();
            ui.heading("Result");
            render_copyable_metric(ui, "Area", &format!("{:.6}", self.result));
        });
        changed
    }

    fn draw_plot(&self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            ui.heading("Visualization");
            let valid_bounds =
                self.min_x.is_finite() && self.max_x.is_finite() && self.max_x > self.min_x;
            let (plot_min, plot_max) = if valid_bounds {
                let span = self.max_x - self.min_x;
                (self.min_x - span * 0.1, self.max_x + span * 0.1)
            } else {
                (-1.0, 1.0)
            };

            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let fill_base = palette.series_color(1);
            let poly_fill = egui::Color32::from_rgba_unmultiplied(
                fill_base.r(),
                fill_base.g(),
                fill_base.b(),
                100,
            );
            let true_line = Line::new(
                self.preset.name(),
                PlotPoints::new(self.cached_true_curve.clone()),
            )
            .name(self.preset.name())
            .color(palette.series_color(0));

            Plot::new("riemann_plot")
                .view_aspect(2.0)
                .x_axis_label("x")
                .y_axis_label("f(x)")
                .show(ui, |plot_ui| {
                    if valid_bounds {
                        plot_ui.line(true_line);
                        for poly in &self.cached_trapezoid_polygons {
                            let polygon = Polygon::new("", PlotPoints::new(poly.clone()))
                                .fill_color(poly_fill);
                            plot_ui.polygon(polygon);
                        }
                    }
                    plot_ui.line(
                        Line::new("", PlotPoints::new(vec![[plot_min, 0.0], [plot_max, 0.0]]))
                            .color(palette.graph_edge),
                    );
                });
        });
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RiemannState {
    preset: FunctionPreset,
    min_x: f64,
    max_x: f64,
    steps: usize,
}

impl InteractiveTool for RiemannIntegrationTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Riemann Integration"
    }

    fn domain(&self) -> &'static str {
        "analysis"
    }

    fn has_finite_parameters(&self) -> bool {
        self.min_x.is_finite()
            && self.max_x.is_finite()
            && (RIEMANN_X_MIN_BOUND..=RIEMANN_X_MAX_BOUND).contains(&self.min_x)
            && (RIEMANN_X_MIN_BOUND..=RIEMANN_X_MAX_BOUND).contains(&self.max_x)
            && self.min_x < self.max_x
            && (1..=200).contains(&self.steps)
            && self.result.is_finite()
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&RiemannState {
            preset: self.preset,
            min_x: self.min_x,
            max_x: self.max_x,
            steps: self.steps,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: RiemannState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Riemann state: {e}"))?;
        if !s.min_x.is_finite()
            || !s.max_x.is_finite()
            || !(RIEMANN_X_MIN_BOUND..=RIEMANN_X_MAX_BOUND).contains(&s.min_x)
            || !(RIEMANN_X_MIN_BOUND..=RIEMANN_X_MAX_BOUND).contains(&s.max_x)
            || s.min_x >= s.max_x
            || !(1..=200).contains(&s.steps)
        {
            return Err("Riemann integration interval or partition count invalid".into());
        }
        self.preset = s.preset;
        self.min_x = s.min_x;
        self.max_x = s.max_x;
        self.steps = s.steps;
        self.recalculate();
        Ok(())
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        if self.draw_controls(parent_ui) {
            self.recalculate();
        }
        self.draw_plot(parent_ui);
    }
}

// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "RiemannIntegrationTool",
        display_name: "Riemann Integration",
        domain: "analysis",
        tags: &[],
        build: || Box::new(RiemannIntegrationTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for RiemannIntegrationTool {
    fn theory_description(&self) -> String {
        format!(
            "Numerical approximation of the definite integral $\\int_a^b f(x) dx$ for {} using composite trapezoidal rule: $T_n = \\sum_{{i=1}}^n \\frac{{f(x_{{i-1}}) + f(x_i)}}{{2}} \\Delta x$ with {} steps over interval [{:.2}, {:.2}].",
            self.preset.name(),
            self.steps,
            self.min_x,
            self.max_x
        )
    }

    fn phonetic_description(&self) -> String {
        "Riemann integration and composite trapezoidal approximation theory".into()
    }

    fn theory_citation(&self) -> String {
        "Calculus, Vol. 1 (Apostol, 1967); Principles of Mathematical Analysis (Rudin, 1976)"
            .into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "SinX".into(),
            "Integral of sin(x) showing oscillatory area cancellation".into(),
        );
        map.insert(
            "XSquared".into(),
            "Integral of x^2 demonstrating parabolic quadrature".into(),
        );
        map.insert(
            "EToX".into(),
            "Integral of e^x demonstrating exponential growth integration".into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Composite Trapezoidal Integration",
    "domain": "analysis",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "preset": "SinX",
    "min_x": 0.0,
    "max_x": 3.141592653589793,
    "steps": 20
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "riemann_integration_starter",
        title: "Composite Trapezoidal Integration",
        description: "Definite integral numerical quadrature via composite trapezoidal rule partitions.",
        category: "Integral Calculus",
        icon: "📐",
        domain: "analysis",
        tool_name: Some("RiemannIntegrationTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

#[cfg(test)]
#[path = "riemann/tests.rs"]
mod tests;
