use crate::accessibility::{AccessibleHoverText, AccessibleTheoryHover};
use crate::framework::{bounded_drag_value, InteractiveTool, MathExpressionInputBar};
use crate::reflective_ui::AnimatedSlider;
use eframe::egui;
use egui_plot::{Plot, PlotPoint, Points};
use math_explorer::physics::chaos::logistic;
use scientific_metadata::theory::TheoryDescribable;
use std::collections::{BTreeMap, HashMap};

const DEFAULT_CUSTOM_MAP_EXPR: &str = "r * sin(pi * x)";
const CUSTOM_BURN_IN: usize = 200;
const CUSTOM_ORBIT_SAMPLES: usize = 100;
const CUSTOM_ESCAPE_GUARD: f64 = 1e6;
const CUSTOM_R_BOUND: f64 = 20.0;

/// Map family evaluated by [`BifurcationDiagram`].
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum BifurcationMapMode {
    /// Canonical logistic map $x_{n+1} = r x_n (1 - x_n)$ on $r \in [0, 4]$.
    #[default]
    Logistic,
    /// User-defined 1D iteration map $x_{n+1} = f(r, x)$ evaluated via `CompiledExpr::eval_2d`.
    Custom,
}

impl BifurcationMapMode {
    const fn is_logistic(&self) -> bool {
        matches!(self, Self::Logistic)
    }
}

pub struct BifurcationDiagram {
    map_mode: BifurcationMapMode,
    r_min: f64,
    r_max: f64,
    steps: usize,
    x0: f64,
    math_input: MathExpressionInputBar,
    points: Vec<PlotPoint>,
}

impl Default for BifurcationDiagram {
    fn default() -> Self {
        let math_input = MathExpressionInputBar::new(
            "x_{n+1}",
            DEFAULT_CUSTOM_MAP_EXPR,
            vec![
                ("Sine Map", "r * sin(pi * x)"),
                ("Quadratic Map", "x^2 - r"),
                ("Ricker Population", "x * exp(r * (1 - x))"),
                ("Cubic Map", "r * x - x^3"),
                ("Perturbed Logistic", "r * x * (1 - x) + a * sin(2 * pi * x)"),
            ],
        )
        .with_axes(&["r", "x"])
        .with_param_range(-2.0..=2.0, 0.05);

        let mut tool = Self {
            map_mode: BifurcationMapMode::Logistic,
            r_min: 2.5,
            r_max: 4.0,
            steps: 500,
            x0: 0.5,
            math_input,
            points: Vec::new(),
        };
        tool.recompute();
        tool
    }
}

impl BifurcationDiagram {
    fn recompute(&mut self) {
        if self.map_mode == BifurcationMapMode::Logistic {
            let raw_points =
                logistic::generate_bifurcation_diagram(self.r_min, self.r_max, self.steps);
            self.points = raw_points
                .into_iter()
                .map(|(r, x)| PlotPoint::new(r, x))
                .collect();
            return;
        }

        let Some(compiled) = self.math_input.active_compiled() else {
            return;
        };
        let params = self.math_input.param_slice();
        let steps = self.steps.clamp(2, 2000);
        let span = self.r_max - self.r_min;
        if !span.is_finite() || span <= 0.0 {
            self.points.clear();
            return;
        }

        let max_cap = steps.saturating_mul(CUSTOM_ORBIT_SAMPLES);
        if self.points.capacity() < max_cap {
            self.points.reserve(max_cap - self.points.capacity());
        }
        self.points.clear();

        let denom = (steps - 1) as f64;
        for step_idx in 0..steps {
            let r = self.r_min + (step_idx as f64 / denom) * span;
            let mut x = self.x0;
            let mut bounded = x.is_finite() && x.abs() <= CUSTOM_ESCAPE_GUARD;
            for _ in 0..CUSTOM_BURN_IN {
                if !bounded {
                    break;
                }
                x = compiled.eval_2d(r, x, params);
                bounded = x.is_finite() && x.abs() <= CUSTOM_ESCAPE_GUARD;
            }
            if !bounded {
                continue;
            }
            for _ in 0..CUSTOM_ORBIT_SAMPLES {
                x = compiled.eval_2d(r, x, params);
                if !x.is_finite() || x.abs() > CUSTOM_ESCAPE_GUARD {
                    break;
                }
                self.points.push(PlotPoint::new(r, x));
            }
        }
    }

    fn draw_map_mode_selector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            if ui
                .radio_value(&mut self.map_mode, BifurcationMapMode::Logistic, "Logistic")
                .clicked()
            {
                self.r_min = self.r_min.clamp(0.0, 3.95);
                self.r_max = self.r_max.clamp(self.r_min + 0.05, 4.0);
                changed = true;
            }
            if ui
                .radio_value(
                    &mut self.map_mode,
                    BifurcationMapMode::Custom,
                    "Custom f(r, x)",
                )
                .clicked()
            {
                changed = true;
            }
        });
        if self.map_mode == BifurcationMapMode::Custom {
            ui.add_space(4.0);
            changed |= self.math_input.ui(ui);
            changed |= ui
                .add(bounded_drag_value(&mut self.x0, -10.0..=10.0, 0.02).prefix("Initial x₀: "))
                .changed();
        }
        changed
    }

    fn draw_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("1D Map Bifurcation");
        ui.separator();

        let mut changed = self.draw_map_mode_selector(ui);
        let (lo_bound, hi_bound) = match self.map_mode {
            BifurcationMapMode::Logistic => (0.0, 4.0),
            BifurcationMapMode::Custom => (-6.0, 6.0),
        };
        changed |= ui
            .add(
                AnimatedSlider::new(&mut self.r_min, lo_bound..=hi_bound)
                    .text("Min r")
                    .id_salt("bifurcation_r_min"),
            )
            .changed();
        if self.r_max <= self.r_min {
            self.r_max = (self.r_min + 0.05).min(hi_bound);
        }
        changed |= ui
            .add(
                AnimatedSlider::new(&mut self.r_max, self.r_min..=hi_bound)
                    .text("Max r")
                    .id_salt("bifurcation_r_max"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.steps, 100..=2000).text("Resolution (steps)"))
            .changed();

        if ui
            .button("▶ Recompute")
            .accessible_hover_text("Regenerate the bifurcation diagram with the current parameters")
            .clicked()
            || changed
        {
            self.recompute();
        }

        ui.separator();
        ui.label("Zoom: Use mouse wheel to zoom, drag to pan.");
        ui.label("Double-click to reset view.");
    }
}

impl TheoryDescribable for BifurcationDiagram {
    fn theory_description(&self) -> String {
        match self.map_mode {
            BifurcationMapMode::Logistic => format!(
                "Logistic map bifurcation diagram, r range: {:.2} to {:.2}",
                self.r_min, self.r_max
            ),
            BifurcationMapMode::Custom => format!(
                "Custom 1D map bifurcation x_{{n+1}} = {}, r range: {:.2} to {:.2}",
                self.math_input.raw_text, self.r_min, self.r_max
            ),
        }
    }

    fn phonetic_description(&self) -> String {
        self.theory_description()
    }

    fn theory_citation(&self) -> String {
        "May, R. M. (1976). Simple mathematical models with very complicated dynamics. Nature, 261(5560), 459-467; Feigenbaum, M. J. (1978). Quantitative universality for a class of nonlinear transformations. Journal of Statistical Physics, 19(1), 25-52.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "default".to_string(),
            "Logistic map bifurcation diagram".to_string(),
        );
        map
    }
}

const fn default_x0() -> f64 {
    0.5
}

fn is_default_x0(v: &f64) -> bool {
    (*v - 0.5).abs() < 1e-12
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BifurcationState {
    r_min: f64,
    r_max: f64,
    steps: usize,
    #[serde(default, skip_serializing_if = "BifurcationMapMode::is_logistic")]
    map_mode: BifurcationMapMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    custom_expr: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    custom_params: BTreeMap<String, f64>,
    #[serde(default = "default_x0", skip_serializing_if = "is_default_x0")]
    x0: f64,
}

impl InteractiveTool for BifurcationDiagram {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Bifurcation Diagram"
    }

    fn domain(&self) -> &'static str {
        "chaos"
    }

    fn save_state(&self) -> Option<String> {
        let is_custom = self.map_mode == BifurcationMapMode::Custom;
        let custom_expr = (is_custom || self.math_input.raw_text != DEFAULT_CUSTOM_MAP_EXPR)
            .then(|| self.math_input.raw_text.clone());
        let custom_params = if is_custom {
            self.math_input.params_map()
        } else {
            BTreeMap::new()
        };
        serde_json::to_string(&BifurcationState {
            r_min: self.r_min,
            r_max: self.r_max,
            steps: self.steps,
            map_mode: self.map_mode,
            custom_expr,
            custom_params,
            x0: self.x0,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: BifurcationState =
            serde_json::from_str(state).map_err(|e| format!("Invalid bifurcation state: {e}"))?;
        let r_bounds = match s.map_mode {
            BifurcationMapMode::Logistic => 0.0..=4.0,
            BifurcationMapMode::Custom => -CUSTOM_R_BOUND..=CUSTOM_R_BOUND,
        };
        if !s.r_min.is_finite()
            || !s.r_max.is_finite()
            || !r_bounds.contains(&s.r_min)
            || !r_bounds.contains(&s.r_max)
            || s.r_min >= s.r_max
            || !(100..=2000).contains(&s.steps)
            || !s.x0.is_finite()
            || !(-10.0..=10.0).contains(&s.x0)
        {
            return Err("Bifurcation r bounds, initial seed, or step count invalid".into());
        }
        self.map_mode = s.map_mode;
        self.r_min = s.r_min;
        self.r_max = s.r_max;
        self.steps = s.steps;
        self.x0 = s.x0;
        self.math_input
            .set_expression(s.custom_expr.as_deref().unwrap_or(DEFAULT_CUSTOM_MAP_EXPR));
        self.math_input.set_params_map(&s.custom_params);
        self.recompute();
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "bifurcation_controls");
        crate::framework::PaneSidePanel::left(panel_id)
            .show_inside(ui, |ui| self.draw_controls(ui));

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let response = Plot::new("bifurcation_plot")
                .view_aspect(2.0)
                .x_axis_label("Control Parameter (r)")
                .y_axis_label("State (x)")
                .show(ui, |plot_ui| {
                    plot_ui.points(
                        Points::new("Attractor", &self.points[..])
                            .radius(1.0_f32)
                            .color(palette.series_color(0)),
                    );
                })
                .response;
            response.accessible_theory_hover(self);
        });
    }
}

// [cite:algorithmic_information_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "BifurcationDiagram",
        display_name: "Bifurcation Diagram",
        domain: "chaos",
        tags: &[],
        build: || Box::new(BifurcationDiagram::default()),
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Bifurcation Diagram",
    "domain": "chaos",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "r_min": 2.5,
    "r_max": 4.0,
    "steps": 800
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "bifurcation",
        title: "Bifurcation Diagram",
        description: "x_{n+1} = r·x_n·(1 - x_n): Period-doubling cascade towards deterministic chaos.",
        category: "Non-Linear Dynamics",
        icon: "🌿",
        domain: "chaos",
        tool_name: Some("BifurcationDiagram"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

#[cfg(test)]
#[path = "bifurcation/tests.rs"]
mod tests;
