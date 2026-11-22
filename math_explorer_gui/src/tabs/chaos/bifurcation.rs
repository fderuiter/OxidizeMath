use crate::accessibility::{AccessibleHoverText, AccessibleTheoryHover};
use crate::framework::InteractiveTool;
use crate::reflective_ui::AnimatedSlider;
use eframe::egui;
use egui_plot::{Plot, PlotPoint, Points};
use scientific_metadata::theory::TheoryDescribable;
use math_explorer::physics::chaos::logistic;
use std::collections::HashMap;

pub struct BifurcationDiagram {
    r_min: f64,
    r_max: f64,
    steps: usize,
    points: Vec<PlotPoint>,
}

impl Default for BifurcationDiagram {
    fn default() -> Self {
        let mut tool = Self {
            r_min: 2.5,
            r_max: 4.0,
            steps: 500,
            points: Vec::new(),
        };
        tool.recompute();
        tool
    }
}

impl BifurcationDiagram {
    fn recompute(&mut self) {
        let raw_points = logistic::generate_bifurcation_diagram(self.r_min, self.r_max, self.steps);
        self.points = raw_points.into_iter().map(|(r, x)| PlotPoint::new(r, x)).collect();
    }
}

impl TheoryDescribable for BifurcationDiagram {
    fn theory_description(&self) -> String {
        format!(
            "Logistic map bifurcation diagram, r range: {:.2} to {:.2}",
            self.r_min, self.r_max
        )
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

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BifurcationState {
    r_min: f64,
    r_max: f64,
    steps: usize,
}

impl InteractiveTool for BifurcationDiagram {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    
    fn name(&self) -> &'static str {
        "Bifurcation Diagram"
    }

    fn domain(&self) -> &'static str {
        "chaos"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&BifurcationState {
            r_min: self.r_min,
            r_max: self.r_max,
            steps: self.steps,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: BifurcationState =
            serde_json::from_str(state).map_err(|e| format!("Invalid bifurcation state: {e}"))?;
        if !s.r_min.is_finite()
            || !s.r_max.is_finite()
            || !(0.0..=4.0).contains(&s.r_min)
            || !(0.0..=4.0).contains(&s.r_max)
            || s.r_min >= s.r_max
            || !(100..=2000).contains(&s.steps)
        {
            return Err("Bifurcation r bounds or step count invalid".into());
        }
        self.r_min = s.r_min;
        self.r_max = s.r_max;
        self.steps = s.steps;
        self.recompute();
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "bifurcation_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            ui.heading("Logistic Map");
            ui.separator();

            let mut changed = false;
            changed |= ui
                .add(
                    AnimatedSlider::new(&mut self.r_min, 0.0..=4.0)
                        .text("Min r")
                        .id_salt("bifurcation_r_min"),
                )
                .changed();

            changed |= ui
                .add(
                    AnimatedSlider::new(&mut self.r_max, self.r_min..=4.0)
                        .text("Max r")
                        .id_salt("bifurcation_r_max"),
                )
                .changed();

            changed |= ui.add(egui::Slider::new(&mut self.steps, 100..=2000).text("Resolution (steps)"))
                .changed();

            if ui
                .button("▶ Recompute")
                .accessible_hover_text(
                    "Regenerate the bifurcation diagram with the current parameters",
                )
                .clicked()
                || changed
            {
                self.recompute();
            }

            ui.separator();
            ui.label("Zoom: Use mouse wheel to zoom, drag to pan.");
            ui.label("Double-click to reset view.");
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let response = Plot::new("bifurcation_plot")
                .view_aspect(2.0)
                .x_axis_label("Growth Rate (r)")
                .y_axis_label("Population (x)")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bifurcation_recompute_and_borrowed_slice() {
        let mut tool = BifurcationDiagram::default();
        assert!(!tool.points.is_empty());

        let count_initial = tool.points.len();
        tool.steps = 200;
        tool.recompute();
        assert_ne!(tool.points.len(), count_initial);

        let slice: &[PlotPoint] = &tool.points[..];
        let plot_points: egui_plot::PlotPoints<'_> = slice.into();
        assert!(matches!(plot_points, egui_plot::PlotPoints::Borrowed(_)));
        assert_eq!(plot_points.points().len(), tool.points.len());
    }

    #[test]
    fn test_bifurcation_animated_slider_sweep() {
        use crate::reflective_ui::AnimatedSliderState;
        let mut tool = BifurcationDiagram::default();
        let ctx = egui::Context::default();
        let initial_r_min = tool.r_min;

        // Simulate activating animation on r_min slider
        let id = egui::Id::new("bifurcation_r_min");
        ctx.data_mut(|d| {
            d.insert_temp(
                id,
                AnimatedSliderState {
                    playing: true,
                    direction: 1.0,
                    ..Default::default()
                },
            );
        });

        let raw_input = egui::RawInput {
            predicted_dt: 0.1,
            ..Default::default()
        };
        let _ = ctx.run(raw_input, |ctx| {
            tool.show(ctx);
        });

        // r_min should have increased, and recompute was triggered
        assert!(tool.r_min > initial_r_min);

        let points_vec: Vec<[f64; 2]> = tool.points.iter().map(|p| [p.x, p.y]).collect();
        let datasets = vec![("Bifurcation Diagram (Logistic Map)".to_string(), points_vec)];
        if let Ok(bytes) = egui_plot::export::export_custom_png(
            &datasets,
            &egui_plot::export::ExportOptions::default(),
        ) {
            let _ = std::fs::write("/tmp/bifurcation_diagram_animated.png", bytes);
        }
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
