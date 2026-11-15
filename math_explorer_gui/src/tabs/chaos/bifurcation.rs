use crate::accessibility::{AccessibleHoverText, AccessibleTheoryHover};
use crate::framework::InteractiveTool;
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
        "[cite:algorithmic_information_rust]".to_string()
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

impl InteractiveTool for BifurcationDiagram {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    
    fn name(&self) -> &'static str {
        "Bifurcation Diagram"
    }

    fn domain(&self) -> &'static str {
        "chaos"
    }

    fn save_state(&self) -> Option<String> {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct BifurcationState {
            r_min: f64,
            r_max: f64,
            steps: usize,
        }
        serde_json::to_string(&BifurcationState {
            r_min: self.r_min,
            r_max: self.r_max,
            steps: self.steps,
        }).ok()
    }

    fn load_state(&mut self, state: &str) {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct BifurcationState {
            r_min: f64,
            r_max: f64,
            steps: usize,
        }
        if let Ok(s) = serde_json::from_str::<BifurcationState>(state) {
            self.r_min = s.r_min;
            self.r_max = s.r_max;
            self.steps = s.steps;
            self.recompute();
        }
    }

    fn show(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("bifurcation_controls").show(ctx, |ui| {
            ui.heading("Logistic Map");
            ui.separator();

            let mut changed = false;
            changed |= ui.add(egui::Slider::new(&mut self.r_min, 0.0..=4.0).text("Min r"))
                .changed();

            changed |= ui.add(egui::Slider::new(&mut self.r_max, self.r_min..=4.0).text("Max r"))
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

        egui::CentralPanel::default().show(ctx, |ui| {
            let response = Plot::new("bifurcation_plot")
                .view_aspect(2.0)
                .x_axis_label("Growth Rate (r)")
                .y_axis_label("Population (x)")
                .show(ui, |plot_ui| {
                    plot_ui.points(
                        Points::new("Attractor", &self.points[..])
                            .radius(1.0_f32)
                            .color(egui::Color32::from_rgb(100, 200, 255)),
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
        name: "BifurcationDiagram",
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
}
