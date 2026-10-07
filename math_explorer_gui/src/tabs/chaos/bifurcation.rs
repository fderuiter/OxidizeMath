use crate::accessibility::{AccessibleHoverText, AccessibleTheoryHover};
use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::{Plot, PlotPoints, Points};
use scientific_metadata::theory::TheoryDescribable;
use math_explorer::physics::chaos::logistic;
use std::collections::HashMap;

pub struct BifurcationDiagram {
    r_min: f64,
    r_max: f64,
    steps: usize,
    points: Vec<[f64; 2]>,
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
        self.points = raw_points.into_iter().map(|(r, x)| [r, x]).collect();
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
            ui.label("Zoom: Use mouse wheel or on-canvas ➕ / ➖ buttons, drag to pan.");
            ui.label("Reset view: Double-click or click on-canvas ⌂ Reset button.");
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let response = Plot::new("bifurcation_plot")
                .view_aspect(2.0)
                .x_axis_label("Growth Rate (r)")
                .y_axis_label("Population (x)")
                .show(ui, |plot_ui| {
                    plot_ui.points(
                        Points::new("Attractor", PlotPoints::new(self.points.clone()))
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
