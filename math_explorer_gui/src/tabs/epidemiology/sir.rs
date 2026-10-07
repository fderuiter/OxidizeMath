use crate::framework::InteractiveTool;
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::epidemiology::compartmental::SIRModel;

pub struct SirTool {
    n: f64,
    i0: f64,
    beta: f64,
    gamma: f64,
    duration: f64,

    // Cached plot data
    s_points: Vec<[f64; 2]>,
    i_points: Vec<[f64; 2]>,
    r_points: Vec<[f64; 2]>,
}

impl Default for SirTool {
    fn default() -> Self {
        let mut tool = Self {
            n: 1000.0,
            i0: 10.0,
            beta: 0.5,
            gamma: 0.1,
            duration: 100.0,
            s_points: vec![],
            i_points: vec![],
            r_points: vec![],
        };
        tool.recalculate();
        tool
    }
}

impl SirTool {
    fn recalculate(&mut self) {
        self.s_points.clear();
        self.i_points.clear();
        self.r_points.clear();

        // Ensure parameters are valid before passing to model
        let n = self.n.max(1.0);
        let i0 = self.i0.clamp(0.0, n);
        let beta = self.beta.max(0.0);
        let gamma = self.gamma.max(0.0);

        if let Ok(mut model) = SIRModel::new(n, i0, beta, gamma) {
            let dt = 0.1;
            let steps = (self.duration / dt) as usize;

            for i in 0..=steps {
                let t = i as f64 * dt;
                let state = model.state();
                self.s_points.push([t, state.s]);
                self.i_points.push([t, state.i]);
                self.r_points.push([t, state.r]);

                model.step(dt);
            }
        }
    }
}

impl InteractiveTool for SirTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "SIR Model"
    }

    fn export_data(&self) -> Option<crate::framework::ExportableData> {
        let datasets = vec![
            ("Susceptible (S)".to_string(), self.s_points.clone()),
            ("Infected (I)".to_string(), self.i_points.clone()),
            ("Recovered (R)".to_string(), self.r_points.clone()),
        ];
        Some(crate::framework::ExportableData::series(
            "SIR Model Simulation",
            vec!["Time (days)".into(), "Individuals".into()],
            datasets,
        ))
    }

    fn show(&mut self, ctx: &egui::Context) {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            self.show_ui(ui);
        });
    }
    fn show_ui(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.heading("Parameters");
            let mut changed = false;

            changed |= ui
                .add(egui::Slider::new(&mut self.n, 100.0..=100_000.0).text("Population (N)"))
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.i0, 1.0..=self.n / 2.0)
                        .text("Initial Infected (I0)"),
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.beta, 0.0..=5.0).text("Transmission Rate (beta)"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.gamma, 0.0..=1.0).text("Recovery Rate (gamma)"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.duration, 10.0..=365.0).text("Duration (days)"))
                .changed();

            if changed {
                self.recalculate();
            }

            let r0 = if self.gamma > 0.0 {
                self.beta / self.gamma
            } else {
                f64::INFINITY
            };
            render_copyable_metric(ui, "Basic Reproduction Number (R₀)", &format!("{:.2}", r0));

            ui.separator();
            ui.heading("Simulation");

            let plot = Plot::new("sir_plot")
                .view_aspect(2.0)
                .legend(egui_plot::Legend::default());

            plot.show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Susceptible", PlotPoints::new(self.s_points.clone()))
                        .color(egui::Color32::BLUE),
                );
                plot_ui.line(
                    Line::new("Infected", PlotPoints::new(self.i_points.clone()))
                        .color(egui::Color32::RED),
                );
                plot_ui.line(
                    Line::new("Recovered", PlotPoints::new(self.r_points.clone()))
                        .color(egui::Color32::GREEN),
                );
            });
        });
    }
}

// [cite:graph_parameters_rust]


inventory::submit! {
    crate::framework::ToolMetadata {
        name: "SirTool",
        domain: "epidemiology",
        tags: &[],
        build: || Box::new(SirTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for SirTool {
    fn theory_description(&self) -> String { "Theoretical context not available.".into() }
    fn phonetic_description(&self) -> String { "Theoretical context not available.".into() }
    fn theory_citation(&self) -> String { "Uncited".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}
