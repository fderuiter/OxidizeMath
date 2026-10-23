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

    sensitivity_engine: crate::sensitivity::SensitivityEngine,
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
            sensitivity_engine: crate::sensitivity::SensitivityEngine::new(),
        };
        tool.recalculate();
        tool
    }
}

impl SirTool {
    fn render_parameter_controls(&mut self, ui: &mut egui::Ui) -> bool {
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
        changed
    }

    fn evaluate_sensitivity(&mut self, ui: &mut egui::Ui) {
        let params = vec![
            ("N".to_string(), self.n),
            ("I0".to_string(), self.i0),
            ("beta".to_string(), self.beta),
            ("gamma".to_string(), self.gamma),
        ];

        self.sensitivity_engine.compute(&params, |p_map| {
            let n = p_map.get("N").copied().unwrap_or(1000.0).max(1.0);
            let i0 = p_map.get("I0").copied().unwrap_or(10.0).clamp(0.0, n);
            let beta = p_map.get("beta").copied().unwrap_or(0.5).max(0.0);
            let gamma = p_map.get("gamma").copied().unwrap_or(0.1).max(0.0001);

            if let Ok(mut model) = SIRModel::new(n, i0, beta, gamma) {
                let dt = 0.2;
                let steps = 250;
                let mut max_i: f64 = 0.0;
                for _ in 0..steps {
                    max_i = max_i.max(model.state().i);
                    model.step(dt);
                }
                max_i
            } else {
                0.0
            }
        });

        crate::sensitivity::render_sensitivity_panel_ui(ui, &self.sensitivity_engine, "sir_tool");
    }

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

    fn show(&mut self, ctx: &egui::Context) {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            self.show_ui(ui);
        });
    }
    fn show_ui(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.heading("Parameters");
            if self.render_parameter_controls(ui) {
                self.recalculate();
            }

            let r0 = if self.gamma > 0.0 {
                self.beta / self.gamma
            } else {
                f64::INFINITY
            };
            render_copyable_metric(ui, "Basic Reproduction Number (R₀)", &format!("{:.2}", r0));

            self.evaluate_sensitivity(ui);

            ui.separator();
            ui.horizontal(|ui| {
                ui.heading("Simulation");
                if ui.button("Import Dataset (CSV)").clicked() {
                    egui_plot::export::trigger_import_csv_for_plot(ui.ctx(), egui::Id::new("sir_plot"));
                }
            });

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
