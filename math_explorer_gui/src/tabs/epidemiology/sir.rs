use crate::framework::{InteractiveTool, TraceSeries, TraceSnapshotManager};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::epidemiology::compartmental::SIRModel;

#[derive(serde::Serialize, serde::Deserialize)]
struct SirState {
    n: f64,
    i0: f64,
    beta: f64,
    gamma: f64,
    duration: f64,
    #[serde(default)]
    snapshot_manager: TraceSnapshotManager,
}

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

    snapshot_manager: TraceSnapshotManager,
    sensitivity_engine: crate::sensitivity::SensitivityEngine,
    last_param_summary: String,
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
            snapshot_manager: TraceSnapshotManager::new(),
            sensitivity_engine: crate::sensitivity::SensitivityEngine::new(),
            last_param_summary: String::new(),
        };
        tool.recalculate();
        tool
    }
}

impl SirTool {
    pub fn param_summary(&self) -> String {
        format!(r"\beta={:.2}, \gamma={:.2}, N={:.0}", self.beta, self.gamma, self.n)
    }

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
        self.last_param_summary = self.param_summary();
    }
}

impl InteractiveTool for SirTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "SIR Model"
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut datasets = egui_plot::get_last_plot_datasets();
        if datasets.is_empty() {
            datasets.push(("Susceptible".to_string(), self.s_points.clone()));
            datasets.push(("Infected".to_string(), self.i_points.clone()));
            datasets.push(("Recovered".to_string(), self.r_points.clone()));
            datasets.extend(self.snapshot_manager.export_datasets());
        }
        datasets
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&SirState {
            n: self.n,
            i0: self.i0,
            beta: self.beta,
            gamma: self.gamma,
            duration: self.duration,
            snapshot_manager: self.snapshot_manager.clone(),
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        if let Ok(s) = serde_json::from_str::<SirState>(state) {
            self.n = s.n;
            self.i0 = s.i0;
            self.beta = s.beta;
            self.gamma = s.gamma;
            self.duration = s.duration;
            self.snapshot_manager = s.snapshot_manager;
            self.recalculate();
        }
    }

    fn show(&mut self, ctx: &egui::Context) {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            self.show_ui(ui);
        });
    }

    #[allow(clippy::too_many_lines)]
    fn show_ui(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.heading("Parameters");
            if self.render_parameter_controls(ui) {
                let series = vec![
                    TraceSeries::new("Susceptible", self.s_points.clone()),
                    TraceSeries::new("Infected", self.i_points.clone()),
                    TraceSeries::new("Recovered", self.r_points.clone()),
                ];
                self.snapshot_manager.push_history_run(&self.last_param_summary, series);
                self.recalculate();
            }

            let r0 = if self.gamma > 0.0 {
                self.beta / self.gamma
            } else {
                f64::INFINITY
            };
            render_copyable_metric(ui, "Basic Reproduction Number (R₀)", &format!("{:.2}", r0));

            ui.add_space(8.0);
            let param_summary = self.param_summary();
            let s_pts = self.s_points.clone();
            let i_pts = self.i_points.clone();
            let r_pts = self.r_points.clone();
            let current_len = self.snapshot_manager.len();

            self.snapshot_manager.render_ui(ui, || {
                let label = format!("Baseline {}", current_len + 1);
                let series = vec![
                    TraceSeries::new("Susceptible", s_pts),
                    TraceSeries::new("Infected", i_pts),
                    TraceSeries::new("Recovered", r_pts),
                ];
                Some((label, param_summary, series))
            });

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
                self.snapshot_manager.render_on_plot(plot_ui);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sir_trace_snapshot_pinning_and_exports() {
        let mut tool = SirTool {
            beta: 0.2,
            ..Default::default()
        };
        tool.recalculate();

        // Pin snapshot for beta = 0.2
        let summary = tool.param_summary();
        let series = vec![
            TraceSeries::new("Susceptible", tool.s_points.clone()),
            TraceSeries::new("Infected", tool.i_points.clone()),
            TraceSeries::new("Recovered", tool.r_points.clone()),
        ];
        assert!(tool.snapshot_manager.add_snapshot("Run 1", summary, series));
        assert_eq!(tool.snapshot_manager.len(), 1);

        // Adjust beta to 0.5
        tool.beta = 0.5;
        tool.recalculate();

        // Snapshot remains preserved
        assert_eq!(tool.snapshot_manager.snapshots()[0].param_summary, r"\beta=0.20, \gamma=0.10, N=1000");

        // Export includes both active simulation curves and pinned snapshot traces
        let exported = tool.export_datasets();
        assert!(exported.iter().any(|(name, _)| name.contains("Infected")));
        assert!(exported.iter().any(|(name, _)| name.contains("Run 1")));

        // State serialization roundtrip preserves pinned snapshot
        let state = tool.save_state().expect("serialize state");
        let mut restored = SirTool::default();
        restored.load_state(&state);
        assert_eq!(restored.snapshot_manager.len(), 1);
        assert_eq!(restored.beta, 0.5);
    }
}
