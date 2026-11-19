use crate::framework::InteractiveTool;
use crate::presets::{InteractivePreset, ReplicatorPresetConfig};
use eframe::egui;
use math_explorer::applied::game_theory::evolutionary::ReplicatorDynamics;
use nalgebra::{DMatrix, DVector};

#[path = "replicator/ui.rs"]
mod ui;

pub struct ReplicatorDynamicsTool {
    payoff_matrix: DMatrix<f64>,
    initial_population: DVector<f64>,
    trajectory: Vec<(f64, DVector<f64>)>,
    time_horizon: f64,
    dt: f64,
    strategy_names: Vec<String>,
    playback_time: f64,
    is_playing: bool,
    playback_speed: f64,
}

impl Default for ReplicatorDynamicsTool {
    fn default() -> Self {
        let payoff = DMatrix::from_row_slice(2, 2, &[-1.0, 2.0, 0.0, 1.0]);
        let init = DVector::from_vec(vec![0.1, 0.9]);
        let mut tool = Self {
            payoff_matrix: payoff,
            initial_population: init,
            trajectory: Vec::new(),
            time_horizon: 20.0,
            dt: 0.05,
            strategy_names: vec!["Hawk".to_string(), "Dove".to_string()],
            playback_time: 0.0,
            is_playing: false,
            playback_speed: 1.0,
        };
        tool.run_simulation();
        tool
    }
}

impl ReplicatorDynamicsTool {
    pub(crate) fn run_simulation(&mut self) {
        let sum = self.initial_population.sum();
        if sum > 0.0 {
            self.initial_population /= sum;
        } else {
            let n = self.initial_population.len();
            self.initial_population = DVector::from_element(n, 1.0 / n as f64);
        }

        if let Ok(system) = ReplicatorDynamics::new(self.payoff_matrix.clone()) {
            self.trajectory =
                system.simulate(self.initial_population.clone(), self.time_horizon, self.dt);
            self.playback_time = 0.0;
            self.is_playing = false;
        }
    }

    pub(crate) fn step_once(&mut self) {
        self.is_playing = false;
        if self.trajectory.is_empty() {
            return;
        }
        self.playback_time += self.dt;
        if self.playback_time > self.time_horizon {
            self.playback_time = 0.0;
        }
    }

    pub(crate) fn reset_playback(&mut self) {
        self.is_playing = false;
        self.playback_time = 0.0;
    }

    pub(crate) fn load_preset_hawk_dove(&mut self) {
        self.payoff_matrix = DMatrix::from_row_slice(2, 2, &[-1.0, 2.0, 0.0, 1.0]);
        self.initial_population = DVector::from_vec(vec![0.1, 0.9]);
        self.strategy_names = vec!["Hawk".to_string(), "Dove".to_string()];
        self.time_horizon = 20.0;
        self.run_simulation();
    }

    pub(crate) fn load_preset_rps(&mut self) {
        self.payoff_matrix =
            DMatrix::from_row_slice(3, 3, &[0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0]);
        self.initial_population = DVector::from_vec(vec![0.33, 0.33, 0.34]);
        self.strategy_names = vec![
            "Rock".to_string(),
            "Paper".to_string(),
            "Scissors".to_string(),
        ];
        self.time_horizon = 50.0;
        self.run_simulation();
    }
}

impl InteractiveTool for ReplicatorDynamicsTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Replicator Dynamics"
    }
    fn domain(&self) -> &'static str {
        "game_theory"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.export_config()).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let config: ReplicatorPresetConfig = serde_json::from_str(state)
            .map_err(|e| format!("Invalid ReplicatorDynamicsTool state: {e}"))?;
        self.apply_config(config)
    }

    fn show(&mut self, ctx: &egui::Context) {
        if self.is_playing && !self.trajectory.is_empty() {
            let dt = f64::from(ctx.input(|i| i.stable_dt));
            self.playback_time += dt * self.playback_speed;
            if self.playback_time > self.time_horizon {
                self.playback_time = 0.0;
            }
            ctx.request_repaint();
        }

        egui::SidePanel::left("replicator_control_panel").show(ctx, |ui| {
            self.show_settings_panel(ui);
        });

        self.show_bottom_panel(ctx);

        egui::CentralPanel::default().show(ctx, |ui| {
            self.show_trajectory_plot(ui);
        });
    }
}

// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "ReplicatorDynamicsTool",
        display_name: "Replicator Dynamics",
        domain: "game_theory",
        tags: &[],
        build: || Box::new(ReplicatorDynamicsTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for ReplicatorDynamicsTool {
    fn theory_description(&self) -> String {
        "Multi-strategy evolutionary replicator dynamics: $\\dot{x}_i = x_i \\left( (A x)_i - x^T A x \\right)$, where strategy frequency $x_i$ increases if its expected payoff $(A x)_i$ exceeds the population average fitness $x^T A x$.".into()
    }
    fn phonetic_description(&self) -> String {
        "Evolutionary game theory replicator dynamics".into()
    }
    fn theory_citation(&self) -> String {
        "Evolution and the Theory of Games (Maynard Smith, 1982); Evolutionary Games and Population Dynamics (Hofbauer & Sigmund, 1998)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

impl InteractivePreset for ReplicatorDynamicsTool {
    type Config = ReplicatorPresetConfig;

    fn domain(&self) -> &'static str {
        "game_theory"
    }

    fn title(&self) -> String {
        "Replicator Dynamics Simulation".to_string()
    }

    fn export_config(&self) -> Self::Config {
        let rows = self.payoff_matrix.nrows();
        let cols = self.payoff_matrix.ncols();
        let mut matrix = Vec::with_capacity(rows);
        for i in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for j in 0..cols {
                row.push(self.payoff_matrix[(i, j)]);
            }
            matrix.push(row);
        }
        let initial_pop: Vec<f64> = self.initial_population.iter().copied().collect();

        ReplicatorPresetConfig {
            payoff_matrix: matrix,
            initial_population: initial_pop,
            time_horizon: self.time_horizon,
            dt: self.dt,
            strategy_names: self.strategy_names.clone(),
        }
    }

    fn apply_config(&mut self, config: Self::Config) -> Result<(), String> {
        config.validate()?;

        let rows = config.payoff_matrix.len();
        let mut flat = Vec::with_capacity(rows * rows);
        for row in &config.payoff_matrix {
            flat.extend_from_slice(row);
        }

        self.payoff_matrix = DMatrix::from_row_slice(rows, rows, &flat);
        self.initial_population = DVector::from_vec(config.initial_population);
        self.time_horizon = config.time_horizon;
        self.dt = config.dt;
        self.strategy_names = config.strategy_names;
        self.run_simulation();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replicator_step_once_and_reset_playback() {
        let mut tool = ReplicatorDynamicsTool::default();
        assert!(!tool.is_playing);
        assert_eq!(tool.playback_time, 0.0);

        tool.step_once();
        assert!(!tool.is_playing);
        assert!((tool.playback_time - tool.dt).abs() < 1e-9);

        tool.reset_playback();
        assert!(!tool.is_playing);
        assert_eq!(tool.playback_time, 0.0);
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "hawk_dove",
        title: "Hawk-Dove Dynamics",
        description: "ẋ_i = x_i((Ax)_i - xᵀAx): Evolutionary game model of conflict and resource competition.",
        category: "Game Theory",
        icon: "🦅",
        domain: "game_theory",
        tool_name: Some("ReplicatorDynamicsTool"),
        preset_generator: || crate::starter_gallery::PRESET_HAWK_DOVE,
    }
}
