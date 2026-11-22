use crate::framework::InteractiveTool;
use crate::presets::ReplicatorPresetConfig;
use eframe::egui;
use math_explorer::applied::game_theory::evolutionary::ReplicatorDynamics;
use nalgebra::{DMatrix, DVector};

#[path = "replicator/spatial.rs"]
mod spatial;
#[path = "replicator/ui.rs"]
mod ui;

const fn default_true() -> bool {
    true
}

fn is_true(v: &bool) -> bool {
    *v
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplicatorToolState {
    payoff_matrix: Vec<Vec<f64>>,
    initial_population: Vec<f64>,
    time_horizon: f64,
    dt: f64,
    strategy_names: Vec<String>,
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    follow_workspace_graph: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seeded_graph: Option<crate::app::data_bus::SharedGraph>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frozen_graph: Option<crate::app::data_bus::SharedGraph>,
}

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
    spatial: spatial::SpatialReplicatorState,
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
            spatial: spatial::SpatialReplicatorState::default(),
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
        self.recompute_spatial_trajectory();
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
        self.spatial.seeded_graph = None;
        self.payoff_matrix = DMatrix::from_row_slice(2, 2, &[-1.0, 2.0, 0.0, 1.0]);
        self.initial_population = DVector::from_vec(vec![0.1, 0.9]);
        self.strategy_names = vec!["Hawk".to_string(), "Dove".to_string()];
        self.time_horizon = 20.0;
        self.run_simulation();
    }

    pub(crate) fn load_preset_rps(&mut self) {
        self.spatial.seeded_graph = None;
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
        let cfg = self.export_config();
        let frozen_graph = (!self.spatial.follow_workspace_graph)
            .then(|| self.spatial.active_graph.as_ref().map(|g| (**g).clone()))
            .flatten();
        serde_json::to_string(&ReplicatorToolState {
            payoff_matrix: cfg.payoff_matrix,
            initial_population: cfg.initial_population,
            time_horizon: cfg.time_horizon,
            dt: cfg.dt,
            strategy_names: cfg.strategy_names,
            follow_workspace_graph: self.spatial.follow_workspace_graph,
            seeded_graph: self.spatial.seeded_graph.clone(),
            frozen_graph,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: ReplicatorToolState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid ReplicatorDynamicsTool state: {e}"))?;
        for g in [&s.seeded_graph, &s.frozen_graph].into_iter().flatten() {
            g.validate(
                crate::app::data_bus::SESSION_MAX_BUS_NODES,
                crate::app::data_bus::SESSION_MAX_BUS_EDGES,
            )?;
        }
        self.spatial.follow_workspace_graph = s.follow_workspace_graph;
        self.spatial.seeded_graph = s.seeded_graph.clone();
        self.spatial.pending_seed_publish = s.seeded_graph.is_some();
        self.apply_config(ReplicatorPresetConfig {
            payoff_matrix: s.payoff_matrix,
            initial_population: s.initial_population,
            time_horizon: s.time_horizon,
            dt: s.dt,
            strategy_names: s.strategy_names,
        })?;
        if let Some(g) = s.frozen_graph.or(s.seeded_graph) {
            self.apply_shared_game_graph(&std::sync::Arc::new(g));
        } else if self.spatial.follow_workspace_graph {
            self.spatial.active_graph = None;
            self.spatial.applied_bus_revision = 0;
            self.spatial.trajectory.clear();
        }
        Ok(())
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        self.sync_spatial_from_bus(parent_ui.ctx());
        if self.is_playing && !self.trajectory.is_empty() {
            let dt = f64::from(parent_ui.ctx().input(|i| i.stable_dt));
            self.playback_time += dt * self.playback_speed;
            if self.playback_time > self.time_horizon {
                self.playback_time = 0.0;
            }
            parent_ui.ctx().request_repaint();
        }

        let panel_id = crate::framework::scoped_panel_id(parent_ui, "replicator_control_panel");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            self.show_settings_panel(ui);
        });

        self.show_bottom_panel(parent_ui);

        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
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

impl ReplicatorDynamicsTool {
    fn export_config(&self) -> ReplicatorPresetConfig {
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

    fn apply_config(&mut self, config: ReplicatorPresetConfig) -> Result<(), String> {
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
