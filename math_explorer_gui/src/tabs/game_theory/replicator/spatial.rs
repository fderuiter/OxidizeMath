//! Spatial graph coupling (`GraphReplicatorSystem`) and starter card for [`super::ReplicatorDynamicsTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::ReplicatorDynamicsTool;
use crate::app::data_bus::ui::{
    render_follow_workspace_toggle, render_node_budget_banner, sync_consumer_follow_flag,
};
use crate::app::data_bus::{
    read_shared_graph, seed_shared_graph_if_empty, SharedGraph, MAX_GAME_NODE_BUDGET,
};
use eframe::egui;
use math_explorer::applied::game_theory::evolutionary::GraphReplicatorSystem;
use std::sync::Arc;

pub(super) struct SpatialReplicatorState {
    pub(super) follow_workspace_graph: bool,
    pub(super) applied_bus_revision: u64,
    pub(super) clamped_source_nodes: Option<usize>,
    pub(super) active_graph: Option<Arc<SharedGraph>>,
    pub(super) seeded_graph: Option<SharedGraph>,
    pub(super) pending_seed_publish: bool,
    pub(super) trajectory: Vec<(f64, Vec<f64>)>,
}

impl Default for SpatialReplicatorState {
    fn default() -> Self {
        Self {
            follow_workspace_graph: true,
            applied_bus_revision: 0,
            clamped_source_nodes: None,
            active_graph: None,
            seeded_graph: None,
            pending_seed_publish: false,
            trajectory: Vec::new(),
        }
    }
}

impl ReplicatorDynamicsTool {
    #[cfg(all(
        test,
        feature = "biology",
        feature = "applied",
        feature = "epidemiology",
        feature = "physics",
        feature = "pure_math"
    ))]
    pub(crate) fn spatial_trajectory(&self) -> &[(f64, Vec<f64>)] {
        &self.spatial.trajectory
    }

    #[cfg(all(
        test,
        feature = "biology",
        feature = "applied",
        feature = "epidemiology",
        feature = "physics",
        feature = "pure_math"
    ))]
    pub(crate) fn follow_workspace_graph(&self) -> bool {
        self.spatial.follow_workspace_graph
    }

    pub(crate) fn recompute_spatial_trajectory(&mut self) {
        let Some(graph) = &self.spatial.active_graph else {
            self.spatial.trajectory.clear();
            return;
        };
        if graph.nodes.is_empty() || self.payoff_matrix.nrows() == 0 {
            self.spatial.trajectory.clear();
            return;
        }
        let n_strat = self.payoff_matrix.nrows();
        let init: Vec<f64> = self.initial_population.iter().copied().collect();
        let Ok(mut sys) = GraphReplicatorSystem::from_population_mean(
            graph.nodes.len(),
            &self.payoff_matrix,
            &graph.edges,
            &init,
        ) else {
            self.spatial.trajectory.clear();
            return;
        };

        let steps = ((self.time_horizon / self.dt).round() as usize).clamp(1, 4_000);
        let mut traj = Vec::with_capacity(steps + 1);
        let mut buf = vec![0.0; n_strat];
        sys.mean_frequencies(&mut buf);
        traj.push((0.0, buf.clone()));
        for step_idx in 1..=steps {
            sys.step(self.dt);
            sys.mean_frequencies(&mut buf);
            traj.push((step_idx as f64 * self.dt, buf.clone()));
        }
        self.spatial.trajectory = traj;
    }

    pub(crate) fn apply_shared_game_graph(&mut self, shared: &Arc<SharedGraph>) {
        let (clamped, orig_count) = shared.clamped_to_budget(MAX_GAME_NODE_BUDGET);
        self.spatial.clamped_source_nodes = orig_count;
        self.spatial.applied_bus_revision = shared.revision;
        if clamped.nodes.is_empty() {
            self.spatial.active_graph = None;
            self.spatial.trajectory.clear();
            return;
        }
        self.spatial.active_graph = Some(clamped);
        self.recompute_spatial_trajectory();
    }

    pub(super) fn sync_spatial_from_bus(&mut self, ctx: &egui::Context) {
        if sync_consumer_follow_flag(
            ctx,
            &mut self.spatial.follow_workspace_graph,
            "Replicator Dynamics",
        ) && self.spatial.follow_workspace_graph
        {
            self.spatial.applied_bus_revision = 0;
        }
        if self.spatial.pending_seed_publish {
            self.spatial.pending_seed_publish = false;
            if let Some(ref seeded) = self.spatial.seeded_graph {
                let arc = seed_shared_graph_if_empty(ctx, seeded);
                if self.spatial.follow_workspace_graph {
                    self.apply_shared_game_graph(&arc);
                }
            }
        }
        if !self.spatial.follow_workspace_graph {
            return;
        }
        if let Some(shared) = read_shared_graph(ctx) {
            if shared.revision != self.spatial.applied_bus_revision {
                self.apply_shared_game_graph(&shared);
            }
        }
    }

    pub(super) fn render_spatial_controls(&mut self, ui: &mut egui::Ui) {
        let rev_opt = (self.spatial.applied_bus_revision > 0)
            .then_some(self.spatial.applied_bus_revision);
        if render_follow_workspace_toggle(
            ui,
            &mut self.spatial.follow_workspace_graph,
            "Replicator Dynamics",
            rev_opt,
        ) && self.spatial.follow_workspace_graph
        {
            self.spatial.applied_bus_revision = 0;
            self.sync_spatial_from_bus(ui.ctx());
        }
        render_node_budget_banner(
            ui,
            self.spatial.clamped_source_nodes,
            MAX_GAME_NODE_BUDGET,
            "spatial game nodes",
        );
    }
}

const SPATIAL_COOPERATORS_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Spatial Cooperators on Clustered Graph",
    "domain": "game_theory",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "payoff_matrix": [
      [1.0, 0.0],
      [1.35, 0.1]
    ],
    "initial_population": [0.55, 0.45],
    "time_horizon": 25.0,
    "dt": 0.05,
    "strategy_names": ["Cooperator", "Defector"],
    "seeded_graph": {
      "revision": 1,
      "nodes": [
        { "id": 0, "norm_pos": [0.20, 0.30] },
        { "id": 1, "norm_pos": [0.32, 0.18] },
        { "id": 2, "norm_pos": [0.32, 0.44] },
        { "id": 3, "norm_pos": [0.20, 0.58] },
        { "id": 4, "norm_pos": [0.68, 0.35] },
        { "id": 5, "norm_pos": [0.80, 0.24] },
        { "id": 6, "norm_pos": [0.80, 0.50] }
      ],
      "edges": [
        [0, 1, 2.5],
        [1, 2, 2.5],
        [2, 0, 2.5],
        [0, 3, 2.0],
        [2, 3, 2.0],
        [2, 4, 0.4],
        [4, 5, 1.0],
        [5, 6, 1.0],
        [6, 4, 1.0]
      ]
    }
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "spatial_cooperators",
        title: "Spatial Cooperators on Clustered Graph",
        description: "Prisoner's Dilemma where clustered +w graph neighborhoods sustain cooperators while well-mixed replicators defect.",
        category: "Game Theory",
        icon: "🤝",
        domain: "game_theory",
        tool_name: Some("ReplicatorDynamicsTool"),
        preset_generator: || SPATIAL_COOPERATORS_PRESET_JSON,
    }
}
