#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod analytics;
mod canvas;
pub mod challenge;
mod controls;
mod dataflow;
mod presets;
#[cfg(test)]
mod tests;
mod topology_sync;

#[cfg(test)]
pub(crate) use analytics::compute_network_analytics;

use crate::app::dataflow_bus::MAX_DATAFLOW_GRAPH_NODES;
use crate::app::{MAX_SHARED_GRAPH_NODES, TopologySource};
use crate::framework::InteractiveTool;
use eframe::egui;
use math_explorer::epidemiology::networks::{NetworkEpidemicModel, NodeState};
use presets::{NetPropState, build_validated_candidate};
use std::hash::{Hash, Hasher};

/// Default deterministic seed matching [`oxidize_core::rng::OxidizeRng::default`].
pub const DEFAULT_NETWORK_SEED: u64 = 0x0001_D1CE_CAFE_BABE;
/// Default containment intervention budget (five vaccinations or edge quarantines).
pub const DEFAULT_CONTAINMENT_BUDGET: usize = 5;
/// Maximum containment budget permitted in serialized tool configurations.
pub const MAX_CONTAINMENT_BUDGET: usize = 200;
/// Maximum slider range for interactive containment budget selection.
pub const MAX_SLIDER_BUDGET: usize = 25;

/// Active pointer and keyboard target mode on the network propagation canvas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContainmentMode {
    /// Toggle immunization (`NodeState::Vaccinated`) on susceptible vertices.
    VaccinateNode,
    /// Toggle symmetric quarantine across undirected contact edges.
    QuarantineEdge,
}

/// Interactive stochastic network contagion and budgeted containment explorer.
pub struct NetworkPropagationTool {
    pub(super) model: NetworkEpidemicModel,
    pub(super) local_num_nodes: usize,
    pub topology_source: TopologySource,
    pub(super) synced_bus_revision: Option<u64>,
    pub(super) rng: oxidize_core::rng::OxidizeRng,
    pub(super) baseline_model: NetworkEpidemicModel,
    pub(super) baseline_rng: oxidize_core::rng::OxidizeRng,
    pub(super) seed: u64,
    pub(super) budget: usize,
    pub(super) vaccinated_nodes: Vec<usize>,
    pub(super) quarantined_edges: Vec<usize>,
    pub(super) interaction_mode: ContainmentMode,
    pub(super) focused_node: usize,
    pub(super) focused_edge: usize,
    #[cfg(test)]
    pub(super) canvas_id: Option<egui::Id>,
    pub(super) is_running: bool,
    pub(super) active_challenge: Option<&'static str>,
    pub(super) revision: u64,
    pub(super) last_upstream_graph_rev: u64,
    pub(super) upstream_linked: bool,
    #[cfg(test)]
    pub(super) sim_steps: u64,
}

impl Default for NetworkPropagationTool {
    fn default() -> Self {
        let mut rng = oxidize_core::rng::OxidizeRng::new(DEFAULT_NETWORK_SEED);
        let model = NetworkEpidemicModel::new_with_rng(50, 0.05, 0.02, &mut rng);
        let mut baseline_rng = oxidize_core::rng::OxidizeRng::new(DEFAULT_NETWORK_SEED);
        let baseline_model = NetworkEpidemicModel::new_with_rng(50, 0.05, 0.02, &mut baseline_rng);
        Self {
            model,
            local_num_nodes: 50,
            topology_source: TopologySource::SharedBus,
            synced_bus_revision: None,
            rng,
            baseline_model,
            baseline_rng,
            seed: DEFAULT_NETWORK_SEED,
            budget: DEFAULT_CONTAINMENT_BUDGET,
            vaccinated_nodes: Vec::new(),
            quarantined_edges: Vec::new(),
            interaction_mode: ContainmentMode::VaccinateNode,
            focused_node: 0,
            focused_edge: 0,
            #[cfg(test)]
            canvas_id: None,
            is_running: false,
            active_challenge: None,
            revision: 1,
            last_upstream_graph_rev: 0,
            upstream_linked: false,
            #[cfg(test)]
            sim_steps: 0,
        }
    }
}

impl NetworkPropagationTool {
    pub(super) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    pub(crate) fn used_budget(&self) -> usize {
        self.vaccinated_nodes.len() + self.quarantined_edges.len()
    }

    pub(crate) fn remaining_budget(&self) -> usize {
        self.budget.saturating_sub(self.used_budget())
    }

    pub(super) fn clamp_focus_indices(&mut self) {
        let n = self.model.num_nodes();
        self.focused_node = if n == 0 {
            0
        } else {
            self.focused_node.min(n - 1)
        };
        let e = self.model.edges().len();
        self.focused_edge = if e == 0 {
            0
        } else {
            self.focused_edge.min(e - 1)
        };
    }

    pub(super) fn rebuild_seeded_graph(&mut self) {
        self.synced_bus_revision = None;
        self.rng = oxidize_core::rng::OxidizeRng::new(self.seed);
        self.model
            .initialize_geometric_graph_with_rng(&mut self.rng);
        self.vaccinated_nodes
            .retain(|&node| self.model.set_node_vaccinated(node, true).is_ok());
        let edges = self.model.edges().to_vec();
        self.quarantined_edges.retain(|&edge_idx| {
            edges
                .get(edge_idx)
                .is_some_and(|&[u, v]| self.model.set_edge_quarantined(u, v, true).is_ok())
        });
        self.is_running = false;
        self.rebuild_baseline_twin();
        self.clamp_focus_indices();
        #[cfg(test)]
        {
            self.sim_steps = 0;
        }
        self.bump_revision();
    }

    pub(super) fn reset_network(&mut self) {
        self.rebuild_seeded_graph();
    }

    pub(crate) fn clear_interventions(&mut self) {
        self.vaccinated_nodes.clear();
        self.quarantined_edges.clear();
        self.rebuild_seeded_graph();
    }

    pub(super) fn set_seed_and_reset(&mut self, next_seed: u64) {
        self.seed = next_seed;
        self.vaccinated_nodes.clear();
        self.quarantined_edges.clear();
        self.rebuild_seeded_graph();
    }

    pub(super) fn enforce_budget_cap(&mut self) {
        while self.used_budget() > self.budget {
            if let Some(edge_idx) = self.quarantined_edges.pop() {
                if let Some(&[u, v]) = self.model.edges().get(edge_idx) {
                    let _ = self.model.set_edge_quarantined(u, v, false);
                }
            } else if let Some(node) = self.vaccinated_nodes.pop() {
                let _ = self.model.set_node_vaccinated(node, false);
            } else {
                break;
            }
        }
    }

    pub(crate) fn toggle_node_vaccination(&mut self, node: usize) -> Result<bool, String> {
        if !self.allows_vaccination() {
            return Err("Node vaccination is locked during edge quarantine challenge".into());
        }
        if node >= self.model.num_nodes() {
            return Err(format!("Node {node} is out of range"));
        }
        if let Ok(pos) = self.vaccinated_nodes.binary_search(&node) {
            self.model
                .set_node_vaccinated(node, false)
                .map_err(|e| e.to_string())?;
            self.vaccinated_nodes.remove(pos);
            self.bump_revision();
            return Ok(false);
        }
        if self.model.states()[node] != NodeState::Susceptible {
            return Err(format!(
                "Cannot vaccinate Node {node} (already infected or recovered)"
            ));
        }
        if self.used_budget() >= self.budget {
            return Err(format!(
                "Containment budget exhausted ({}/{} used)",
                self.budget, self.budget
            ));
        }
        self.model
            .set_node_vaccinated(node, true)
            .map_err(|e| e.to_string())?;
        let insert_at = self.vaccinated_nodes.partition_point(|&v| v < node);
        self.vaccinated_nodes.insert(insert_at, node);
        self.bump_revision();
        Ok(true)
    }

    pub(crate) fn toggle_edge_quarantine(&mut self, edge_idx: usize) -> Result<bool, String> {
        if !self.allows_quarantine() {
            return Err("Edge quarantine is locked during hub vaccination challenge".into());
        }
        let &[u, v] = self
            .model
            .edges()
            .get(edge_idx)
            .ok_or_else(|| format!("Edge {edge_idx} is out of range"))?;
        if let Ok(pos) = self.quarantined_edges.binary_search(&edge_idx) {
            self.model
                .set_edge_quarantined(u, v, false)
                .map_err(|e| e.to_string())?;
            self.quarantined_edges.remove(pos);
            self.bump_revision();
            return Ok(false);
        }
        if self.used_budget() >= self.budget {
            return Err(format!(
                "Containment budget exhausted ({}/{} used)",
                self.budget, self.budget
            ));
        }
        self.model
            .set_edge_quarantined(u, v, true)
            .map_err(|e| e.to_string())?;
        let insert_at = self.quarantined_edges.partition_point(|&e| e < edge_idx);
        self.quarantined_edges.insert(insert_at, edge_idx);
        self.bump_revision();
        Ok(true)
    }

    pub(crate) fn step_once(&mut self) {
        if self.is_running {
            self.is_running = false;
        }
        self.step_models_lockstep();
        self.bump_revision();
        #[cfg(test)]
        {
            self.sim_steps = self.sim_steps.saturating_add(1);
        }
    }
}

impl InteractiveTool for NetworkPropagationTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Network Propagation"
    }
    fn domain(&self) -> &'static str {
        "epidemiology"
    }
    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::none()
    }
    fn observables(&self) -> Vec<(&'static str, f64)> {
        self.topology_observables()
    }
    fn has_finite_parameters(&self) -> bool {
        let on_external = self.last_upstream_graph_rev != 0 || self.synced_bus_revision.is_some();
        let min_nodes = if on_external { 1 } else { 10 };
        let max_nodes = MAX_SHARED_GRAPH_NODES.max(MAX_DATAFLOW_GRAPH_NODES);
        self.model.beta().is_finite()
            && self.model.gamma().is_finite()
            && (min_nodes..=max_nodes).contains(&self.model.num_nodes())
            && (10..=200).contains(&self.local_num_nodes)
    }
    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.revision.hash(&mut h);
        self.local_num_nodes.hash(&mut h);
        self.topology_source.hash(&mut h);
        self.model.beta().to_bits().hash(&mut h);
        self.model.gamma().to_bits().hash(&mut h);
        self.seed.hash(&mut h);
        self.budget.hash(&mut h);
        self.vaccinated_nodes.hash(&mut h);
        self.quarantined_edges.hash(&mut h);
        self.is_running.hash(&mut h);
        self.active_challenge.hash(&mut h);
        h.finish().max(1)
    }
    fn save_state(&self) -> Option<String> {
        let (num_nodes, vaccinated_nodes, quarantined_edges) = if self.last_upstream_graph_rev != 0
        {
            (
                self.model.num_nodes().clamp(10, 200),
                Vec::new(),
                Vec::new(),
            )
        } else if self.synced_bus_revision.is_some() {
            (self.local_num_nodes, Vec::new(), Vec::new())
        } else {
            (
                self.model.num_nodes(),
                self.vaccinated_nodes.clone(),
                self.quarantined_edges.clone(),
            )
        };
        serde_json::to_string(&NetPropState {
            num_nodes,
            beta: self.model.beta(),
            gamma: self.model.gamma(),
            seed: self.seed,
            budget: self.budget,
            vaccinated_nodes,
            quarantined_edges,
            is_running: self.is_running,
            topology_source: self.topology_source,
        })
        .ok()
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: NetPropState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid network propagation state: {e}"))?;
        let (candidate, candidate_rng, vax_sorted, edge_sorted) = build_validated_candidate(&s)?;
        self.model = candidate;
        self.local_num_nodes = s.num_nodes;
        self.topology_source = s.topology_source;
        self.synced_bus_revision = None;
        self.last_upstream_graph_rev = 0;
        self.upstream_linked = false;
        self.rng = candidate_rng;
        self.seed = s.seed;
        self.budget = s.budget;
        self.vaccinated_nodes = vax_sorted;
        self.quarantined_edges = edge_sorted;
        self.is_running = s.is_running;
        self.rebuild_baseline_twin();
        self.enforce_challenge_locks();
        self.clamp_focus_indices();
        #[cfg(test)]
        {
            self.sim_steps = 0;
        }
        self.bump_revision();
        Ok(())
    }

    fn apply_challenge(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        self.set_epi_challenge(challenge_id)
    }

    fn active_challenge_id(&self) -> Option<&'static str> {
        self.active_challenge
    }

    fn evaluate_challenge(&self) -> Option<crate::framework::ChallengeEvaluation> {
        self.eval_epi_challenge()
    }

    fn step_simulation(&mut self) {
        self.step_once();
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.poll_upstream_graph(&ctx);
        if !self.upstream_linked {
            self.sync_shared_topology(&ctx);
        }
        egui::CentralPanel::default().show_inside(ui, |ui| {
            if self.is_running {
                self.step_models_lockstep();
                #[cfg(test)]
                {
                    self.sim_steps = self.sim_steps.saturating_add(1);
                }
                ui.ctx().request_repaint();
            }

            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.set_max_width(290.0);
                    self.draw_controls_column(ui);
                });
                ui.separator();
                ui.vertical(|ui| {
                    self.draw_graph_column(ui);
                });
            });
        });
    }
}

// [cite:algorithmic_information_rust]
// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "NetworkPropagationTool",
        display_name: "Network Propagation",
        domain: "epidemiology",
        tags: &["containment", "r0", "vaccination", "quarantine", "contact network", "sir"],
        build: || Box::new(NetworkPropagationTool::default()),
    }
}
