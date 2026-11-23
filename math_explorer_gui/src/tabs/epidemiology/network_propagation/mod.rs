#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod analytics;
mod bus_sync;
mod canvas;
mod controls;
mod presets;
#[cfg(test)]
mod tests;

pub(crate) use analytics::compute_network_analytics;

use crate::app::data_bus::{SharedGraph, MAX_EPIDEMIC_NODE_BUDGET};
use crate::framework::InteractiveTool;
use eframe::egui;
use math_explorer::epidemiology::networks::{NetworkEpidemicModel, NodeState};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

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

const fn default_seed() -> u64 {
    DEFAULT_NETWORK_SEED
}

const fn is_default_seed(seed: &u64) -> bool {
    *seed == DEFAULT_NETWORK_SEED
}

const fn default_budget() -> usize {
    DEFAULT_CONTAINMENT_BUDGET
}

const fn is_default_budget(budget: &usize) -> bool {
    *budget == DEFAULT_CONTAINMENT_BUDGET
}

const fn default_true() -> bool {
    true
}

const fn is_false(value: &bool) -> bool {
    !*value
}

const fn is_true(value: &bool) -> bool {
    *value
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct NetPropState {
    pub(super) num_nodes: usize,
    pub(super) beta: f64,
    pub(super) gamma: f64,
    #[serde(default = "default_seed", skip_serializing_if = "is_default_seed")]
    pub(super) seed: u64,
    #[serde(default = "default_budget", skip_serializing_if = "is_default_budget")]
    pub(super) budget: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) vaccinated_nodes: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) quarantined_edges: Vec<usize>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub(super) is_running: bool,
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub(super) follow_workspace_graph: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) seeded_graph: Option<SharedGraph>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) frozen_graph: Option<SharedGraph>,
}

/// Interactive stochastic network contagion, containment, and workspace graph explorer.
pub struct NetworkPropagationTool {
    pub(super) model: NetworkEpidemicModel,
    pub(super) rng: oxidize_core::rng::OxidizeRng,
    pub(super) seed: u64,
    pub(super) budget: usize,
    pub(super) vaccinated_nodes: Vec<usize>,
    pub(super) quarantined_edges: Vec<usize>,
    pub(super) interaction_mode: ContainmentMode,
    pub(super) focused_node: usize,
    pub(super) focused_edge: usize,
    pub(super) canvas_id: Option<egui::Id>,
    pub(super) is_running: bool,
    pub(super) revision: u64,
    pub(crate) follow_workspace_graph: bool,
    pub(crate) applied_bus_revision: u64,
    pub(super) clamped_source_nodes: Option<usize>,
    pub(super) active_shared_graph: Option<Arc<SharedGraph>>,
    pub(super) seeded_graph: Option<SharedGraph>,
    pub(super) pending_seed_publish: bool,
    pub(super) trace_step: usize,
    pub(super) trace_s: Vec<[f64; 2]>,
    pub(super) trace_i: Vec<[f64; 2]>,
    pub(super) trace_r: Vec<[f64; 2]>,
    pub(super) trace_dirty: bool,
    #[cfg(test)]
    pub(super) sim_steps: u64,
}

impl Default for NetworkPropagationTool {
    fn default() -> Self {
        let mut rng = oxidize_core::rng::OxidizeRng::new(DEFAULT_NETWORK_SEED);
        let model = NetworkEpidemicModel::new_with_rng(50, 0.05, 0.02, &mut rng);
        let mut tool = Self {
            model,
            rng,
            seed: DEFAULT_NETWORK_SEED,
            budget: DEFAULT_CONTAINMENT_BUDGET,
            vaccinated_nodes: Vec::new(),
            quarantined_edges: Vec::new(),
            interaction_mode: ContainmentMode::VaccinateNode,
            focused_node: 0,
            focused_edge: 0,
            canvas_id: None,
            is_running: false,
            revision: 1,
            follow_workspace_graph: true,
            applied_bus_revision: 0,
            clamped_source_nodes: None,
            active_shared_graph: None,
            seeded_graph: None,
            pending_seed_publish: false,
            trace_step: 0,
            trace_s: Vec::new(),
            trace_i: Vec::new(),
            trace_r: Vec::new(),
            trace_dirty: true,
            #[cfg(test)]
            sim_steps: 0,
        };
        tool.reset_empirical_trace();
        tool
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

    pub(super) fn retain_valid_interventions(&mut self) {
        self.vaccinated_nodes
            .retain(|&node| self.model.set_node_vaccinated(node, true).is_ok());
        let edges = self.model.edges().to_vec();
        self.quarantined_edges.retain(|&edge_idx| {
            edges
                .get(edge_idx)
                .is_some_and(|&[u, v]| self.model.set_edge_quarantined(u, v, true).is_ok())
        });
    }

    pub(super) fn rebuild_seeded_graph(&mut self) {
        self.rng = oxidize_core::rng::OxidizeRng::new(self.seed);
        self.model
            .initialize_geometric_graph_with_rng(&mut self.rng);
        self.retain_valid_interventions();
        self.is_running = false;
        self.clamp_focus_indices();
        self.reset_empirical_trace();
        #[cfg(test)]
        {
            self.sim_steps = 0;
        }
        self.bump_revision();
    }

    pub(super) fn reset_network(&mut self) {
        if let Some(shared) = self.active_shared_graph.clone() {
            let saved_vax = self.vaccinated_nodes.clone();
            let saved_quar = self.quarantined_edges.clone();
            self.apply_shared_graph(&shared);
            self.vaccinated_nodes = saved_vax;
            self.quarantined_edges = saved_quar;
            self.retain_valid_interventions();
            self.reset_empirical_trace();
            self.is_running = false;
            #[cfg(test)]
            {
                self.sim_steps = 0;
            }
            self.bump_revision();
        } else {
            self.rebuild_seeded_graph();
        }
    }

    pub(crate) fn clear_interventions(&mut self) {
        self.vaccinated_nodes.clear();
        self.quarantined_edges.clear();
        self.reset_network();
    }

    pub(super) fn set_seed_and_reset(&mut self, next_seed: u64) {
        self.seed = next_seed;
        self.active_shared_graph = None;
        self.seeded_graph = None;
        self.clamped_source_nodes = None;
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
        self.trace_dirty = true;
    }

    pub(crate) fn toggle_node_vaccination(&mut self, node: usize) -> Result<bool, String> {
        if node >= self.model.num_nodes() {
            return Err(format!("Node {node} is out of range"));
        }
        if let Ok(pos) = self.vaccinated_nodes.binary_search(&node) {
            self.model
                .set_node_vaccinated(node, false)
                .map_err(|e| e.to_string())?;
            self.vaccinated_nodes.remove(pos);
            self.trace_dirty = true;
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
        self.trace_dirty = true;
        self.bump_revision();
        Ok(true)
    }

    pub(crate) fn toggle_edge_quarantine(&mut self, edge_idx: usize) -> Result<bool, String> {
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
            self.trace_dirty = true;
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
        self.trace_dirty = true;
        self.bump_revision();
        Ok(true)
    }

    pub(crate) fn step_once(&mut self) {
        if self.is_running {
            self.is_running = false;
            self.bump_revision();
        }
        self.model.step_with_rng(&mut self.rng);
        self.trace_step = self.trace_step.saturating_add(1);
        self.record_trace_sample();
        #[cfg(test)]
        {
            self.sim_steps = self.sim_steps.saturating_add(1);
        }
    }

    #[cfg(test)]
    pub(crate) fn sim_steps(&self) -> u64 {
        self.sim_steps
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
    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.canvas_id
    }
    fn has_finite_parameters(&self) -> bool {
        let nodes_ok = if self.active_shared_graph.is_some() || self.seeded_graph.is_some() {
            (1..=MAX_EPIDEMIC_NODE_BUDGET).contains(&self.model.num_nodes())
        } else {
            (10..=200).contains(&self.model.num_nodes())
        };
        self.model.beta().is_finite() && self.model.gamma().is_finite() && nodes_ok
    }
    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.revision.hash(&mut h);
        self.model.num_nodes().hash(&mut h);
        self.model.beta().to_bits().hash(&mut h);
        self.model.gamma().to_bits().hash(&mut h);
        self.seed.hash(&mut h);
        self.budget.hash(&mut h);
        self.vaccinated_nodes.hash(&mut h);
        self.quarantined_edges.hash(&mut h);
        self.is_running.hash(&mut h);
        self.follow_workspace_graph.hash(&mut h);
        self.applied_bus_revision.hash(&mut h);
        h.finish().max(1)
    }
    fn save_state(&self) -> Option<String> {
        let frozen_graph = (!self.follow_workspace_graph)
            .then(|| self.active_shared_graph.as_deref().cloned())
            .flatten();
        serde_json::to_string(&NetPropState {
            num_nodes: self.model.num_nodes(),
            beta: self.model.beta(),
            gamma: self.model.gamma(),
            seed: self.seed,
            budget: self.budget,
            vaccinated_nodes: self.vaccinated_nodes.clone(),
            quarantined_edges: self.quarantined_edges.clone(),
            is_running: self.is_running,
            follow_workspace_graph: self.follow_workspace_graph,
            seeded_graph: self.seeded_graph.clone(),
            frozen_graph,
        })
        .ok()
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.load_validated_state_json(state)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.sync_from_bus(&ctx);

        egui::CentralPanel::default().show_inside(ui, |ui| {
            if self.is_running {
                self.model.step_with_rng(&mut self.rng);
                self.trace_step = self.trace_step.saturating_add(1);
                self.record_trace_sample();
                #[cfg(test)]
                {
                    self.sim_steps = self.sim_steps.saturating_add(1);
                }
                ctx.request_repaint();
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

        self.flush_trace_to_bus(&ctx);
    }
}

// [cite:algorithmic_information_rust]
// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "NetworkPropagationTool",
        display_name: "Network Propagation",
        domain: "epidemiology",
        tags: &["containment", "r0", "vaccination", "quarantine"],
        build: || Box::new(NetworkPropagationTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for NetworkPropagationTool {
    fn theory_description(&self) -> String {
        let summary = compute_network_analytics(&self.model);
        format!(
            "Stochastic contact network epidemic propagation and containment: per-contact transmission $\\beta = {:.2}$, recovery $\\gamma = {:.2}$, residual heterogeneous reproduction number $R_0 = \\frac{{\\beta}}{{\\gamma}}\\frac{{\\langle k^2 \\rangle - \\langle k \\rangle}}{{\\langle k \\rangle}} = {:.2}$, and branching extinction probability $P_{{ext}} = {:.2}$ under vertex vaccination and symmetric edge quarantine.",
            self.model.beta(),
            self.model.gamma(),
            summary.effective_r0,
            summary.extinction_probability,
        )
    }
    fn phonetic_description(&self) -> String {
        "Network epidemic propagation, hub vaccination, and edge quarantine theory".into()
    }
    fn theory_citation(&self) -> String {
        "Epidemic processes in complex networks (Pastor-Satorras et al., 2015); Networks (Newman, 2010); Vulnerability in graphs (Barefoot et al., 1987)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}
