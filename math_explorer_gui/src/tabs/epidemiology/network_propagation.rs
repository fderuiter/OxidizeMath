#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::app::workspace_bus::{GraphTopologyChannel, SharedWorkspaceBus};
use crate::framework::InteractiveTool;
use eframe::egui;
use math_explorer::epidemiology::networks::{heterogeneous_r0, NetworkEpidemicModel, NodeState};

mod view;

/// Interactive node-click action mode on the contact network canvas.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NodeClickAction {
    /// Cycles clicked nodes `Susceptible -> Infected -> Recovered -> Susceptible`.
    #[default]
    CycleState,
    /// Seeds (`Susceptible -> Infected`) or clears (`Infected -> Susceptible`) index cases.
    SeedInfected,
    /// Quarantines (`-> Recovered`) or restores (`Recovered -> Susceptible`) hub nodes.
    QuarantineRecovered,
}

pub struct NetworkPropagationTool {
    model: NetworkEpidemicModel,
    rng: oxidize_core::rng::OxidizeRng,
    is_running: bool,
    standalone_num_nodes: usize,
    click_action: NodeClickAction,
    synced_bus_revision: Option<u64>,
    using_bus_topology: bool,
    revision_counter: u64,
    #[cfg(test)]
    sim_steps: u64,
}

impl Default for NetworkPropagationTool {
    fn default() -> Self {
        Self {
            model: NetworkEpidemicModel::new(50, 0.05, 0.02),
            rng: oxidize_core::rng::OxidizeRng::default(),
            is_running: false,
            standalone_num_nodes: 50,
            click_action: NodeClickAction::CycleState,
            synced_bus_revision: None,
            using_bus_topology: false,
            revision_counter: 1,
            #[cfg(test)]
            sim_steps: 0,
        }
    }
}

impl NetworkPropagationTool {
    fn bump_state_rev(&mut self) {
        self.revision_counter = self.revision_counter.saturating_add(1).max(1);
    }

    fn reset_network(&mut self) {
        self.rng = oxidize_core::rng::OxidizeRng::default();
        self.is_running = false;
        if self.using_bus_topology {
            let n = self.model.num_nodes();
            let states = initial_seeded_states(n);
            *self.model.states_mut() = states;
        } else {
            let _ = self.model.set_num_nodes(self.standalone_num_nodes);
            self.model.initialize_geometric_graph_with_rng(&mut self.rng);
        }
        #[cfg(test)]
        {
            self.sim_steps = 0;
        }
        self.bump_state_rev();
    }

    pub(crate) fn step_once(&mut self) {
        self.is_running = false;
        self.model.step_with_rng(&mut self.rng);
        #[cfg(test)]
        {
            self.sim_steps = self.sim_steps.saturating_add(1);
        }
        self.bump_state_rev();
    }

    #[cfg(test)]
    pub(crate) fn sim_steps(&self) -> u64 {
        self.sim_steps
    }

    /// Computes live heterogeneous network basic reproduction number $R_0$ for the active graph.
    #[must_use]
    pub fn current_heterogeneous_r0(&self) -> f64 {
        let n = self.model.num_nodes();
        if n == 0 {
            return 0.0;
        }
        let adj = self.model.adjacency();
        let mut sum_k = 0.0;
        let mut sum_k_sq = 0.0;
        for neighbors in adj.iter().take(n) {
            let k = neighbors.len() as f64;
            sum_k += k;
            sum_k_sq += k * k;
        }
        let mean_k = sum_k / (n as f64);
        let var_k = (sum_k_sq / (n as f64) - mean_k * mean_k).max(0.0);
        heterogeneous_r0(self.model.beta(), self.model.gamma(), mean_k, var_k)
    }

    /// Active node states slice.
    #[cfg(test)]
    #[must_use]
    pub fn node_states(&self) -> &[NodeState] {
        self.model.states()
    }

    /// Applies the active click-to-seed / click-to-quarantine transition to `node_idx`.
    pub fn apply_node_click(&mut self, node_idx: usize) -> Option<NodeState> {
        let current = *self.model.states().get(node_idx)?;
        let next = match self.click_action {
            NodeClickAction::CycleState => match current {
                NodeState::Susceptible => NodeState::Infected,
                NodeState::Infected => NodeState::Recovered,
                NodeState::Recovered => NodeState::Susceptible,
            },
            NodeClickAction::SeedInfected => {
                if current == NodeState::Infected {
                    NodeState::Susceptible
                } else {
                    NodeState::Infected
                }
            }
            NodeClickAction::QuarantineRecovered => {
                if current == NodeState::Recovered {
                    NodeState::Susceptible
                } else {
                    NodeState::Recovered
                }
            }
        };
        self.model.states_mut()[node_idx] = next;
        self.bump_state_rev();
        Some(next)
    }

    /// Sets the interactive node-click action mode.
    #[cfg(test)]
    pub fn set_click_action(&mut self, mode: NodeClickAction) {
        if self.click_action != mode {
            self.click_action = mode;
            self.bump_state_rev();
        }
    }

    pub(crate) fn ingest_topology_channel(&mut self, channel: &GraphTopologyChannel) {
        let rev = channel.revision();
        if rev == 0 || self.synced_bus_revision == Some(rev) {
            return;
        }
        self.synced_bus_revision = Some(rev);
        if channel.positions().is_empty() {
            return;
        }
        let n = channel.positions().len();
        let positions: Vec<[f32; 2]> = channel
            .positions()
            .iter()
            .map(|&[nx, ny]| [nx * 165.0, ny * 165.0])
            .collect();
        let mut adjacency = vec![Vec::new(); n];
        for &(u, v, _) in channel.edges() {
            if u < n && v < n && u != v {
                adjacency[u].push(v);
                adjacency[v].push(u);
            }
        }
        let states = if self.using_bus_topology {
            (0..n)
                .map(|i| {
                    self.model
                        .states()
                        .get(i)
                        .copied()
                        .unwrap_or(NodeState::Susceptible)
                })
                .collect()
        } else {
            initial_seeded_states(n)
        };
        if let Ok(built) = NetworkEpidemicModel::builder()
            .num_nodes(n)
            .parameters(self.model.beta(), self.model.gamma())
            .states(states)
            .positions(positions)
            .adjacency(adjacency)
            .build()
        {
            self.model = built;
            self.using_bus_topology = true;
            self.bump_state_rev();
        }
    }

    fn on_bus_unlinked(&mut self) {
        if self.using_bus_topology {
            self.using_bus_topology = false;
            self.synced_bus_revision = None;
            self.reset_network();
        }
    }
}

fn initial_seeded_states(num_nodes: usize) -> Vec<NodeState> {
    let mut states = vec![NodeState::Susceptible; num_nodes];
    if !states.is_empty() {
        states[0] = NodeState::Infected;
    }
    states
}

const fn is_false(value: &bool) -> bool {
    !*value
}

fn is_default_click_action(action: &NodeClickAction) -> bool {
    *action == NodeClickAction::CycleState
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NetPropState {
    num_nodes: usize,
    beta: f64,
    gamma: f64,
    #[serde(default, skip_serializing_if = "is_false")]
    is_running: bool,
    #[serde(default, skip_serializing_if = "is_default_click_action")]
    click_action: NodeClickAction,
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

    fn sync_workspace_bus(&mut self, bus: &mut SharedWorkspaceBus) {
        self.ingest_topology_channel(&bus.topology);
    }

    fn state_revision(&self) -> u64 {
        self.revision_counter
    }

    fn has_finite_parameters(&self) -> bool {
        self.model.beta().is_finite() && self.model.gamma().is_finite()
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&NetPropState {
            num_nodes: self.standalone_num_nodes,
            beta: self.model.beta(),
            gamma: self.model.gamma(),
            is_running: self.is_running,
            click_action: self.click_action,
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
        if !(10..=200).contains(&s.num_nodes)
            || !s.beta.is_finite()
            || !(0.0..=1.0).contains(&s.beta)
            || !s.gamma.is_finite()
            || !(0.0..=1.0).contains(&s.gamma)
        {
            return Err("Network propagation parameters out of bounds".into());
        }
        self.standalone_num_nodes = s.num_nodes;
        self.click_action = s.click_action;
        if self.using_bus_topology {
            let _ = self.model.set_beta(s.beta);
            let _ = self.model.set_gamma(s.gamma);
        } else {
            let mut candidate = NetworkEpidemicModel::new(50, 0.05, 0.02);
            candidate
                .set_num_nodes(s.num_nodes)
                .map_err(|e| e.to_string())?;
            candidate.set_beta(s.beta).map_err(|e| e.to_string())?;
            candidate.set_gamma(s.gamma).map_err(|e| e.to_string())?;
            self.model = candidate;
            self.reset_network();
        }
        self.is_running = s.is_running;
        self.bump_state_rev();
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        eframe::egui::CentralPanel::default().show_inside(ui, |ui| {
            self.show_ui(ui);
        });
    }

    fn show_ui(&mut self, ui: &mut egui::Ui) {
        if crate::app::workspace_bus::with_linked_bus(|bus| {
            self.ingest_topology_channel(&bus.topology);
        })
        .is_none()
        {
            self.on_bus_unlinked();
        }

        if self.is_running {
            self.model.step_with_rng(&mut self.rng);
            #[cfg(test)]
            {
                self.sim_steps = self.sim_steps.saturating_add(1);
            }
            self.bump_state_rev();
            ui.ctx().request_repaint();
        }

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                self.draw_controls_column(ui);
            });
            ui.separator();
            ui.vertical(|ui| {
                self.draw_graph_column(ui);
            });
        });
    }
}

// [cite:algorithmic_information_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "NetworkPropagationTool",
        display_name: "Network Propagation",
        domain: "epidemiology",
        tags: &[],
        build: || Box::new(NetworkPropagationTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for NetworkPropagationTool {
    fn theory_description(&self) -> String {
        "Stochastic contact network epidemic propagation: infection transmission across graph edges with per-contact probability $\\beta$, recovery rate $\\gamma$, heterogeneous reproduction number $R_0 = \\frac{\\beta}{\\gamma}\\frac{\\langle k^2 \\rangle - \\langle k \\rangle}{\\langle k \\rangle}$, and transitions $S \\to I \\to R$.".into()
    }
    fn phonetic_description(&self) -> String {
        "Network epidemic propagation and contact graph theory".into()
    }
    fn theory_citation(&self) -> String {
        "Epidemic processes in complex networks (Pastor-Satorras et al., 2015); Networks (Newman, 2010)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_propagation_step_once_keeps_paused() {
        let mut tool = NetworkPropagationTool::default();
        let _ = tool.model.set_beta(1.0);
        assert!(!tool.is_running);
        tool.step_once();
        assert!(!tool.is_running);
    }

    #[test]
    fn test_network_propagation_builder_r0_and_click_state_toggles() {
        let mut tool = NetworkPropagationTool::default();
        let _ = tool.model.set_beta(0.5);
        let _ = tool.model.set_gamma(0.1);

        let mut ch = GraphTopologyChannel::default();
        // 4-node star graph: hub 0 connected to leaves 1, 2, 3.
        ch.publish(
            &[[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            &[(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)],
        );
        tool.ingest_topology_channel(&ch);
        assert_eq!(tool.model.num_nodes(), 4);
        let r0_star = tool.current_heterogeneous_r0();
        assert!(r0_star > 4.0, "Star hub should produce high R0, got {r0_star}");

        // Cycle click on node 1: Susceptible -> Infected -> Recovered -> Susceptible
        assert_eq!(tool.node_states()[1], NodeState::Susceptible);
        assert_eq!(tool.apply_node_click(1), Some(NodeState::Infected));
        assert_eq!(tool.apply_node_click(1), Some(NodeState::Recovered));
        assert_eq!(tool.apply_node_click(1), Some(NodeState::Susceptible));

        // Explicit Quarantine mode on hub 0
        tool.set_click_action(NodeClickAction::QuarantineRecovered);
        assert_eq!(tool.apply_node_click(0), Some(NodeState::Recovered));

        // Cutting hub edges drops heterogeneous_r0 immediately
        ch.publish(&[[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], &[]);
        tool.ingest_topology_channel(&ch);
        assert_eq!(tool.current_heterogeneous_r0(), 0.0);
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Graph Epidemic Spread",
    "domain": "epidemiology",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "num_nodes": 60,
    "beta": 0.15,
    "gamma": 0.05
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "network_propagation_starter",
        title: "Graph Epidemic Spread",
        description: "Stochastic SIR contagion propagation across a random geometric contact network.",
        category: "Network Epidemiology",
        icon: "🕸️",
        domain: "epidemiology",
        tool_name: Some("NetworkPropagationTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
