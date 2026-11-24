//! Synchronizes [`NetworkPropagationTool`] with live CSR contact networks from [`SharedSimulationBus`]
//! using [`NetworkEpidemicModel::builder`] and `NetworkEpidemicModelBuilder::csr_adjacency`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::NetworkPropagationTool;
use crate::app::{SharedGraphTopology, SharedSimulationBus, render_topology_source_bar};
use eframe::egui;
use math_explorer::epidemiology::networks::{NetworkEpidemicModel, NodeState};

impl NetworkPropagationTool {
    /// Synchronizes the contact network from [`SharedSimulationBus`] or restores the seeded geometric graph.
    pub fn sync_shared_topology(&mut self, ctx: &egui::Context) {
        if self.is_challenge_locked() {
            if self.synced_bus_revision.take().is_some() {
                self.enforce_challenge_locks();
            }
            return;
        }
        match SharedSimulationBus::active_graph_for(ctx, self.topology_source) {
            Some(snap) => {
                if self.synced_bus_revision != Some(snap.revision)
                    && self.apply_shared_contact_graph(&snap)
                {
                    self.synced_bus_revision = Some(snap.revision);
                }
            }
            None => {
                if self.synced_bus_revision.take().is_some() {
                    let _ = self.model.set_num_nodes(self.local_num_nodes);
                    self.vaccinated_nodes.clear();
                    self.quarantined_edges.clear();
                    self.rebuild_seeded_graph();
                }
            }
        }
    }

    fn apply_shared_contact_graph(&mut self, snap: &SharedGraphTopology) -> bool {
        let n = snap.node_count();
        if n == 0 {
            return false;
        }
        let patient_zero = (0..n)
            .find(|&u| snap.sym_row_ptr[u] < snap.sym_row_ptr[u + 1])
            .unwrap_or(0);
        let positions = snap.map_positions_to_bounds(-170.0, -170.0, 170.0, 170.0);
        let mut states = vec![NodeState::Susceptible; n];
        states[patient_zero] = NodeState::Infected;
        let (beta, gamma) = (self.model.beta(), self.model.gamma());

        let Ok(built) = NetworkEpidemicModel::builder()
            .num_nodes(n)
            .parameters(beta, gamma)
            .states(states.clone())
            .positions(positions.clone())
            .csr_adjacency(&snap.sym_row_ptr, &snap.sym_col_idx)
            .and_then(|b| b.build())
        else {
            return false;
        };
        let Ok(built_baseline) = NetworkEpidemicModel::builder()
            .num_nodes(n)
            .parameters(beta, gamma)
            .states(states)
            .positions(positions)
            .csr_adjacency(&snap.sym_row_ptr, &snap.sym_col_idx)
            .and_then(|b| b.build())
        else {
            return false;
        };

        self.model = built;
        self.baseline_model = built_baseline;
        self.rng = oxidize_core::rng::OxidizeRng::new(self.seed);
        self.baseline_rng = oxidize_core::rng::OxidizeRng::new(self.seed);
        self.vaccinated_nodes.clear();
        self.quarantined_edges.clear();
        self.last_upstream_graph_rev = 0;
        self.is_running = false;
        self.clamp_focus_indices();
        #[cfg(test)]
        {
            self.sim_steps = 0;
        }
        self.bump_revision();
        true
    }

    /// Renders the `Topology Source` bar and live heterogeneous $R_0$ readout in controls.
    pub(super) fn render_topology_sync_bar(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let locked = self.is_challenge_locked();
        let mut changed = false;
        ui.add_enabled_ui(!locked, |ui| {
            changed = render_topology_source_bar(
                ui,
                "network_propagation",
                &mut self.topology_source,
                self.synced_bus_revision,
                None,
            );
        });
        if changed {
            self.sync_shared_topology(&ctx);
        }
        if self.synced_bus_revision.is_some() {
            ui.small(format!(
                "Shared Contact Network: |V| = {}, |E| = {}, R0(het) = {:.3}",
                self.model.num_nodes(),
                self.model.edges().len(),
                self.live_heterogeneous_r0()
            ));
        }
    }

    /// Returns scalar observables for Challenge HUD and multi-pane sync tests.
    #[must_use]
    pub(super) fn topology_observables(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("node_count", self.model.num_nodes() as f64),
            ("edge_count", self.model.edges().len() as f64),
            ("heterogeneous_r0", self.live_heterogeneous_r0()),
            (
                "synced_bus_revision",
                self.synced_bus_revision.unwrap_or(0) as f64,
            ),
        ]
    }

    /// Returns the currently synced shared bus revision, if active.
    #[cfg(test)]
    #[must_use]
    pub fn synced_bus_revision(&self) -> Option<u64> {
        self.synced_bus_revision
    }

    /// Evaluates the live heterogeneous basic reproduction number $R_0$ on the active contact graph.
    #[must_use]
    pub fn live_heterogeneous_r0(&self) -> f64 {
        self.model.effective_r0()
    }

    /// Returns the number of nodes in the active contact network.
    #[cfg(test)]
    #[must_use]
    pub fn active_node_count(&self) -> usize {
        self.model.num_nodes()
    }
}
