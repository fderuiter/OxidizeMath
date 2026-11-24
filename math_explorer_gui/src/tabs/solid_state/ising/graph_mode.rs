//! Shared CSR and `WorkspaceDataBus` [`GraphSpinSystem`] synchronization for [`IsingModelTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::preset::IsingInteractionMode;
use super::{IsingModelTool, MAX_SPATIAL_HISTORY};
use crate::app::data_bus::ui::sync_consumer_follow_flag;
use crate::app::data_bus::{
    read_shared_graph, seed_shared_graph_if_empty, SharedGraph, MAX_SPIN_NODE_BUDGET,
};
use crate::app::{SharedGraphTopology, SharedSimulationBus, TopologySource};
use eframe::egui;
use math_explorer::applied::game_theory::evolutionary::MatrixPayoff;
use math_explorer::physics::stat_mech::{GraphSpinSystem, KB};
use nalgebra::DMatrix;
use std::sync::Arc;

pub(super) struct GraphSpinMode {
    pub(super) follow_workspace_graph: bool,
    pub(super) applied_bus_revision: u64,
    pub(super) clamped_source_nodes: Option<usize>,
    pub(super) active_graph: Option<Arc<SharedGraph>>,
    pub(super) seeded_graph: Option<SharedGraph>,
    pub(super) pending_seed_publish: bool,
    pub(super) frustrated_edges: usize,
}

impl Default for GraphSpinMode {
    fn default() -> Self {
        Self {
            follow_workspace_graph: true,
            applied_bus_revision: 0,
            clamped_source_nodes: None,
            active_graph: None,
            seeded_graph: None,
            pending_seed_publish: false,
            frustrated_edges: 0,
        }
    }
}

impl IsingModelTool {
    pub(super) fn sync_worker_target_state(
        &mut self,
        state: Option<crate::async_sim::SimWorkerState>,
        running: Option<bool>,
    ) {
        use crate::async_sim::{SimCommand, SimWorkerState};
        let target_state = state.unwrap_or_else(|| {
            if running.unwrap_or(false) {
                SimWorkerState::Running
            } else {
                SimWorkerState::Ready
            }
        });
        if target_state == SimWorkerState::Running {
            if !self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Start);
            }
        } else if self.worker.is_running() {
            let _ = self.worker.send_command(SimCommand::Pause);
        }
    }

    /// Returns `true` when `IsingModelTool` is currently driving a shared CSR [`GraphSpinSystem`].
    #[must_use]
    pub fn is_graph_mode(&self) -> bool {
        self.active_challenge.is_none()
            && self.graph_system.is_some()
            && (self.shared_topology.is_some() || self.graph_mode.active_graph.is_some())
    }

    /// Alias for [`Self::is_graph_mode`] used by cross-domain `WorkspaceDataBus` coupling checks.
    #[cfg(test)]
    #[must_use]
    pub fn is_graph_spin_mode(&self) -> bool {
        self.is_graph_mode()
    }

    /// Returns the number of frustrated undirected bonds ($J w_{ij} s_i s_j < 0$) in the active spin graph.
    #[must_use]
    pub fn frustrated_edge_count(&self) -> usize {
        self.graph_mode.frustrated_edges
    }

    /// Returns whether `IsingModelTool` follows `WorkspaceDataBus` topology updates.
    #[cfg(test)]
    #[must_use]
    pub fn follow_workspace_graph(&self) -> bool {
        self.graph_mode.follow_workspace_graph
    }

    /// Returns the active spin node count when in graph mode.
    #[cfg(test)]
    #[must_use]
    pub fn graph_spin_count(&self) -> Option<usize> {
        self.active_graph_node_count()
    }

    /// Returns the currently synced `SharedSimulationBus` revision, if active.
    #[cfg(test)]
    #[must_use]
    pub fn synced_bus_revision(&self) -> Option<u64> {
        self.synced_bus_revision
    }

    /// Returns the active node count when running on a shared CSR graph.
    #[must_use]
    pub fn active_graph_node_count(&self) -> Option<usize> {
        self.graph_system.as_ref().map(GraphSpinSystem::n_nodes)
    }

    pub(super) fn detach_shared_csr_graph(&mut self, refresh_lattice_overlays: bool) {
        let had_any = self.synced_bus_revision.take().is_some()
            || self.shared_topology.take().is_some()
            || self.graph_mode.active_graph.take().is_some()
            || self.graph_system.take().is_some();
        self.graph_mode.applied_bus_revision = 0;
        self.graph_mode.clamped_source_nodes = None;
        self.graph_mode.frustrated_edges = 0;
        self.graph_running = false;
        self.graph_sweeps = 0.0;
        if had_any {
            if refresh_lattice_overlays {
                self.sync_metrics_from_latest();
                self.reset_trajectory_overlays();
            }
            self.dirty = true;
        }
    }

    /// Synchronizes [`GraphSpinSystem`] with [`SharedSimulationBus`] or reverts to the 100x100 lattice.
    pub fn sync_shared_topology(&mut self, ctx: &egui::Context) {
        if self.active_challenge.is_some() {
            self.detach_shared_csr_graph(true);
            return;
        }
        match SharedSimulationBus::active_graph_for(ctx, self.topology_source) {
            Some(snap) => {
                if self.synced_bus_revision != Some(snap.revision)
                    && self.apply_shared_csr_graph(snap.clone())
                {
                    self.synced_bus_revision = Some(snap.revision);
                }
            }
            None => {
                if self.synced_bus_revision.is_some() || self.shared_topology.is_some() {
                    self.detach_shared_csr_graph(true);
                }
            }
        }
    }

    fn apply_shared_csr_graph(&mut self, snap: Arc<SharedGraphTopology>) -> bool {
        let n = snap.node_count();
        if n == 0 {
            return false;
        }
        let Ok(sys) = GraphSpinSystem::try_from_csr(
            n,
            snap.sym_row_ptr.clone(),
            snap.sym_col_idx.clone(),
            snap.sym_weights.clone(),
            Some(42),
        ) else {
            return false;
        };
        self.graph_mode.active_graph = None;
        self.graph_mode.applied_bus_revision = 0;
        self.graph_mode.clamped_source_nodes = None;
        self.graph_system = Some(sys);
        self.shared_topology = Some(snap);
        self.graph_sweeps = 0.0;
        self.refresh_graph_observables();
        self.reset_trajectory_overlays();
        self.dirty = true;
        true
    }

    pub(crate) fn apply_shared_spin_graph(&mut self, shared: &Arc<SharedGraph>) {
        let (clamped, orig_count) = shared.clamped_to_budget(MAX_SPIN_NODE_BUDGET);
        self.graph_mode.clamped_source_nodes = orig_count;
        self.graph_mode.applied_bus_revision = shared.revision;
        if clamped.nodes.is_empty() {
            self.graph_system = None;
            self.graph_mode.active_graph = None;
            self.graph_mode.frustrated_edges = 0;
            return;
        }
        let sys = GraphSpinSystem::new(clamped.nodes.len(), &clamped.edges, Some(42));
        self.shared_topology = None;
        self.synced_bus_revision = None;
        self.graph_mode.active_graph = Some(clamped);
        self.graph_system = Some(sys);
        self.graph_sweeps = 0.0;
        self.refresh_graph_observables();
        self.reset_trajectory_overlays();
        self.dirty = true;
    }

    pub(super) fn sync_graph_from_bus(&mut self, ctx: &egui::Context) {
        if self.active_challenge.is_some() {
            self.detach_shared_csr_graph(true);
            return;
        }
        if self.topology_source == TopologySource::LocalPreset {
            if self.graph_mode.active_graph.is_some() && self.graph_mode.seeded_graph.is_none() {
                self.detach_shared_csr_graph(true);
            }
            return;
        }
        if sync_consumer_follow_flag(
            ctx,
            &mut self.graph_mode.follow_workspace_graph,
            "Ising Model",
        ) && self.graph_mode.follow_workspace_graph
        {
            self.graph_mode.applied_bus_revision = 0;
        }
        if self.graph_mode.pending_seed_publish {
            self.graph_mode.pending_seed_publish = false;
            if let Some(ref seeded) = self.graph_mode.seeded_graph {
                let arc = seed_shared_graph_if_empty(ctx, seeded);
                if self.graph_mode.follow_workspace_graph {
                    self.apply_shared_spin_graph(&arc);
                }
            }
        }
        if !self.graph_mode.follow_workspace_graph {
            return;
        }
        if let Some(shared) = read_shared_graph(ctx) {
            if shared.revision != self.graph_mode.applied_bus_revision {
                self.apply_shared_spin_graph(&shared);
            }
        }
    }

    pub(super) fn refresh_graph_metrics(&mut self) -> bool {
        if !self.is_graph_mode() {
            return false;
        }
        self.refresh_graph_observables();
        true
    }

    pub(super) fn refresh_graph_observables(&mut self) {
        let Some(ref sys) = self.graph_system else {
            return;
        };
        let n = sys.n_nodes().max(1) as f64;
        let norm_m = sys.normalized_magnetization();
        self.magnetization = norm_m;
        self.uniform_magnetization = norm_m;
        self.staggered_magnetization = norm_m;
        self.graph_mode.frustrated_edges = sys.frustrated_edge_count(self.j_coupling);
        self.energy = match self.mode {
            IsingInteractionMode::Ferromagnetic => {
                sys.total_energy(self.j_coupling, self.h_field) / n
            }
            IsingInteractionMode::FermiGame => {
                let mat = DMatrix::from_row_slice(2, 2, &self.game_payoffs);
                MatrixPayoff::new(mat).map_or(0.0, |p| -sys.mean_game_payoff(&p))
            }
        };
    }

    /// Advances the active CSR [`GraphSpinSystem`] by `num_trials` elementary updates.
    pub fn step_graph_trials(&mut self, num_trials: usize) {
        let mode = self.mode;
        let temp = self.temperature;
        let j = self.j_coupling;
        let h = self.h_field;
        let payoffs = self.game_payoffs;
        let Some(ref mut sys) = self.graph_system else {
            return;
        };
        let n = sys.n_nodes().max(1);
        match mode {
            IsingInteractionMode::Ferromagnetic => {
                let t_phys = temp.max(1e-3) / KB;
                sys.evolve(num_trials, t_phys, j, h);
            }
            IsingInteractionMode::FermiGame => {
                let mat = DMatrix::from_row_slice(2, 2, &payoffs);
                if let Ok(payoff) = MatrixPayoff::new(mat) {
                    sys.evolve_fermi_game(num_trials, temp, &payoff);
                }
            }
        }
        self.graph_sweeps += (num_trials as f64) / (n as f64);
        self.refresh_graph_observables();
        if self.spatial_history.len() >= MAX_SPATIAL_HISTORY {
            self.spatial_history.pop_front();
        }
        self.spatial_history
            .push_back([self.graph_sweeps, self.cooperator_fraction()]);
    }

    pub(super) fn reset_graph_spins(&mut self) -> bool {
        let Some(ref mut sys) = self.graph_system else {
            return false;
        };
        sys.randomize();
        self.graph_sweeps = 0.0;
        self.refresh_graph_observables();
        self.reset_trajectory_overlays();
        true
    }
}
