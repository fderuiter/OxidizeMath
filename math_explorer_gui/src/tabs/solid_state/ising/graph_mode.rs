//! Shared CSR [`GraphSpinSystem`] simulation and interactive canvas for [`IsingModelTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::preset::IsingInteractionMode;
use super::{ising_spin_color, IsingModelTool, MAX_SPATIAL_HISTORY};
use crate::accessibility::{
    announce_live_status, paint_canvas_focus_ring, register_canvas_accessibility,
    AccessibleHoverText,
};
use crate::app::{
    contrasting_text_color, render_topology_source_bar, SemanticPalette, SharedGraphTopology,
    SharedSimulationBus,
};
use crate::framework::{PlaybackAction, PlaybackTransport};
use eframe::egui::{self, Pos2};
use math_explorer::applied::game_theory::evolutionary::MatrixPayoff;
use math_explorer::physics::stat_mech::GraphSpinSystem;
use math_explorer::physics::stat_mech::KB;
use nalgebra::DMatrix;
use std::sync::Arc;

impl IsingModelTool {
    /// Returns `true` when `IsingModelTool` is currently driving a shared CSR [`GraphSpinSystem`].
    #[must_use]
    pub fn is_graph_mode(&self) -> bool {
        self.graph_system.is_some() && self.shared_topology.is_some()
    }

    /// Returns the currently synced shared bus revision, if active.
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
        if self.synced_bus_revision.take().is_some()
            || self.graph_system.is_some()
            || self.shared_topology.is_some()
        {
            self.graph_system = None;
            self.shared_topology = None;
            self.graph_running = false;
            self.graph_sweeps = 0.0;
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
            None => self.detach_shared_csr_graph(true),
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
        self.graph_system = Some(sys);
        self.shared_topology = Some(snap);
        self.graph_sweeps = 0.0;
        self.refresh_graph_observables();
        self.reset_trajectory_overlays();
        self.dirty = true;
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

    pub(super) fn draw_topology_source_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let mut changed = false;
        ui.add_enabled_ui(self.active_challenge.is_none(), |ui| {
            changed = render_topology_source_bar(
                ui,
                "ising_model",
                &mut self.topology_source,
                self.synced_bus_revision,
                None,
            );
        });
        if changed {
            self.sync_shared_topology(ctx);
        }
        if let Some(ref snap) = self.shared_topology {
            ui.small(format!(
                "Shared CSR Graph: |V| = {}, |E| = {}",
                snap.node_count(),
                snap.undirected_edges.len()
            ));
        }
    }

    pub(super) fn draw_graph_transport(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if let Some(action) = PlaybackTransport::in_thread(
            "ising_graph_transport",
            "Ising CSR Graph",
            self.graph_running,
        )
        .show(ui)
        {
            match action {
                PlaybackAction::Play => self.graph_running = true,
                PlaybackAction::Pause => self.graph_running = false,
                PlaybackAction::Step => {
                    self.graph_running = false;
                    let trials = self.active_graph_node_count().unwrap_or(16).max(16);
                    self.step_graph_trials(trials);
                }
                PlaybackAction::StepBack | PlaybackAction::ScrubTo(_) => {}
                PlaybackAction::Reset => {
                    if let Some(ref mut sys) = self.graph_system {
                        sys.randomize();
                    }
                    self.graph_sweeps = 0.0;
                    self.refresh_graph_observables();
                    self.reset_trajectory_overlays();
                }
            }
            ctx.request_repaint();
        }
    }

    pub(super) fn draw_graph_spin_canvas(&mut self, ui: &mut egui::Ui, side: f32) {
        let (response, painter) = ui.allocate_painter(egui::vec2(side, side), egui::Sense::click());
        let n = self.active_graph_node_count().unwrap_or(0);
        let label = "CSR Graph Ising Spin & Fermi Strategy Visualization";
        let _ = response.clone().accessible_hover_text(label);
        let summary = format!(
            "{n} CSR nodes, M = {:.3}, E = {:.3}, C = {:.1}%",
            self.magnetization,
            self.energy,
            self.cooperator_fraction() * 100.0
        );
        register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            label,
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &response);

        let palette = SemanticPalette::from_visuals(ui.visuals());
        painter.rect_filled(response.rect, 4.0, ui.visuals().extreme_bg_color);
        painter.rect_stroke(
            response.rect,
            4.0,
            ui.visuals().widgets.noninteractive.bg_stroke,
            egui::StrokeKind::Inside,
        );
        if let Some(idx) = self.paint_graph_spin_elements(&response, &painter, side, &palette) {
            if let Some(ref mut sys) = self.graph_system {
                sys.flip_spin(idx);
            }
            self.refresh_graph_observables();
            announce_live_status(ui.ctx(), format!("Flipped spin on CSR node {idx}"));
        }
        ui.horizontal(|ui| {
            ui.colored_label(ising_spin_color(1, &palette), "● +1 (C / ↑)");
            ui.colored_label(ising_spin_color(-1, &palette), "● -1 (D / ↓)");
            ui.small("Click a node to flip its spin");
        });
    }

    fn paint_graph_spin_elements(
        &self,
        response: &egui::Response,
        painter: &egui::Painter,
        side: f32,
        palette: &SemanticPalette,
    ) -> Option<usize> {
        let snap = self.shared_topology.as_ref()?;
        let sys_spins = self.graph_system.as_ref()?.spins();
        let pad = 22.0_f32.min(side * 0.15);
        let mapped = snap.map_positions_to_bounds(
            response.rect.min.x + pad,
            response.rect.min.y + pad,
            response.rect.max.x - pad,
            response.rect.max.y - pad,
        );
        for &(u, v, _) in &snap.undirected_edges {
            if u < mapped.len() && v < mapped.len() {
                let pu = Pos2::new(mapped[u][0], mapped[u][1]);
                let pv = Pos2::new(mapped[v][0], mapped[v][1]);
                painter.line_segment([pu, pv], (1.6, palette.graph_edge));
            }
        }
        let node_r = if mapped.len() > 64 { 6.0 } else { 11.0 };
        let click_pos = response
            .clicked()
            .then(|| response.interact_pointer_pos())
            .flatten();
        let mut clicked_idx = None;
        for (i, &[x, y]) in mapped.iter().enumerate() {
            let center = Pos2::new(x, y);
            if click_pos.is_some_and(|cp| (cp - center).length() <= node_r + 4.0) {
                clicked_idx = Some(i);
            }
            let spin = sys_spins.get(i).copied().unwrap_or(1);
            let fill = ising_spin_color(spin, palette);
            painter.circle(center, node_r, fill, (1.0, palette.node_stroke));
            if node_r >= 10.0 {
                painter.text(
                    center,
                    egui::Align2::CENTER_CENTER,
                    if spin > 0 { "+" } else { "-" },
                    egui::FontId::proportional(10.0),
                    contrasting_text_color(fill),
                );
            }
        }
        clicked_idx
    }
}
