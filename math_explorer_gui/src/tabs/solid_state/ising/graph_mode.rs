//! Arbitrary signed-graph spin coupling (`GraphSpinSystem`) for [`super::IsingModelTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{ising_spin_color, IsingModelTool};
use crate::accessibility::AccessibleTheoryHover;
use crate::app::data_bus::ui::{
    render_follow_workspace_toggle, render_node_budget_banner, sync_consumer_follow_flag,
};
use crate::app::data_bus::{
    read_shared_graph, seed_shared_graph_if_empty, SharedGraph, MAX_SPIN_NODE_BUDGET,
};
use crate::app::{contrasting_text_color, SemanticPalette};
use eframe::egui;
use math_explorer::physics::stat_mech::GraphSpinSystem;
use math_explorer::physics::stat_mech::KB;
use std::sync::Arc;

pub(super) struct GraphSpinMode {
    pub(super) follow_workspace_graph: bool,
    pub(super) applied_bus_revision: u64,
    pub(super) clamped_source_nodes: Option<usize>,
    pub(super) active_graph: Option<Arc<SharedGraph>>,
    pub(super) seeded_graph: Option<SharedGraph>,
    pub(super) pending_seed_publish: bool,
    pub(super) system: Option<GraphSpinSystem>,
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
            system: None,
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

    pub(crate) fn is_graph_spin_mode(&self) -> bool {
        self.graph_mode.system.is_some()
    }

    pub(crate) fn frustrated_edge_count(&self) -> usize {
        self.graph_mode.frustrated_edges
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
        self.graph_mode.follow_workspace_graph
    }

    #[cfg(all(
        test,
        feature = "biology",
        feature = "applied",
        feature = "epidemiology",
        feature = "physics",
        feature = "pure_math"
    ))]
    pub(crate) fn graph_spin_count(&self) -> Option<usize> {
        self.graph_mode
            .system
            .as_ref()
            .map(GraphSpinSystem::num_nodes)
    }

    pub(super) fn refresh_graph_metrics(&mut self) -> bool {
        let Some(sys) = &self.graph_mode.system else {
            return false;
        };
        self.magnetization = sys.mean_magnetization();
        self.energy = sys.hamiltonian(self.j_coupling, self.h_field);
        self.graph_mode.frustrated_edges = sys.frustrated_edge_count(self.j_coupling);
        true
    }

    pub(crate) fn apply_shared_spin_graph(&mut self, shared: &Arc<SharedGraph>) {
        let (clamped, orig_count) = shared.clamped_to_budget(MAX_SPIN_NODE_BUDGET);
        self.graph_mode.clamped_source_nodes = orig_count;
        self.graph_mode.applied_bus_revision = shared.revision;
        if clamped.nodes.is_empty() {
            self.graph_mode.system = None;
            self.graph_mode.active_graph = None;
            self.graph_mode.frustrated_edges = 0;
            return;
        }
        let sys = GraphSpinSystem::new(clamped.nodes.len(), &clamped.edges, Some(42));
        self.graph_mode.active_graph = Some(clamped);
        self.graph_mode.system = Some(sys);
        self.refresh_graph_metrics();
    }

    pub(super) fn sync_graph_from_bus(&mut self, ctx: &egui::Context) {
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

    pub(super) fn step_graph_spins(&mut self, sweeps: usize) -> bool {
        let Some(sys) = &mut self.graph_mode.system else {
            return false;
        };
        let j_scale = if self.j_coupling.abs() < 1e-6 {
            1.0
        } else {
            self.j_coupling.abs()
        };
        let t_phys = self.temperature * j_scale / KB;
        sys.evolve(sweeps.max(1), t_phys, self.j_coupling, self.h_field);
        self.refresh_graph_metrics();
        true
    }

    pub(super) fn reset_graph_spins(&mut self) -> bool {
        let Some(active) = self.graph_mode.active_graph.clone() else {
            return false;
        };
        self.apply_shared_spin_graph(&active);
        true
    }

    pub(super) fn render_graph_spin_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let rev_opt = (self.graph_mode.applied_bus_revision > 0)
            .then_some(self.graph_mode.applied_bus_revision);
        if render_follow_workspace_toggle(
            ui,
            &mut self.graph_mode.follow_workspace_graph,
            "Ising Model",
            rev_opt,
        ) && self.graph_mode.follow_workspace_graph
        {
            self.graph_mode.applied_bus_revision = 0;
            self.sync_graph_from_bus(ctx);
        }
        render_node_budget_banner(
            ui,
            self.graph_mode.clamped_source_nodes,
            MAX_SPIN_NODE_BUDGET,
            "graph spins",
        );
    }

    pub(super) fn render_graph_spin_canvas(&self, ui: &mut egui::Ui) -> bool {
        let (Some(sys), Some(graph)) = (&self.graph_mode.system, &self.graph_mode.active_graph)
        else {
            return false;
        };
        let avail = ui.available_size().max(egui::vec2(160.0, 160.0));
        let (response, painter) =
            ui.allocate_painter(avail, egui::Sense::focusable_noninteractive());
        let response = response.accessible_theory_hover(self);
        let label = format!(
            "Graph Ising spin network ({} spins, {} frustrated edges, M = {:.3}, E = {:.3})",
            sys.num_nodes(),
            self.graph_mode.frustrated_edges,
            self.magnetization,
            self.energy
        );
        ui.ctx().accesskit_node_builder(response.id, |b| {
            b.set_role(egui::accesskit::Role::GraphicsDocument);
            b.set_label(label);
        });

        let palette = SemanticPalette::from_visuals(ui.visuals());
        paint_spin_graph(
            &painter,
            response.rect,
            graph,
            sys,
            self.j_coupling,
            &palette,
        );
        crate::accessibility::paint_canvas_focus_ring(ui, &response);
        true
    }
}

fn paint_spin_graph(
    painter: &egui::Painter,
    rect: egui::Rect,
    graph: &SharedGraph,
    sys: &GraphSpinSystem,
    j_coupling: f64,
    palette: &SemanticPalette,
) {
    painter.rect_filled(rect, 4.0_f32, palette.surface_fill);
    let inner = rect.shrink(28.0_f32);
    let pos_of = |idx: usize| -> egui::Pos2 {
        let [nx, ny] = graph.nodes[idx].norm_pos;
        egui::pos2(
            inner.left() + nx.clamp(0.0_f32, 1.0_f32) * inner.width().max(1.0_f32),
            inner.top() + ny.clamp(0.0_f32, 1.0_f32) * inner.height().max(1.0_f32),
        )
    };
    let spins = sys.spins();
    for &(u, v, w) in &graph.edges {
        if u >= spins.len() || v >= spins.len() || u == v {
            continue;
        }
        let frustrated = j_coupling * w * f64::from(spins[u]) * f64::from(spins[v]) < -1e-12;
        let color = if frustrated {
            palette.warning
        } else {
            palette.muted_text
        };
        let width = if frustrated { 2.6_f32 } else { 1.4_f32 };
        painter.line_segment([pos_of(u), pos_of(v)], egui::Stroke::new(width, color));
    }
    let radius = (220.0_f32 / (spins.len().max(4) as f32).sqrt()).clamp(4.0_f32, 14.0_f32);
    for (idx, &spin) in spins.iter().enumerate() {
        let center = pos_of(idx);
        let fill = ising_spin_color(spin, palette);
        let border = contrasting_text_color(fill);
        painter.circle_filled(center, radius, fill);
        painter.circle_stroke(center, radius, egui::Stroke::new(1.2_f32, border));
    }
}
