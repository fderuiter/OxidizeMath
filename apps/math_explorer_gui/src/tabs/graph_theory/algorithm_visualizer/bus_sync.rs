//! Workspace data bus synchronization for [`super::AlgorithmVisualizerTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{AlgorithmVisualizerState, AlgorithmVisualizerTool};
use crate::app::data_bus::ui::sync_consumer_follow_flag;
use crate::app::data_bus::{
    read_shared_graph, SharedGraph, SESSION_MAX_BUS_EDGES, SESSION_MAX_BUS_NODES,
};
use eframe::egui::{self, Pos2};
use math_explorer::pure_math::graph_theory::graph::Graph;
use std::sync::Arc;

impl AlgorithmVisualizerTool {
    pub(super) fn apply_loaded_topology_state(
        &mut self,
        loaded: &AlgorithmVisualizerState,
    ) -> Result<(), String> {
        if let Some(ref g) = loaded.frozen_graph {
            g.validate(SESSION_MAX_BUS_NODES, SESSION_MAX_BUS_EDGES)?;
        }
        self.synced_bus_revision = None;
        if let Some(compact) = &loaded.custom_topology {
            compact.validate()?;
            self.active_shared_graph = None;
            self.load_compact_topology(compact);
        } else if let Some(ref g) = loaded.frozen_graph {
            self.apply_shared_graph(Arc::new(g.clone()));
        } else if self.is_custom_topology || self.active_shared_graph.take().is_some() {
            self.applied_bus_revision = 0;
            self.build_sample_graph();
        }
        let check_local = !loaded.follow_workspace_graph
            || loaded.custom_topology.is_some()
            || loaded.topology_source == crate::app::TopologySource::LocalPreset;
        if let Some(id) = loaded.start_node {
            if check_local && !self.node_indices.contains_key(&id) {
                return Err(format!("Unknown start_node id: {id}"));
            }
            if id >= SESSION_MAX_BUS_NODES {
                return Err(format!("start_node id out of bounds: {id}"));
            }
        }
        if let Some(sink) = loaded.sink_node {
            if (check_local && !self.node_indices.contains_key(&sink))
                || sink >= SESSION_MAX_BUS_NODES
                || Some(sink) == loaded.start_node
            {
                return Err(format!("Invalid sink_node id: {sink}"));
            }
        }
        Ok(())
    }

    pub(crate) fn apply_shared_graph(&mut self, shared: Arc<SharedGraph>) {
        self.graph = Graph::new();
        self.node_indices.clear();
        self.node_positions.clear();
        self.edges.clear();

        for node in &shared.nodes {
            let pos = Pos2::new(
                60.0_f32 + node.norm_pos[0].clamp(0.0_f32, 1.0_f32) * 300.0_f32,
                60.0_f32 + node.norm_pos[1].clamp(0.0_f32, 1.0_f32) * 260.0_f32,
            );
            let idx = self.graph.add_node(pos);
            self.node_indices.insert(node.id, idx);
            self.node_positions.insert(node.id, pos);
        }

        for &(u, v, w) in &shared.edges {
            if let (Some(nu), Some(nv)) = (shared.nodes.get(u), shared.nodes.get(v)) {
                if let (Some(&u_idx), Some(&v_idx)) =
                    (self.node_indices.get(&nu.id), self.node_indices.get(&nv.id))
                {
                    let cost = w.abs().max(0.1);
                    self.edges.push((nu.id, nv.id, cost));
                    self.graph.add_edge(u_idx, v_idx, cost);
                }
            }
        }

        self.directed = false;
        self.is_custom_topology = false;
        if !self
            .start_node
            .is_some_and(|id| self.node_indices.contains_key(&id))
        {
            self.start_node = self.node_positions.keys().copied().min();
        }
        if let Some(id) = self.start_node {
            self.focused_node = id;
        }
        self.sink_node = self.effective_sink_node(self.start_node);
        self.applied_bus_revision = shared.revision;
        self.active_shared_graph = Some(shared);
        self.bump_revision();
        self.run_algorithm();
    }

    pub(super) fn sync_from_bus(&mut self, ctx: &egui::Context) {
        if self.topology_source == crate::app::TopologySource::LocalPreset {
            if self.active_shared_graph.take().is_some() && !self.is_custom_topology {
                self.applied_bus_revision = 0;
                self.build_sample_graph();
                self.bump_revision();
            }
            return;
        }
        if sync_consumer_follow_flag(
            ctx,
            &mut self.follow_workspace_graph,
            "Algorithm Visualizer",
        ) && self.follow_workspace_graph
        {
            self.is_custom_topology = false;
            self.applied_bus_revision = 0;
        }
        if !self.follow_workspace_graph {
            return;
        }
        if let Some(shared) = read_shared_graph(ctx) {
            if shared.revision != self.applied_bus_revision {
                self.apply_shared_graph(shared);
            }
        }
    }

    #[cfg(all(
        test,
        feature = "biology",
        feature = "applied",
        feature = "epidemiology",
        feature = "physics",
        feature = "pure_math"
    ))]
    pub(crate) fn visit_count(&self) -> usize {
        self.visit_order.len()
    }
}
