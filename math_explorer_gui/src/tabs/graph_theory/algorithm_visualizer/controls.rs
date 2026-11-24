//! Sidebar controls, terminal selectors, and playback transport for [`super::AlgorithmVisualizerTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{Algorithm, AlgorithmVisualizerTool};
use crate::accessibility::AccessibleHoverText;
use crate::app::{render_topology_source_bar, SharedGraphTopology, SharedSimulationBus};
use eframe::egui::{self, Pos2};
use math_explorer::pure_math::graph_theory::graph::Graph;

impl AlgorithmVisualizerTool {
    /// Synchronizes this tool's graph with `SharedSimulationBus` or restores the 4x4 sample grid.
    pub fn sync_shared_topology(&mut self, ctx: &egui::Context) {
        match SharedSimulationBus::active_graph_for(ctx, self.topology_source) {
            Some(snap) => {
                if self.synced_bus_revision != Some(snap.revision) {
                    self.apply_shared_graph(&snap);
                    self.synced_bus_revision = Some(snap.revision);
                }
            }
            None => {
                if self.synced_bus_revision.take().is_some() {
                    self.build_sample_graph();
                    self.bump_revision();
                }
            }
        }
    }

    fn apply_shared_graph(&mut self, snap: &SharedGraphTopology) {
        if snap.node_count() == 0 {
            return;
        }
        self.graph = Graph::new();
        self.node_indices.clear();
        self.node_positions.clear();
        self.edges.clear();
        self.directed = snap.directed;
        self.is_custom_topology = false;

        let mapped = snap.map_positions_to_bounds(70.0, 70.0, 380.0, 340.0);
        for (idx_in_snap, &orig_id) in snap.node_ids.iter().enumerate() {
            let [x, y] = mapped[idx_in_snap];
            let pos = Pos2::new(x, y);
            let g_idx = self.graph.add_node(pos);
            self.node_indices.insert(orig_id, g_idx);
            self.node_positions.insert(orig_id, pos);
        }
        self.populate_shared_graph_edges(snap);
        if self
            .start_node
            .is_none_or(|id| !self.node_indices.contains_key(&id))
        {
            let first_id = snap.node_ids[0];
            self.start_node = Some(first_id);
            self.focused_node = first_id;
        }
        self.sink_node = self
            .sink_node
            .filter(|id| self.node_positions.contains_key(id) && Some(*id) != self.start_node)
            .or_else(|| self.effective_sink_node(self.start_node));
        self.bump_revision();
        self.run_algorithm();
    }

    fn populate_shared_graph_edges(&mut self, snap: &SharedGraphTopology) {
        if snap.directed {
            for u in 0..snap.node_count() {
                let u_id = snap.node_ids[u];
                for idx in snap.row_ptr[u]..snap.row_ptr[u + 1] {
                    let v_id = snap.node_ids[snap.col_idx[idx]];
                    if let (Some(&u_idx), Some(&v_idx)) =
                        (self.node_indices.get(&u_id), self.node_indices.get(&v_id))
                    {
                        let safe_w = snap.weights[idx].abs().max(0.01);
                        self.edges.push((u_id, v_id, safe_w));
                        self.graph.add_edge(u_idx, v_idx, safe_w);
                    }
                }
            }
        } else {
            for &(u, v, w) in &snap.undirected_edges {
                let u_id = snap.node_ids[u];
                let v_id = snap.node_ids[v];
                if let (Some(&u_idx), Some(&v_idx)) =
                    (self.node_indices.get(&u_id), self.node_indices.get(&v_id))
                {
                    let safe_w = w.abs().max(0.01);
                    self.edges.push((u_id, v_id, safe_w));
                    self.graph.add_edge(u_idx, v_idx, safe_w);
                }
            }
        }
    }

    /// Returns the currently synced `SharedSimulationBus` revision, if active.
    #[cfg(all(test, feature = "epidemiology"))]
    #[must_use]
    pub fn synced_bus_revision(&self) -> Option<u64> {
        self.synced_bus_revision
    }

    /// Returns the number of vertices currently active in the visualizer.
    #[cfg(all(test, feature = "epidemiology"))]
    #[must_use]
    pub fn active_node_count(&self) -> usize {
        self.node_positions.len()
    }

    pub(super) fn render_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id =
            crate::framework::scoped_panel_id(parent_ui, "algorithm_visualizer_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Algorithm Visualizer");
            let ctx = ui.ctx().clone();
            if render_topology_source_bar(
                ui,
                "algorithm_visualizer",
                &mut self.topology_source,
                self.synced_bus_revision,
                None,
            ) {
                self.bump_revision();
                self.sync_shared_topology(&ctx);
            }
            ui.separator();
            self.render_algorithm_radios(ui);
            ui.separator();
            self.render_terminal_selectors(ui);
            ui.separator();
            self.render_playback_controls(ui);
            ui.separator();
            self.render_topology_io_controls(ui);
            ui.separator();
            ui.label(
                "Click a node (or Shift+Click for Sink t in Max-Flow) or use Arrow keys + Enter.",
            );
        });
    }

    fn render_algorithm_radios(&mut self, ui: &mut egui::Ui) {
        ui.label("Select Algorithm:");
        let mut changed = false;
        for (alg, label) in [
            (Algorithm::Dijkstra, "Dijkstra (Predecessor Tree)"),
            (Algorithm::Bfs, "BFS (Discovery Tree)"),
            (Algorithm::Dfs, "DFS (Discovery Tree)"),
            (Algorithm::MaxFlowMinCut, "Edmonds-Karp Max-Flow / Min-Cut"),
            (
                Algorithm::MinimumSpanningTree,
                "Kruskal Minimum Spanning Tree",
            ),
        ] {
            if ui
                .radio_value(&mut self.selected_algorithm, alg, label)
                .clicked()
            {
                changed = true;
            }
        }
        if changed {
            self.bump_revision();
            self.run_algorithm();
        }
    }

    fn render_terminal_selectors(&mut self, ui: &mut egui::Ui) {
        let mut node_ids: Vec<usize> = self.node_positions.keys().copied().collect();
        node_ids.sort_unstable();
        let start_text = self
            .start_node
            .map_or_else(|| "None".to_string(), |id| format!("Node {id}"));
        ui.horizontal(|ui| {
            ui.label("Start Node:");
            egui::ComboBox::from_id_salt("algorithm_visualizer_start_node")
                .selected_text(start_text)
                .show_ui(ui, |ui| {
                    for &id in &node_ids {
                        if ui
                            .selectable_label(self.start_node == Some(id), format!("Node {id}"))
                            .clicked()
                        {
                            self.start_node = Some(id);
                            if self.synced_bus_revision.is_none() {
                                self.local_start_node = Some(id);
                            }
                            self.focused_node = id;
                            if self.sink_node == Some(id) {
                                self.sink_node = self.effective_sink_node(Some(id));
                            }
                            self.bump_revision();
                            self.run_algorithm();
                            crate::accessibility::announce_live_status(
                                ui.ctx(),
                                format!("Start node set to Node {id}"),
                            );
                        }
                    }
                });
        });
        if self.selected_algorithm == Algorithm::MaxFlowMinCut {
            self.render_sink_selector(ui, &node_ids);
        }
    }

    fn render_sink_selector(&mut self, ui: &mut egui::Ui, node_ids: &[usize]) {
        let cur_sink = self.effective_sink_node(self.start_node);
        let sink_text = cur_sink.map_or_else(|| "None".to_string(), |id| format!("Node {id}"));
        ui.horizontal(|ui| {
            ui.label("Sink Node:");
            egui::ComboBox::from_id_salt("algorithm_visualizer_sink_node")
                .selected_text(sink_text)
                .show_ui(ui, |ui| {
                    for &id in node_ids {
                        if Some(id) == self.start_node {
                            continue;
                        }
                        if ui
                            .selectable_label(cur_sink == Some(id), format!("Node {id}"))
                            .clicked()
                        {
                            self.sink_node = Some(id);
                            self.bump_revision();
                            self.run_algorithm();
                        }
                    }
                });
        });
    }

    fn render_playback_controls(&mut self, ui: &mut egui::Ui) {
        let total = self.total_steps();
        if total == 0 {
            return;
        }
        ui.label(format!(
            "Animation Step: {} / {}",
            self.animation_step, total
        ));
        self.render_playback_buttons(ui, total);
        if ui
            .add(egui::Slider::new(&mut self.animation_step, 0..=total).text("Step"))
            .changed()
        {
            self.step_timer = 0.0;
        }
        ui.add(
            egui::Slider::new(&mut self.playback_speed, 0.5..=5.0)
                .text("Speed")
                .suffix("x"),
        );
    }

    fn render_playback_buttons(&mut self, ui: &mut egui::Ui, total: usize) {
        ui.horizontal(|ui| {
            if ui
                .button("↻ Reset")
                .accessible_hover_text("Restart the algorithm visualization")
                .clicked()
            {
                self.animation_step = 0;
                self.is_playing = false;
                self.step_timer = 0.0;
            }
            let (lbl, tip) = if self.is_playing {
                ("⏸ Pause", "Pause the algorithm visualization")
            } else {
                (
                    "▶ Play",
                    "Start or resume continuous algorithm visualization",
                )
            };
            if ui.button(lbl).accessible_hover_text(tip).clicked() {
                if !self.is_playing && self.animation_step >= total {
                    self.animation_step = 0;
                }
                self.is_playing = !self.is_playing;
                self.step_timer = 0.0;
            }
            let can_step = self.animation_step < total;
            if ui
                .add_enabled(can_step, egui::Button::new("▶ Step"))
                .accessible_hover_text("Advance visualization by one step")
                .clicked()
            {
                self.animation_step += 1;
                self.is_playing = false;
                self.step_timer = 0.0;
            }
            if ui
                .add_enabled(can_step, egui::Button::new("⏹ Finish"))
                .accessible_hover_text("Skip to the end of the visualization")
                .clicked()
            {
                self.animation_step = total;
                self.is_playing = false;
                self.step_timer = 0.0;
            }
        });
    }
}

// [cite:graph_parameters_rust]
