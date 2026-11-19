#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::AlgorithmVisualizerTool;
use egui_plot::export_primitives::ExportPrimitive;

impl AlgorithmVisualizerTool {
    pub(super) fn step_export_traversal(&mut self) {
        if self.visit_order.is_empty() {
            self.run_algorithm();
        }
        if !self.visit_order.is_empty() {
            self.animation_step = (self.animation_step + 1) % (self.visit_order.len() + 1);
        }
    }

    pub(super) fn cold_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = crate::app::palette::DARK_PALETTE;
        let mut primitives = Vec::with_capacity(self.edges.len() + 4);
        self.append_edge_primitives(&palette, &mut primitives);
        self.append_node_primitives(&palette, &mut primitives);
        primitives
    }

    fn append_edge_primitives(
        &self,
        palette: &crate::app::SemanticPalette,
        primitives: &mut Vec<ExportPrimitive>,
    ) {
        for (idx, &(u, v, _)) in self.edges.iter().enumerate() {
            if let (Some(&pos_u), Some(&pos_v)) =
                (self.node_positions.get(&u), self.node_positions.get(&v))
            {
                let u_vis = self
                    .visit_order
                    .iter()
                    .take(self.animation_step)
                    .any(|&id| id == u);
                let v_vis = self
                    .visit_order
                    .iter()
                    .take(self.animation_step)
                    .any(|&id| id == v);
                let color = if u_vis && v_vis {
                    palette.success
                } else {
                    palette.graph_edge
                };
                primitives.push(ExportPrimitive::Polyline {
                    name: format!("edge_{idx}"),
                    points: vec![
                        [f64::from(pos_u.x), -f64::from(pos_u.y)],
                        [f64::from(pos_v.x), -f64::from(pos_v.y)],
                    ],
                    color,
                    stroke_width: 2.0,
                    closed: false,
                    fill: None,
                });
            }
        }
    }

    fn append_node_primitives(
        &self,
        palette: &crate::app::SemanticPalette,
        primitives: &mut Vec<ExportPrimitive>,
    ) {
        let mut unvisited = Vec::new();
        let mut visited = Vec::new();
        let mut current = Vec::new();
        let mut start = Vec::new();

        let mut sorted_ids: Vec<usize> = self.node_positions.keys().copied().collect();
        sorted_ids.sort_unstable();
        for id in sorted_ids {
            let pos = self.node_positions[&id];
            let pt = [f64::from(pos.x), -f64::from(pos.y)];
            let visit_idx = self.visit_order.iter().position(|&vid| vid == id);
            if self.start_node == Some(id) {
                start.push(pt);
            } else if visit_idx.is_some_and(|idx| {
                idx == self.animation_step.saturating_sub(1) && self.animation_step > 0
            }) {
                current.push(pt);
            } else if visit_idx.is_some_and(|idx| idx < self.animation_step) {
                visited.push(pt);
            } else {
                unvisited.push(pt);
            }
        }

        for (label, pts, color) in [
            ("unvisited_nodes", unvisited, palette.series_color(0)),
            ("visited_nodes", visited, palette.success),
            ("current_node", current, palette.error),
            ("start_node", start, palette.accent),
        ] {
            if !pts.is_empty() {
                primitives.push(ExportPrimitive::Points {
                    name: label.to_string(),
                    points: pts,
                    color,
                    radius: 8.0,
                });
            }
        }
    }

    pub(super) fn cold_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut datasets = Vec::new();
        let mut sorted_ids: Vec<usize> = self.node_positions.keys().copied().collect();
        sorted_ids.sort_unstable();
        if !sorted_ids.is_empty() {
            let nodes = sorted_ids
                .iter()
                .map(|id| {
                    let pos = self.node_positions[id];
                    [f64::from(pos.x), f64::from(pos.y)]
                })
                .collect();
            datasets.push(("graph_nodes".to_string(), nodes));
        }
        if !self.visit_order.is_empty() {
            let order = self
                .visit_order
                .iter()
                .enumerate()
                .map(|(step, &id)| [step as f64, id as f64])
                .collect();
            datasets.push(("traversal_visit_order".to_string(), order));
        }
        if !self.distances.is_empty() {
            let mut dists: Vec<[f64; 2]> = self
                .distances
                .iter()
                .map(|(&id, &d)| [id as f64, d])
                .collect();
            dists.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
            datasets.push(("shortest_path_distances".to_string(), dists));
        }
        datasets
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Dijkstra Shortest Path",
    "domain": "graph_theory",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "selected_algorithm": "Dijkstra",
    "start_node": 5,
    "playback_speed": 3.0
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "algorithm_visualizer_starter",
        title: "Dijkstra Shortest Path",
        description: "Priority-queue shortest path relaxation across weighted undirected graph topologies.",
        category: "Graph Algorithms",
        icon: "🛤️",
        domain: "graph_theory",
        tool_name: Some("AlgorithmVisualizerTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
