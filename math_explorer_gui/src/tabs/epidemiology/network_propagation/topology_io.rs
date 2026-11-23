//! Client-side `.dot`, `.graphml`, and `.csv`/`.tsv` contact network import and topology export
//! for [`super::NetworkPropagationTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::analytics::compute_network_analytics;
use super::NetworkPropagationTool;
use crate::accessibility::{announce_status_with_priority, AccessibleHoverText};
use crate::app::SemanticPalette;
use crate::framework::export_topology::TOPOLOGY_FILE_DIALOG_CONFIG;
use crate::web_io::{sanitize_download_filename, spawn_open_file_dialog, strip_utf8_bom};
use eframe::egui;
use math_explorer::epidemiology::networks::{NetworkEpidemicModel, NodeState};
use math_explorer::pure_math::graph_theory::graph_io::{
    GraphTopologyDocument, TopologyEdge, TopologyEdgeStyle, TopologyNode, TopologyNodeShape,
    DEFAULT_TOPOLOGY_LAYOUT_BOUNDS, MAX_TOPOLOGY_IMPORT_BYTES, MAX_TOPOLOGY_NODES,
};
use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver};

#[derive(Clone)]
pub(super) struct CustomContactTopology {
    pub(super) name: String,
    pub(super) labels: Vec<String>,
    pub(super) positions: Vec<[f32; 2]>,
    pub(super) adjacency: Vec<Vec<usize>>,
}

#[derive(Default)]
pub(super) struct NetPropTopologyUi {
    pub(super) custom: Option<CustomContactTopology>,
    rx_import: Option<Receiver<Result<(String, String), String>>>,
    banner: Option<(String, bool)>,
}

impl NetworkPropagationTool {
    /// Parses and installs an empirical `.dot`, `.graphml`, or `.csv`/`.tsv` contact topology.
    pub fn import_contact_topology(&mut self, filename: &str, content: &str) -> Result<(), String> {
        let safe_name = sanitize_download_filename(filename, "contact_network.dot");
        let stripped = strip_utf8_bom(content);
        if stripped.len() > MAX_TOPOLOGY_IMPORT_BYTES {
            let err =
                format!("Contact topology {safe_name} exceeds maximum allowed size of 2.0 MiB.");
            announce_status_with_priority(&err, "assertive");
            self.topology_ui.banner = Some((err.clone(), true));
            return Err(err);
        }
        let doc = match GraphTopologyDocument::parse_with_filename(&safe_name, stripped) {
            Ok(mut parsed) => {
                parsed.name = safe_name.clone();
                parsed
            }
            Err(e) => {
                let err = format!("Failed to import {safe_name}: {e}");
                announce_status_with_priority(&err, "assertive");
                self.topology_ui.banner = Some((err.clone(), true));
                return Err(err);
            }
        };
        match self.apply_contact_document(&doc) {
            Ok(()) => {
                let msg = format!(
                    "Imported contact network {safe_name} ({} nodes, {} edges).",
                    self.model.num_nodes(),
                    self.model.edges().len()
                );
                announce_status_with_priority(&msg, "polite");
                self.topology_ui.banner = Some((msg, false));
                Ok(())
            }
            Err(err) => {
                announce_status_with_priority(&err, "assertive");
                self.topology_ui.banner = Some((err.clone(), true));
                Err(err)
            }
        }
    }

    pub(super) fn apply_contact_document(
        &mut self,
        doc: &GraphTopologyDocument,
    ) -> Result<(), String> {
        let n = doc.nodes.len();
        if n == 0 || n > MAX_TOPOLOGY_NODES {
            return Err(format!(
                "Contact network must contain between 1 and {MAX_TOPOLOGY_NODES} nodes (found {n})."
            ));
        }
        let mut positioned = doc.clone();
        positioned.ensure_positions(DEFAULT_TOPOLOGY_LAYOUT_BOUNDS);
        let positions = normalize_centered_positions(&positioned);
        let adjacency = build_undirected_adjacency(&positioned);
        let labels = positioned.nodes.iter().map(|nd| nd.label.clone()).collect();
        self.topology_ui.custom = Some(CustomContactTopology {
            name: positioned.name,
            labels,
            positions,
            adjacency,
        });
        self.vaccinated_nodes.clear();
        self.quarantined_edges.clear();
        self.rebuild_active_graph();
        Ok(())
    }

    pub(super) fn rebuild_active_graph(&mut self) {
        let Some(custom) = self.topology_ui.custom.clone() else {
            self.rebuild_seeded_graph();
            return;
        };
        let n = custom.positions.len();
        let mut states = vec![NodeState::Susceptible; n];
        states[0] = NodeState::Infected;
        let Ok(mut rebuilt) = NetworkEpidemicModel::builder()
            .num_nodes(n)
            .parameters(self.model.beta(), self.model.gamma())
            .positions(custom.positions)
            .adjacency(custom.adjacency)
            .states(states)
            .build()
        else {
            return;
        };
        self.vaccinated_nodes
            .retain(|&node| rebuilt.set_node_vaccinated(node, true).is_ok());
        let edges = rebuilt.edges().to_vec();
        self.quarantined_edges.retain(|&edge_idx| {
            edges
                .get(edge_idx)
                .is_some_and(|&[u, v]| rebuilt.set_edge_quarantined(u, v, true).is_ok())
        });
        self.model = rebuilt;
        self.is_running = false;
        self.clamp_focus_indices();
        #[cfg(test)]
        {
            self.sim_steps = 0;
        }
        self.bump_revision();
    }

    pub(super) fn reset_to_seeded_graph(&mut self) {
        self.topology_ui.custom = None;
        self.topology_ui.banner = None;
        self.vaccinated_nodes.clear();
        self.quarantined_edges.clear();
        if !(10..=200).contains(&self.model.num_nodes()) {
            let _ = self.model.set_num_nodes(50);
        }
        self.rebuild_seeded_graph();
    }

    pub(super) fn render_topology_import_controls(&mut self, ui: &mut egui::Ui) {
        self.poll_topology_import_channel(ui.ctx());
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("📂 Import Contact Network")
                .accessible_hover_text(
                    "Import a .dot, .graphml, or .csv/.tsv contact graph for containment analysis",
                )
                .clicked()
            {
                let (tx, rx) = mpsc::channel();
                self.topology_ui.rx_import = Some(rx);
                spawn_open_file_dialog(tx, TOPOLOGY_FILE_DIALOG_CONFIG, |path| {
                    path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("contact_network.dot")
                        .to_string()
                });
            }
            if self.topology_ui.custom.is_some()
                && ui
                    .small_button("↺ Seeded Graph")
                    .accessible_hover_text("Restore procedural geometric contact network")
                    .clicked()
            {
                self.reset_to_seeded_graph();
                announce_status_with_priority("Restored seeded contact network.", "polite");
            }
        });
        if let Some((msg, is_err)) = &self.topology_ui.banner {
            let color = if *is_err { palette.error } else { palette.info };
            ui.small(egui::RichText::new(msg).color(color));
        }
    }

    fn poll_topology_import_channel(&mut self, ctx: &egui::Context) {
        let received = self
            .topology_ui
            .rx_import
            .as_ref()
            .and_then(|rx| rx.try_recv().ok());
        let Some(res) = received else {
            return;
        };
        self.topology_ui.rx_import = None;
        match res {
            Ok((filename, content)) => {
                let _ = self.import_contact_topology(&filename, &content);
            }
            Err(err) => {
                announce_status_with_priority(&err, "assertive");
                self.topology_ui.banner = Some((err, true));
            }
        }
        ctx.request_repaint();
    }

    pub(super) fn build_export_topology(&self) -> GraphTopologyDocument {
        let name = self
            .topology_ui
            .custom
            .as_ref()
            .map_or_else(|| "NetworkPropagation".to_string(), |c| c.name.clone());
        let mut doc = GraphTopologyDocument::new(name, false);
        let summary = compute_network_analytics(&self.model);
        doc.attributes
            .insert("beta".into(), format!("{:.4}", self.model.beta()));
        doc.attributes
            .insert("gamma".into(), format!("{:.4}", self.model.gamma()));
        doc.attributes.insert(
            "effective_r0".into(),
            format!("{:.4}", summary.effective_r0),
        );
        doc.attributes.insert(
            "extinction_probability".into(),
            format!("{:.4}", summary.extinction_probability),
        );
        doc.attributes.insert(
            "budget".into(),
            format!("{}/{}", self.used_budget(), self.budget),
        );
        self.populate_export_nodes_and_edges(&mut doc);
        doc
    }

    fn populate_export_nodes_and_edges(&self, doc: &mut GraphTopologyDocument) {
        for i in 0..self.model.num_nodes() {
            let p = self.model.positions()[i];
            let st = self.model.states()[i];
            let label = self
                .topology_ui
                .custom
                .as_ref()
                .and_then(|c| c.labels.get(i).cloned())
                .unwrap_or_else(|| format!("N{i}"));
            let (st_str, color) = match st {
                NodeState::Susceptible => ("Susceptible", "#2563EB"),
                NodeState::Infected => ("Infected", "#DC2626"),
                NodeState::Recovered => ("Recovered", "#6B7280"),
                NodeState::Vaccinated => ("Vaccinated", "#16A34A"),
            };
            let mut node = TopologyNode::new(i.to_string(), label)
                .with_position(300.0 + f64::from(p[0]), 240.0 + f64::from(p[1]))
                .with_color(color)
                .with_attribute("state", st_str)
                .with_attribute("active_degree", self.model.active_degree(i).to_string());
            if st == NodeState::Vaccinated || self.model.patient_zero() == Some(i) {
                node = node.with_shape(TopologyNodeShape::DoubleCircle);
            }
            doc.nodes.push(node);
        }
        for &[u, v] in self.model.edges() {
            let quarantined = self.model.is_edge_quarantined(u, v);
            let mut edge = TopologyEdge::new(u.to_string(), v.to_string(), 1.0)
                .with_attribute("quarantined", if quarantined { "true" } else { "false" });
            if quarantined {
                edge = edge
                    .with_style(TopologyEdgeStyle::Dashed)
                    .with_color("#D97706")
                    .with_label("quarantined");
            }
            doc.edges.push(edge);
        }
    }
}

fn normalize_centered_positions(doc: &GraphTopologyDocument) -> Vec<[f32; 2]> {
    if doc.nodes.len() == 1 {
        return vec![[0.0, 0.0]];
    }
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for nd in &doc.nodes {
        let [x, y] = nd.position.unwrap_or([0.0, 0.0]);
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let sx = (max_x - min_x).max(1e-6);
    let sy = (max_y - min_y).max(1e-6);
    doc.nodes
        .iter()
        .map(|nd| {
            let [x, y] = nd.position.unwrap_or([min_x, min_y]);
            let nx = (((x - min_x) / sx) - 0.5) * 330.0;
            let ny = (((y - min_y) / sy) - 0.5) * 330.0;
            [nx as f32, ny as f32]
        })
        .collect()
}

fn build_undirected_adjacency(doc: &GraphTopologyDocument) -> Vec<Vec<usize>> {
    let mut id_map = HashMap::with_capacity(doc.nodes.len());
    for (idx, nd) in doc.nodes.iter().enumerate() {
        id_map.insert(nd.id.as_str(), idx);
    }
    let mut adjacency = vec![Vec::new(); doc.nodes.len()];
    for edge in &doc.edges {
        if let (Some(&u), Some(&v)) = (
            id_map.get(edge.source.as_str()),
            id_map.get(edge.target.as_str()),
        ) {
            if u != v {
                adjacency[u].push(v);
                adjacency[v].push(u);
            }
        }
    }
    for nbrs in &mut adjacency {
        nbrs.sort_unstable();
        nbrs.dedup();
    }
    adjacency
}
