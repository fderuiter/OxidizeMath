//! Non-blocking `.dot`/`.graphml`/`.csv`/`.tsv` topology import and [`GraphTopologyDocument`] export for [`super::NetworkPropagationTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::NetworkPropagationTool;
use crate::accessibility::{announce_live_status, AccessibleHoverText};
use crate::app::TopologySource;
use crate::web_io::{spawn_open_file_dialog, OpenDialogConfig, PlatformInstant};
use eframe::egui;
use math_explorer::epidemiology::networks::{NetworkEpidemicModel, NodeState};
use math_explorer::pure_math::graph_theory::interchange::{
    GraphTopologyDocument, TopologyEdge, TopologyEdgeStyle, TopologyNode, TopologyNodeShape,
    DEFAULT_TOPOLOGY_LAYOUT_BOUNDS,
};
use petgraph::visit::EdgeRef;
use std::collections::HashSet;
use std::sync::mpsc::{self, Receiver, TryRecvError};

const OPEN_EPIDEMIC_TOPOLOGY_CONFIG: OpenDialogConfig = OpenDialogConfig {
    filter_name: "Contact Network (.dot, .graphml, .csv, .tsv)",
    extensions: &["dot", "gv", "graphml", "csv", "tsv"],
    accept: ".dot,.gv,.graphml,.csv,.tsv,text/vnd.graphviz,application/graphml+xml,application/xml,text/csv,text/tab-separated-values",
    read_error_prefix: "Failed to read contact network file",
};

type OpenTopologyRx = Receiver<Result<(String, String), String>>;

#[derive(Clone, Debug)]
pub(crate) struct ImportedEpidemicGraph {
    pub(crate) name: String,
    pub(crate) positions: Vec<[f32; 2]>,
    pub(crate) csr_row_ptr: Vec<usize>,
    pub(crate) csr_col_idx: Vec<usize>,
}

/// Async file-import channel and active custom contact-network state for [`NetworkPropagationTool`].
#[derive(Default)]
pub(crate) struct EpidemicTopologyUi {
    pub(crate) custom: Option<ImportedEpidemicGraph>,
    open_rx: Option<OpenTopologyRx>,
    pub(crate) last_status: Option<(String, bool, PlatformInstant)>,
}

fn normalize_epidemic_positions(doc: &GraphTopologyDocument) -> Vec<[f32; 2]> {
    let n = doc.nodes.len();
    if n == 0 {
        return Vec::new();
    }
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for node in &doc.nodes {
        if let Some([x, y]) = node.position {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    }
    let span_x = (max_x - min_x).max(1e-6);
    let span_y = (max_y - min_y).max(1e-6);
    doc.nodes
        .iter()
        .map(|node| {
            let [x, y] = node.position.unwrap_or([0.0, 0.0]);
            let nx = if max_x > min_x {
                -170.0 + 340.0 * ((x - min_x) / span_x)
            } else {
                0.0
            };
            let ny = if max_y > min_y {
                -170.0 + 340.0 * ((y - min_y) / span_y)
            } else {
                0.0
            };
            [nx as f32, ny as f32]
        })
        .collect()
}

fn build_csr_from_topology(
    doc: &GraphTopologyDocument,
) -> Result<(Vec<usize>, Vec<usize>), String> {
    let (g, indices) = doc.to_undirected_petgraph().map_err(|e| e.to_string())?;
    let n = indices.len();
    let mut adj = vec![Vec::new(); n];
    for edge in g.graph.edge_references() {
        let u = edge.source().index();
        let v = edge.target().index();
        if u < n && v < n && u != v {
            adj[u].push(v);
            adj[v].push(u);
        }
    }
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    row_ptr.push(0);
    for nbrs in &mut adj {
        nbrs.sort_unstable();
        nbrs.dedup();
        col_idx.extend_from_slice(nbrs);
        row_ptr.push(col_idx.len());
    }
    Ok((row_ptr, col_idx))
}

impl NetworkPropagationTool {
    pub(super) fn poll_topology_import(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.topology_ui.open_rx else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok((filename, text))) => {
                self.topology_ui.open_rx = None;
                match self.import_topology_text(&filename, &text) {
                    Ok(()) => {
                        let msg = format!(
                            "Imported {} ({} nodes, {} edges)",
                            filename,
                            self.model.num_nodes(),
                            self.model.edges().len()
                        );
                        announce_live_status(ctx, msg.clone());
                        self.topology_ui.last_status = Some((msg, false, PlatformInstant::now()));
                    }
                    Err(err) => {
                        let msg = format!("Topology import rejected ({filename}): {err}");
                        announce_live_status(ctx, msg.clone());
                        self.topology_ui.last_status = Some((msg, true, PlatformInstant::now()));
                    }
                }
            }
            Ok(Err(err)) => {
                self.topology_ui.open_rx = None;
                announce_live_status(ctx, err.clone());
                self.topology_ui.last_status = Some((err, true, PlatformInstant::now()));
            }
            Err(TryRecvError::Disconnected) => {
                self.topology_ui.open_rx = None;
            }
            Err(TryRecvError::Empty) => {}
        }
    }

    /// Parses and installs an external `.dot`/`.graphml`/`.csv`/`.tsv` contact topology.
    pub fn import_topology_text(&mut self, filename: &str, raw: &str) -> Result<(), String> {
        if self.is_challenge_locked() {
            return Err("Custom topology import is disabled while a challenge is active".into());
        }
        let mut doc =
            GraphTopologyDocument::parse_with_filename(filename, raw).map_err(|e| e.to_string())?;
        if doc.nodes.is_empty() {
            return Err("Imported topology must contain at least 1 vertex".into());
        }
        doc.ensure_positions(DEFAULT_TOPOLOGY_LAYOUT_BOUNDS);
        let (csr_row_ptr, csr_col_idx) = build_csr_from_topology(&doc)?;
        let positions = normalize_epidemic_positions(&doc);

        self.topology_source = TopologySource::LocalPreset;
        self.synced_bus_revision = None;
        self.upstream_linked = false;
        self.last_upstream_graph_rev = 0;
        self.vaccinated_nodes.clear();
        self.quarantined_edges.clear();
        self.topology_ui.custom = Some(ImportedEpidemicGraph {
            name: filename.to_string(),
            positions,
            csr_row_ptr,
            csr_col_idx,
        });
        self.rebuild_active_graph();
        Ok(())
    }

    /// Rebuilds the active custom contact graph (if present) or falls back to the seeded graph.
    pub(super) fn rebuild_active_graph(&mut self) {
        let Some(custom) = self.topology_ui.custom.clone() else {
            self.rebuild_seeded_graph();
            return;
        };
        let n = custom.positions.len();
        if n == 0 {
            self.topology_ui.custom = None;
            self.rebuild_seeded_graph();
            return;
        }
        let patient_zero = (0..n)
            .find(|&u| custom.csr_row_ptr[u] < custom.csr_row_ptr[u + 1])
            .unwrap_or(0);
        let mut states = vec![NodeState::Susceptible; n];
        states[patient_zero] = NodeState::Infected;
        let (beta, gamma) = (self.model.beta(), self.model.gamma());

        let Ok(mut built) = NetworkEpidemicModel::builder()
            .num_nodes(n)
            .parameters(beta, gamma)
            .states(states.clone())
            .positions(custom.positions.clone())
            .csr_adjacency(&custom.csr_row_ptr, &custom.csr_col_idx)
            .and_then(|b| b.build())
        else {
            return;
        };
        let Ok(built_baseline) = NetworkEpidemicModel::builder()
            .num_nodes(n)
            .parameters(beta, gamma)
            .states(states)
            .positions(custom.positions)
            .csr_adjacency(&custom.csr_row_ptr, &custom.csr_col_idx)
            .and_then(|b| b.build())
        else {
            return;
        };

        self.vaccinated_nodes
            .retain(|&v| built.set_node_vaccinated(v, true).is_ok());
        let edges = built.edges().to_vec();
        self.quarantined_edges.retain(|&idx| {
            edges
                .get(idx)
                .is_some_and(|&[u, v]| built.set_edge_quarantined(u, v, true).is_ok())
        });

        self.model = built;
        self.baseline_model = built_baseline;
        self.rng = oxidize_core::rng::OxidizeRng::new(self.seed);
        self.baseline_rng = oxidize_core::rng::OxidizeRng::new(self.seed);
        self.is_running = false;
        self.clamp_focus_indices();
        #[cfg(test)]
        {
            self.sim_steps = 0;
        }
        self.bump_revision();
    }

    /// Clears an imported custom contact topology and restores the deterministic seeded network.
    pub fn clear_custom_topology(&mut self) {
        self.topology_ui.custom = None;
        self.vaccinated_nodes.clear();
        self.quarantined_edges.clear();
        let _ = self.model.set_num_nodes(self.local_num_nodes);
        self.rebuild_seeded_graph();
    }

    pub(super) fn render_topology_import_controls(&mut self, ui: &mut egui::Ui) {
        let locked = self.is_challenge_locked();
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(!locked, egui::Button::new("📂 Import Topology"))
                .accessible_hover_text(
                    "Import an external .dot, .graphml, .csv, or .tsv contact network",
                )
                .clicked()
            {
                let (tx, rx) = mpsc::channel();
                self.topology_ui.open_rx = Some(rx);
                spawn_open_file_dialog(tx, OPEN_EPIDEMIC_TOPOLOGY_CONFIG, |path| {
                    path.file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("contact.dot")
                        .to_string()
                });
            }
            if self.topology_ui.custom.is_some()
                && ui
                    .add_enabled(!locked, egui::Button::new("↺ Seeded Graph"))
                    .accessible_hover_text(
                        "Restore the deterministic seeded geometric contact network",
                    )
                    .clicked()
            {
                self.clear_custom_topology();
                announce_live_status(ui.ctx(), "Restored seeded geometric contact network");
            }
        });
        if let Some(custom) = &self.topology_ui.custom {
            ui.small(format!(
                "Custom Network ({}): |V| = {}, |E| = {}, R0(het) = {:.3}",
                custom.name,
                self.model.num_nodes(),
                self.model.edges().len(),
                self.live_heterogeneous_r0()
            ));
        }
        if let Some((msg, is_err, stamp)) = &self.topology_ui.last_status {
            if stamp.elapsed().as_secs_f32() <= 12.0 {
                let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
                let col = if *is_err {
                    palette.error
                } else {
                    palette.success
                };
                ui.colored_label(col, egui::RichText::new(msg).small());
            }
        }
    }

    pub(super) fn build_epidemic_topology(&self) -> Result<GraphTopologyDocument, String> {
        let mut doc = GraphTopologyDocument::new("EpidemicContactNetwork", false);
        doc.attributes
            .insert("beta".to_string(), format!("{:.4}", self.model.beta()));
        doc.attributes
            .insert("gamma".to_string(), format!("{:.4}", self.model.gamma()));
        doc.attributes.insert(
            "heterogeneous_r0".to_string(),
            format!("{:.4}", self.live_heterogeneous_r0()),
        );
        doc.attributes
            .insert("budget".to_string(), self.budget.to_string());

        let states = self.model.states();
        let positions = self.model.positions();
        for (idx, &st) in states.iter().enumerate() {
            let [x, y] = positions.get(idx).copied().unwrap_or([0.0, 0.0]);
            let (state_str, hex, shape) = match st {
                NodeState::Susceptible => ("susceptible", "#3B82F6", TopologyNodeShape::Circle),
                NodeState::Infected => ("infected", "#EF4444", TopologyNodeShape::DoubleCircle),
                NodeState::Recovered => ("recovered", "#10B981", TopologyNodeShape::Circle),
                NodeState::Vaccinated => ("vaccinated", "#F59E0B", TopologyNodeShape::Box),
            };
            doc.nodes.push(
                TopologyNode::new(idx.to_string(), format!("{idx}:{state_str}"))
                    .with_position(f64::from(x) + 300.0, f64::from(y) + 220.0)
                    .with_shape(shape)
                    .with_color(hex)
                    .with_attribute("compartment", state_str),
            );
        }

        let quar_set: HashSet<usize> = self.quarantined_edges.iter().copied().collect();
        for (edge_idx, &[u, v]) in self.model.edges().iter().enumerate() {
            let quarantined = quar_set.contains(&edge_idx);
            let mut edge = TopologyEdge::new(u.to_string(), v.to_string(), 1.0)
                .with_attribute("quarantined", quarantined.to_string());
            if quarantined {
                edge.style = TopologyEdgeStyle::Dashed;
                edge.color_hex = Some("#9CA3AF".to_string());
                edge.label = Some("quarantined".to_string());
            }
            doc.edges.push(edge);
        }

        doc.validate().map_err(|e| e.to_string())?;
        Ok(doc)
    }
}

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::*;
    use crate::framework::InteractiveTool;
    use math_explorer::pure_math::graph_theory::interchange::{
        TopologyFormat, MAX_TOPOLOGY_IMPORT_BYTES,
    };

    #[test]
    fn test_epidemic_import_dot_graphml_tsv_and_export_roundtrip() {
        let mut tool = NetworkPropagationTool::default();
        tool.import_topology_text(
            "star.dot",
            "\u{feff}graph Star { 0 -- 1; 0 -- 2; 0 -- 3; 0 -- 4; }",
        )
        .expect("import star.dot");
        assert_eq!(tool.model.num_nodes(), 5);
        assert_eq!(tool.model.edges().len(), 4);
        assert!(tool.has_finite_parameters());

        tool.toggle_node_vaccination(1).expect("vaccinate node 1");
        let doc = tool.export_topology_document().expect("export topology");
        let graphml = doc.serialize(TopologyFormat::GraphMl).expect("graphml");
        assert!(graphml.contains("vaccinated"));

        tool.clear_interventions();
        assert_eq!(tool.model.num_nodes(), 5);
        assert!(tool.vaccinated_nodes.is_empty());

        tool.import_topology_text("chain.tsv", "source\ttarget\tweight\na\tb\t1\nb\tc\t1\n")
            .expect("import chain.tsv");
        assert_eq!(tool.model.num_nodes(), 3);

        tool.clear_custom_topology();
        assert_eq!(tool.model.num_nodes(), 50);
    }

    #[test]
    fn test_epidemic_import_rejects_xxe_oversized_and_challenge_lock() {
        let mut tool = NetworkPropagationTool::default();
        let xxe = "<!DOCTYPE foo [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]><graphml/>";
        assert!(tool.import_topology_text("evil.graphml", xxe).is_err());

        let huge = "a".repeat(MAX_TOPOLOGY_IMPORT_BYTES + 16);
        assert!(tool.import_topology_text("huge.dot", &huge).is_err());

        tool.import_topology_text("star.dot", "graph S { 0 -- 1; 0 -- 2; }")
            .expect("custom import before challenge");
        assert!(tool.topology_ui.custom.is_some());
        tool.apply_challenge(Some(super::super::challenge::CHALLENGE_EPI_HUB_VACCINATION))
            .expect("activate challenge");
        assert!(tool.topology_ui.custom.is_none());
        assert_eq!(tool.model.num_nodes(), 72);
        assert!(tool
            .import_topology_text("star.dot", "graph S { 0 -- 1; }")
            .is_err());
    }
}
