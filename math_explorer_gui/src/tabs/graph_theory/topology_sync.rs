//! Client-side `.dot`, `.graphml`, and `.csv`/`.tsv` topology file import and live cross-tool
//! synchronization across `GraphEditorTool`, `AlgorithmVisualizerTool`, and `NetworkMetricsTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::{announce_status_with_priority, AccessibleHoverText};
use crate::app::SemanticPalette;
pub use crate::framework::export_topology::TOPOLOGY_FILE_DIALOG_CONFIG;
use crate::framework::{scoped_panel_id, SimulationFramework};
use crate::web_io::{sanitize_download_filename, spawn_open_file_dialog, strip_utf8_bom};
use eframe::egui::{self, Pos2};
use math_explorer::pure_math::graph_theory::graph_io::{
    GraphTopologyDocument, DEFAULT_TOPOLOGY_LAYOUT_BOUNDS, MAX_TOPOLOGY_IMPORT_BYTES,
};
use std::collections::BTreeMap;
use std::sync::mpsc::{self, Receiver};

/// Parses a bounded topology string with UTF-8 BOM stripping and automatic layout fallback.
pub fn parse_bounded_topology_file(
    filename: &str,
    raw_content: &str,
) -> Result<GraphTopologyDocument, String> {
    let stripped = strip_utf8_bom(raw_content);
    if stripped.len() > MAX_TOPOLOGY_IMPORT_BYTES {
        return Err(format!(
            "Topology file {filename} ({:.2} MiB) exceeds maximum allowed size of 2.0 MiB.",
            stripped.len() as f64 / (1024.0 * 1024.0)
        ));
    }
    let mut doc = GraphTopologyDocument::parse_with_filename(filename, stripped)
        .map_err(|e| format!("Failed to import {filename}: {e}"))?;
    normalize_document_canvas_bounds(&mut doc, (80.0, 80.0, 560.0, 420.0));
    Ok(doc)
}

/// Rescales positioned vertices into `target_bounds` when coordinates fall outside canvas viewport range.
pub fn normalize_document_canvas_bounds(
    doc: &mut GraphTopologyDocument,
    target_bounds: (f32, f32, f32, f32),
) {
    doc.ensure_positions(DEFAULT_TOPOLOGY_LAYOUT_BOUNDS);
    if doc.nodes.len() <= 1 {
        if let Some(first) = doc.nodes.first_mut() {
            let cx = f64::from(target_bounds.0 + target_bounds.2) * 0.5;
            let cy = f64::from(target_bounds.1 + target_bounds.3) * 0.5;
            first.position = Some([cx, cy]);
        }
        return;
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
    let fits_viewport = min_x >= 20.0 && max_x <= 720.0 && min_y >= 20.0 && max_y <= 540.0;
    let span_x = max_x - min_x;
    let span_y = max_y - min_y;
    if fits_viewport && (span_x >= 40.0 || span_y >= 40.0) {
        return;
    }
    let sx = span_x.max(1.0);
    let sy = span_y.max(1.0);
    let w = f64::from(target_bounds.2 - target_bounds.0);
    let h = f64::from(target_bounds.3 - target_bounds.1);
    for node in &mut doc.nodes {
        let [x, y] = node.position.unwrap_or([min_x, min_y]);
        node.position = Some([
            f64::from(target_bounds.0) + ((x - min_x) / sx) * w,
            f64::from(target_bounds.1) + ((y - min_y) / sy) * h,
        ]);
    }
}

/// Indexed canvas graph representation `(labels, positions, weighted_edges)`.
pub type IndexedCanvasGraph = (
    BTreeMap<usize, String>,
    BTreeMap<usize, Pos2>,
    Vec<(usize, usize, f64)>,
);

/// Maps a [`GraphTopologyDocument`] into integer node IDs, labels, `Pos2` coordinates, and weighted edges.
pub fn extract_indexed_canvas_graph(doc: &GraphTopologyDocument) -> IndexedCanvasGraph {
    let all_small_uint = !doc.nodes.is_empty()
        && doc
            .nodes
            .iter()
            .all(|n| n.id.parse::<usize>().is_ok_and(|id| id < 10_000));
    let mut str_to_id = std::collections::HashMap::with_capacity(doc.nodes.len());
    let mut labels = BTreeMap::new();
    let mut positions = BTreeMap::new();
    for (seq_idx, node) in doc.nodes.iter().enumerate() {
        let num_id = if all_small_uint {
            node.id.parse::<usize>().unwrap_or(seq_idx)
        } else {
            seq_idx
        };
        str_to_id.insert(node.id.as_str(), num_id);
        labels.insert(num_id, node.label.clone());
        let [x, y] = node.position.unwrap_or([300.0, 220.0]);
        positions.insert(num_id, Pos2::new(x as f32, y as f32));
    }
    let mut edges = Vec::with_capacity(doc.edges.len());
    for edge in &doc.edges {
        if let (Some(&u), Some(&v)) = (
            str_to_id.get(edge.source.as_str()),
            str_to_id.get(edge.target.as_str()),
        ) {
            if u != v {
                edges.push((u, v, edge.weight));
            }
        }
    }
    (labels, positions, edges)
}

/// Coordinates topology imports and live cross-tool synchronization inside `GraphTheoryTab`.
#[derive(Default)]
pub struct GraphTopologySyncHub {
    shared_doc: Option<GraphTopologyDocument>,
    sync_epoch: u64,
    applied_target: Option<(&'static str, u64)>,
    last_editor_rev: Option<u64>,
    rx_import: Option<Receiver<Result<(String, String), String>>>,
    banner_status: Option<(String, bool)>,
}

impl GraphTopologySyncHub {
    /// Returns the currently synchronized [`GraphTopologyDocument`], if any.
    #[cfg(test)]
    #[must_use]
    pub fn shared_document(&self) -> Option<&GraphTopologyDocument> {
        self.shared_doc.as_ref()
    }

    /// Parses and applies a `.dot`, `.graphml`, or `.csv`/`.tsv` topology across `GraphTheoryTab`.
    pub fn import_topology_into_framework(
        &mut self,
        framework: &mut SimulationFramework,
        filename: &str,
        content: &str,
    ) -> Result<(), String> {
        let safe_name = sanitize_download_filename(filename, "topology.dot");
        match parse_bounded_topology_file(&safe_name, content) {
            Ok(doc) => {
                let n = doc.nodes.len();
                let m = doc.edges.len();
                self.sync_epoch = self.sync_epoch.wrapping_add(1).max(1);
                self.shared_doc = Some(doc.clone());
                self.applied_target = None;
                if let Some(tool) = framework.active_tool.as_mut() {
                    let _ = tool.apply_topology_document(&doc);
                    self.applied_target = Some((tool.name(), self.sync_epoch));
                    if tool.name() == "Graph Editor" {
                        self.last_editor_rev = Some(tool.state_revision());
                    }
                }
                let msg = format!("Imported {n} nodes and {m} edges from {safe_name}.");
                announce_status_with_priority(&msg, "polite");
                self.banner_status = Some((msg, false));
                Ok(())
            }
            Err(err) => {
                announce_status_with_priority(&err, "assertive");
                self.banner_status = Some((err.clone(), true));
                Err(err)
            }
        }
    }

    /// Renders the `GraphTheoryTab` topology import and live-sync status toolbar.
    pub fn show_sync_bar(&mut self, ui: &mut egui::Ui, framework: &mut SimulationFramework) {
        self.poll_import_channel(ui.ctx(), framework);
        let bar_id = scoped_panel_id(ui, "graph_theory_topology_sync_bar");
        egui::TopBottomPanel::top(bar_id).show_inside(ui, |ui| {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            ui.horizontal_wrapped(|ui| {
                if ui
                    .button("📂 Import Topology (.dot / .graphml / .csv)")
                    .accessible_hover_text(
                        "Import a local Graphviz DOT, GraphML, CSV, or TSV topology into Graph Theory tools",
                    )
                    .clicked()
                {
                    let (tx, rx) = mpsc::channel();
                    self.rx_import = Some(rx);
                    spawn_open_file_dialog(tx, TOPOLOGY_FILE_DIALOG_CONFIG, |path| {
                        path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("topology.dot")
                            .to_string()
                    });
                }
                if let Some(doc) = &self.shared_doc {
                    ui.colored_label(
                        palette.info,
                        format!(
                            "🔗 Synced: {} ({} nodes, {} edges)",
                            doc.name,
                            doc.nodes.len(),
                            doc.edges.len()
                        ),
                    );
                    if ui
                        .small_button("↺ Reset Default Graph")
                        .accessible_hover_text("Clear shared topology and restore tool default sample graph")
                        .clicked()
                    {
                        self.clear_shared_topology(framework);
                    }
                }
                if let Some((msg, is_err)) = &self.banner_status {
                    let color = if *is_err { palette.error } else { palette.success };
                    ui.colored_label(color, msg);
                }
            });
        });
    }

    fn clear_shared_topology(&mut self, framework: &mut SimulationFramework) {
        self.shared_doc = None;
        self.sync_epoch = self.sync_epoch.wrapping_add(1).max(1);
        self.applied_target = None;
        self.last_editor_rev = None;
        self.banner_status = None;
        if let Some(tool) = framework.active_tool.as_mut() {
            tool.reset_default_topology();
        }
        announce_status_with_priority("Restored default sample graph.", "polite");
    }

    fn poll_import_channel(&mut self, ctx: &egui::Context, framework: &mut SimulationFramework) {
        let outcome = self.rx_import.as_ref().and_then(|rx| rx.try_recv().ok());
        let Some(res) = outcome else {
            return;
        };
        self.rx_import = None;
        match res {
            Ok((filename, content)) => {
                let _ = self.import_topology_into_framework(framework, &filename, &content);
            }
            Err(err) => {
                announce_status_with_priority(&err, "assertive");
                self.banner_status = Some((err, true));
            }
        }
        ctx.request_repaint();
    }

    /// Propagates pending shared topologies into newly selected tools before rendering.
    pub fn sync_before_frame(&mut self, framework: &mut SimulationFramework) {
        let Some(tool) = framework.active_tool.as_mut() else {
            return;
        };
        let tool_name = tool.name();
        let Some(doc) = &self.shared_doc else {
            return;
        };
        if self.applied_target == Some((tool_name, self.sync_epoch)) {
            return;
        }
        if tool.apply_topology_document(doc).is_ok() {
            self.applied_target = Some((tool_name, self.sync_epoch));
            if tool_name == "Graph Editor" {
                self.last_editor_rev = Some(tool.state_revision());
            }
        }
    }

    /// Captures interactive `GraphEditorTool` edits (including undo/redo) into the shared topology hub.
    pub fn sync_after_frame(&mut self, framework: &SimulationFramework) {
        let Some(tool) = framework.active_tool.as_ref() else {
            return;
        };
        if tool.name() != "Graph Editor" {
            return;
        }
        let rev = tool.state_revision();
        if self.last_editor_rev == Some(rev) {
            return;
        }
        self.last_editor_rev = Some(rev);
        if let Some(Ok(doc)) = tool.export_topology_document() {
            if doc.nodes.is_empty() {
                self.shared_doc = None;
                self.applied_target = None;
            } else {
                self.sync_epoch = self.sync_epoch.wrapping_add(1).max(1);
                self.shared_doc = Some(doc);
                self.applied_target = Some(("Graph Editor", self.sync_epoch));
            }
        }
    }
}
