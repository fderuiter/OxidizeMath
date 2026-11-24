//! Offscreen plot exports, styled `.dot`/`.graphml`/`.csv` serialization, and topology handoff for `GraphEditorTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{AffineLatency, BraessCongestionConfig, GraphEditorTool, SerializedGraphState};
use crate::accessibility::AccessibleHoverText;
use crate::tabs::graph_theory::{
    io_helpers::{color32_to_hex, render_handoff_buttons},
    plot_export::{
        EdgePlotSpec, NodePlotGroup, build_network_datasets, build_network_plot_primitives,
    },
    shared_topology::{
        SharedGraphTopology, TopologyHandoffTarget, btree_pos2_to_tuples, take_pending_handoff_for,
    },
};
use eframe::egui::{self, Pos2};
use egui_plot::{ExportPrimitive, SharedPoints};
use math_explorer::pure_math::graph_theory::interchange::{
    DotEdgeStyle, DotNodeStyle, DotStyleOverlay, to_edge_csv, to_graphml, to_styled_dot,
};

/// Serializes a [`SharedGraphTopology`] into a `SerializedGraphState` JSON string for `GraphEditorTool`.
#[must_use]
pub fn editor_state_json_from_shared(topo: &SharedGraphTopology) -> Option<String> {
    let next_node_id = topo
        .nodes
        .keys()
        .next_back()
        .map_or(0, |max_id| max_id.saturating_add(1));
    let mut congestion = BraessCongestionConfig::default();
    if topo.directed {
        congestion.enabled = true;
        if let Some(s) = topo.source_node {
            congestion.source_node = s;
        }
        if let Some(t) = topo.sink_node {
            congestion.sink_node = t;
        }
    }
    let state = SerializedGraphState {
        node_positions: topo.nodes.clone(),
        edges: topo.edges.clone(),
        selected_node: None,
        next_node_id,
        edge_latencies: topo.edge_latencies.clone(),
        congestion,
    };
    serde_json::to_string(&state).ok()
}

impl GraphEditorTool {
    /// Exports the current editor graph (and optional Braess terminals/latencies) as a [`SharedGraphTopology`].
    #[must_use]
    pub fn export_shared_topology(&self) -> SharedGraphTopology {
        let directed = self.directed_edges || self.braess.config.enabled;
        let source_node = if directed
            && self
                .node_positions
                .contains_key(&self.braess.config.source_node)
        {
            Some(self.braess.config.source_node)
        } else {
            self.node_positions.keys().next().copied()
        };
        let sink_node = if directed
            && self
                .node_positions
                .contains_key(&self.braess.config.sink_node)
        {
            Some(self.braess.config.sink_node)
        } else {
            self.node_positions
                .keys()
                .next_back()
                .copied()
                .filter(|&t| Some(t) != source_node)
        };
        SharedGraphTopology {
            nodes: btree_pos2_to_tuples(&self.node_positions),
            edges: self.edges.clone(),
            directed,
            source_node,
            sink_node,
            edge_latencies: if self.braess.edge_latencies.len() == self.edges.len() {
                self.braess.edge_latencies.clone()
            } else {
                Vec::new()
            },
            revision: self.compute_state_revision(),
        }
    }

    /// Replaces the active editor graph with `topo`, recording an undo snapshot if previously non-empty.
    pub fn apply_shared_topology(&mut self, topo: &SharedGraphTopology) {
        if !self.node_positions.is_empty() || !self.edges.is_empty() {
            self.push_undo_snapshot();
        } else {
            self.bump_revision();
        }
        self.node_positions = topo
            .nodes
            .iter()
            .map(|(&id, &(x, y))| (id, Pos2::new(x, y)))
            .collect();
        self.edges = topo.edges.clone();
        self.selected_node = None;
        self.next_node_id = self
            .node_positions
            .keys()
            .next_back()
            .map_or(0, |max_id| max_id.saturating_add(1));
        self.braess.edge_latencies = topo
            .edge_latencies
            .iter()
            .map(|l| AffineLatency::clamped(l.a, l.b))
            .collect();
        self.braess.sync_with_edges(&self.edges);
        self.directed_edges = topo.directed;
        self.braess.config.enabled = topo.directed;
        if let Some(s) = topo.source_node {
            self.braess.config.source_node = s;
        }
        if let Some(t) = topo.sink_node {
            self.braess.config.sink_node = t;
        }
        self.rebuild_graph();
        if self.braess.config.enabled {
            self.recompute_braess();
        }
    }

    /// Serializes the active editor graph (including Wardrop congestion styling when active) as `.dot`.
    pub fn to_styled_dot_string(&self) -> Result<String, String> {
        let topo = self.export_shared_topology();
        let interchange = topo.to_interchange_graph()?;
        let mut overlay = DotStyleOverlay::default();
        if self.braess.config.enabled {
            self.populate_braess_dot_overlay(&mut overlay);
        }
        to_styled_dot(&interchange, &overlay).map_err(|e| e.to_string())
    }

    fn populate_braess_dot_overlay(&self, overlay: &mut DotStyleOverlay) {
        let palette = crate::app::SemanticPalette::for_theme(crate::app::ThemeMode::Dark);
        if let Ok(summary) = self.braess.status {
            overlay.graph_label = Some(format!(
                "Wardrop Latency: {:.2} (Demand D = {:.2})",
                summary.commuter_latency, self.braess.config.total_demand
            ));
        }
        for &id in self.node_positions.keys() {
            if id == self.braess.config.source_node || id == self.braess.config.sink_node {
                let role = if id == self.braess.config.source_node {
                    "s"
                } else {
                    "t"
                };
                overlay.node_styles.insert(
                    id,
                    DotNodeStyle {
                        color: Some(color32_to_hex(palette.accent)),
                        fillcolor: Some(color32_to_hex(palette.series_color(0))),
                        penwidth: Some(2.5),
                        xlabel: Some(role.to_string()),
                    },
                );
            }
        }
        for idx in 0..self.edges.len() {
            overlay
                .edge_styles
                .insert(idx, self.braess_dot_edge_style(idx, &palette));
        }
    }

    fn braess_dot_edge_style(
        &self,
        idx: usize,
        palette: &crate::app::SemanticPalette,
    ) -> DotEdgeStyle {
        let flow = self
            .braess
            .equilibrium_flows
            .get(idx)
            .copied()
            .unwrap_or(0.0);
        let lat = self
            .braess
            .equilibrium_latencies
            .get(idx)
            .copied()
            .unwrap_or(self.edges[idx].2);
        let active = flow > 1e-3;
        let col = if active {
            palette.series_color(3)
        } else {
            palette.graph_edge
        };
        DotEdgeStyle {
            color: Some(color32_to_hex(col)),
            penwidth: Some(if active {
                2.0 + 3.0 * flow.min(2.0)
            } else {
                1.5
            }),
            xlabel: Some(format!("f={flow:.2}, l={lat:.2}")),
        }
    }

    /// Serializes the active editor graph as `.graphml`.
    pub fn to_graphml_string(&self) -> Result<String, String> {
        let interchange = self.export_shared_topology().to_interchange_graph()?;
        to_graphml(&interchange).map_err(|e| e.to_string())
    }

    /// Serializes the active editor graph as `.csv` edge list.
    pub fn to_csv_string(&self) -> Result<String, String> {
        let interchange = self.export_shared_topology().to_interchange_graph()?;
        to_edge_csv(&interchange).map_err(|e| e.to_string())
    }

    pub(super) fn render_interchange_and_handoff_ui(&mut self, ui: &mut egui::Ui) {
        if let Some(pending) =
            take_pending_handoff_for(ui.ctx(), TopologyHandoffTarget::GraphEditor)
        {
            self.apply_shared_topology(&pending);
        }
        if let Some(imported) = self.file_io.poll(ui.ctx()) {
            self.apply_shared_topology(&imported);
        }
        let non_empty = !self.node_positions.is_empty();
        ui.label("Topology Handoff & File I/O:");
        let current_snapshot = self.export_shared_topology();
        if let Some(shared) =
            render_handoff_buttons(ui, TopologyHandoffTarget::GraphEditor, non_empty, || {
                current_snapshot
            })
        {
            self.apply_shared_topology(&shared);
        }
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("📂 Import")
                .accessible_hover_text("Import a .dot, .graphml, .csv, or .json graph topology")
                .clicked()
            {
                self.file_io.trigger_import_dialog();
            }
            if ui
                .add_enabled(non_empty, egui::Button::new("⬇ .dot"))
                .accessible_hover_text("Download styled Graphviz .dot topology")
                .clicked()
            {
                let content = self.to_styled_dot_string();
                self.file_io.trigger_save_dot("graph_editor.dot", content);
            }
            if ui
                .add_enabled(non_empty, egui::Button::new("⬇ .graphml"))
                .accessible_hover_text("Download GraphML XML topology")
                .clicked()
            {
                let content = self.to_graphml_string();
                self.file_io
                    .trigger_save_graphml("graph_editor.graphml", content);
            }
            if ui
                .add_enabled(non_empty, egui::Button::new("⬇ .csv"))
                .accessible_hover_text("Download weighted CSV edge list")
                .clicked()
            {
                let content = self.to_csv_string();
                self.file_io
                    .trigger_save_csv("graph_editor_edges.csv", content);
            }
        });
        self.file_io.render_status_banner(ui);
    }

    pub(super) fn build_editor_plot_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = crate::app::SemanticPalette::for_theme(crate::app::ThemeMode::Dark);
        if self.braess.config.enabled && !self.braess.active_curve.is_empty() {
            return self.build_braess_curve_primitives(&palette);
        }
        let mut edge_specs = Vec::with_capacity(self.edges.len());
        for (i, &(u, v, _)) in self.edges.iter().enumerate() {
            let flow = self.braess.equilibrium_flows.get(i).copied().unwrap_or(0.0);
            let active = self.braess.config.enabled && flow > 1e-3;
            edge_specs.push(EdgePlotSpec {
                name: format!("Edge {u}->{v}"),
                u,
                v,
                color: if active {
                    palette.series_color(3)
                } else {
                    palette.graph_edge
                },
                width: if active { 3.0 } else { 2.0 },
                directed: self.braess.config.enabled,
            });
        }
        let positions: Vec<Pos2> = self.node_positions.values().copied().collect();
        let groups = [NodePlotGroup {
            name: "Vertices".to_string(),
            positions,
            color: palette.series_color(0),
            radius: 6.0,
        }];
        build_network_plot_primitives(&self.node_positions, &edge_specs, &groups)
    }

    fn build_braess_curve_primitives(
        &self,
        palette: &crate::app::SemanticPalette,
    ) -> Vec<ExportPrimitive> {
        let mut prims = Vec::new();
        if !self.braess.no_shortcut_curve.is_empty() {
            let pts: Vec<[f64; 2]> = self
                .braess
                .no_shortcut_curve
                .iter()
                .map(|p| [p.x, p.y])
                .collect();
            prims.push(ExportPrimitive::Polyline {
                name: "Without u->v".to_string(),
                points: SharedPoints::from(pts),
                color: palette.success,
                stroke_width: 2.0,
                closed: false,
                fill: None,
            });
        }
        let act_pts: Vec<[f64; 2]> = self
            .braess
            .active_curve
            .iter()
            .map(|p| [p.x, p.y])
            .collect();
        prims.push(ExportPrimitive::Polyline {
            name: "Active Network".to_string(),
            points: SharedPoints::from(act_pts),
            color: palette.series_color(3),
            stroke_width: 2.5,
            closed: false,
            fill: None,
        });
        if let Ok(summary) = self.braess.status {
            prims.push(ExportPrimitive::Points {
                name: "Equilibrium".to_string(),
                points: SharedPoints::from(vec![[
                    self.braess.config.total_demand,
                    summary.commuter_latency,
                ]]),
                color: palette.accent,
                radius: 5.0,
            });
        }
        prims
    }

    pub(super) fn build_editor_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        if self.braess.config.enabled && !self.braess.active_curve.is_empty() {
            let mut out = Vec::new();
            if !self.braess.no_shortcut_curve.is_empty() {
                out.push((
                    "Without u->v".to_string(),
                    self.braess
                        .no_shortcut_curve
                        .iter()
                        .map(|p| [p.x, p.y])
                        .collect(),
                ));
            }
            out.push((
                "Active Network".to_string(),
                self.braess
                    .active_curve
                    .iter()
                    .map(|p| [p.x, p.y])
                    .collect(),
            ));
            return out;
        }
        build_network_datasets(&self.node_positions, &self.edges)
    }
}

// [cite:graph_parameters_rust]
