//! Non-blocking file open/save dialog channels and Color32 hex formatting for `GraphTheoryTab` tools.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::shared_topology::{
    parse_topology_document, queue_topology_handoff, read_shared_topology_buffer,
    SharedGraphTopology, TopologyHandoffTarget,
};
use crate::accessibility::{announce_live_status, AccessibleHoverText};
use crate::web_io::{
    spawn_open_file_dialog, spawn_save_as_file_dialog, OpenDialogConfig, PlatformInstant,
    SaveAsDialogConfig,
};
use eframe::egui;
use std::sync::mpsc::{self, Receiver, TryRecvError};

const TOPOLOGY_OPEN_CONFIG: OpenDialogConfig = OpenDialogConfig {
    filter_name: "Graph Topology (.dot, .graphml, .csv, .json)",
    extensions: &["dot", "gv", "graphml", "csv", "json"],
    accept:
        ".dot,.gv,.graphml,.csv,.json,text/vnd.graphviz,application/xml,text/csv,application/json",
    read_error_prefix: "Failed to read graph file",
};

const DOT_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "Graphviz DOT (.dot)",
    extensions: &["dot"],
    fallback_filename: "network.dot",
    write_error_prefix: "Failed to write .dot file",
};

const GRAPHML_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "GraphML Topology (.graphml)",
    extensions: &["graphml"],
    fallback_filename: "network.graphml",
    write_error_prefix: "Failed to write .graphml file",
};

const CSV_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "CSV Table (.csv)",
    extensions: &["csv"],
    fallback_filename: "network.csv",
    write_error_prefix: "Failed to write .csv file",
};

type OpenRx = Receiver<Result<(String, String), String>>;
type SaveRx = Receiver<Result<String, String>>;

/// Non-blocking async file import/export channel state owned by each `GraphTheoryTab` tool.
#[derive(Default)]
pub struct GraphFileIoState {
    open_rx: Option<OpenRx>,
    save_rx: Option<SaveRx>,
    /// Most recent status or validation message together with `(is_error, timestamp)`.
    pub last_status: Option<(String, bool, PlatformInstant)>,
}

impl GraphFileIoState {
    /// Records a status banner message stamped with [`PlatformInstant::now()`].
    pub fn set_status(&mut self, msg: impl Into<String>, is_error: bool) {
        self.last_status = Some((msg.into(), is_error, PlatformInstant::now()));
    }

    /// Spawns a non-blocking `.dot`, `.graphml`, `.csv`, or `.json` file picker via [`spawn_open_file_dialog`].
    pub fn trigger_import_dialog(&mut self) {
        let (tx, rx) = mpsc::channel();
        self.open_rx = Some(rx);
        spawn_open_file_dialog(tx, TOPOLOGY_OPEN_CONFIG, |path| {
            path.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("imported.dot")
                .to_string()
        });
    }

    /// Spawns a non-blocking `.dot` download/save dialog via [`spawn_save_as_file_dialog`].
    pub fn trigger_save_dot(&mut self, default_filename: &str, content: Result<String, String>) {
        self.spawn_save(DOT_SAVE_CONFIG, default_filename, content);
    }

    /// Spawns a non-blocking `.graphml` download/save dialog via [`spawn_save_as_file_dialog`].
    pub fn trigger_save_graphml(
        &mut self,
        default_filename: &str,
        content: Result<String, String>,
    ) {
        self.spawn_save(GRAPHML_SAVE_CONFIG, default_filename, content);
    }

    /// Spawns a non-blocking `.csv` download/save dialog via [`spawn_save_as_file_dialog`].
    pub fn trigger_save_csv(&mut self, default_filename: &str, content: Result<String, String>) {
        self.spawn_save(CSV_SAVE_CONFIG, default_filename, content);
    }

    fn spawn_save(
        &mut self,
        config: SaveAsDialogConfig,
        default_filename: &str,
        content: Result<String, String>,
    ) {
        match content {
            Ok(payload) => {
                let (tx, rx) = mpsc::channel();
                self.save_rx = Some(rx);
                spawn_save_as_file_dialog(
                    tx,
                    config,
                    default_filename,
                    payload,
                    |path, fallback| {
                        path.file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or(fallback)
                            .to_string()
                    },
                );
            }
            Err(err) => self.set_status(err, true),
        }
    }

    /// Polls pending open/save channels without blocking, returning any newly imported topology.
    pub fn poll(&mut self, ctx: &egui::Context) -> Option<SharedGraphTopology> {
        self.poll_save_channel(ctx);
        self.poll_open_channel(ctx)
    }

    fn poll_save_channel(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.save_rx else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(name)) => {
                let msg = format!("Exported {name}");
                announce_live_status(ctx, msg.clone());
                self.set_status(msg, false);
                self.save_rx = None;
            }
            Ok(Err(err)) => {
                announce_live_status(ctx, err.clone());
                self.set_status(err, true);
                self.save_rx = None;
            }
            Err(TryRecvError::Disconnected) => {
                self.save_rx = None;
            }
            Err(TryRecvError::Empty) => {}
        }
    }

    fn poll_open_channel(&mut self, ctx: &egui::Context) -> Option<SharedGraphTopology> {
        let rx = self.open_rx.as_ref()?;
        match rx.try_recv() {
            Ok(Ok((filename, text))) => {
                self.open_rx = None;
                match parse_topology_document(&filename, &text) {
                    Ok(topo) => {
                        let msg = format!(
                            "Imported {} ({} nodes, {} edges)",
                            filename,
                            topo.nodes.len(),
                            topo.edges.len()
                        );
                        announce_live_status(ctx, msg.clone());
                        self.set_status(msg, false);
                        Some(topo)
                    }
                    Err(err) => {
                        let msg = format!("Import rejected ({filename}): {err}");
                        announce_live_status(ctx, msg.clone());
                        self.set_status(msg, true);
                        None
                    }
                }
            }
            Ok(Err(err)) => {
                self.open_rx = None;
                announce_live_status(ctx, err.clone());
                self.set_status(err, true);
                None
            }
            Err(TryRecvError::Disconnected) => {
                self.open_rx = None;
                None
            }
            Err(TryRecvError::Empty) => None,
        }
    }

    /// Renders compact status feedback if an import/export completed within the last 12 seconds.
    pub fn render_status_banner(&self, ui: &mut egui::Ui) {
        let Some((msg, is_error, stamp)) = &self.last_status else {
            return;
        };
        if stamp.elapsed().as_secs_f32() > 12.0 {
            return;
        }
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let color = if *is_error {
            palette.error
        } else {
            palette.success
        };
        ui.colored_label(color, egui::RichText::new(msg).small());
    }
}

/// Converts an `egui::Color32` into a deterministic lowercase `#rrggbb` Graphviz hex string.
#[must_use]
pub fn color32_to_hex(color: egui::Color32) -> String {
    format!("#{:02x}{:02x}{:02x}", color.r(), color.g(), color.b())
}

fn peer_handoff_specs(
    current: TopologyHandoffTarget,
) -> [(TopologyHandoffTarget, &'static str, &'static str); 2] {
    match current {
        TopologyHandoffTarget::GraphEditor => [
            (
                TopologyHandoffTarget::AlgorithmVisualizer,
                "↗ Visualizer",
                "Send active graph topology to Algorithm Visualizer",
            ),
            (
                TopologyHandoffTarget::NetworkMetrics,
                "↗ Metrics",
                "Send active graph topology to Network Metrics",
            ),
        ],
        TopologyHandoffTarget::AlgorithmVisualizer => [
            (
                TopologyHandoffTarget::GraphEditor,
                "↗ Editor",
                "Open active graph topology in Graph Editor",
            ),
            (
                TopologyHandoffTarget::NetworkMetrics,
                "↗ Metrics",
                "Send active graph topology to Network Metrics",
            ),
        ],
        TopologyHandoffTarget::NetworkMetrics => [
            (
                TopologyHandoffTarget::GraphEditor,
                "↗ Editor",
                "Open active graph topology in Graph Editor",
            ),
            (
                TopologyHandoffTarget::AlgorithmVisualizer,
                "↗ Visualizer",
                "Send active graph topology to Algorithm Visualizer",
            ),
        ],
    }
}

/// Renders one-click cross-tool handoff buttons (`Send to ...` and `Load Shared`).
pub fn render_handoff_buttons(
    ui: &mut egui::Ui,
    current: TopologyHandoffTarget,
    can_send: bool,
    build_current: impl FnOnce() -> SharedGraphTopology,
) -> Option<SharedGraphTopology> {
    let mut loaded = None;
    let peers = peer_handoff_specs(current);
    let mut send_target = None;
    ui.horizontal_wrapped(|ui| {
        for (target, label, tip) in peers {
            if ui
                .add_enabled(can_send, egui::Button::new(label))
                .accessible_hover_text(tip)
                .clicked()
            {
                send_target = Some(target);
            }
        }
        if let Some(shared) = read_shared_topology_buffer(ui.ctx()) {
            if ui
                .button("📥 Shared")
                .accessible_hover_text("Load the shared topology buffer from Graph Theory tab")
                .clicked()
            {
                loaded = Some(shared);
            }
        }
    });
    if let Some(target) = send_target {
        queue_topology_handoff(ui.ctx(), target, build_current());
    }
    loaded
}

// [cite:graph_parameters_rust]
