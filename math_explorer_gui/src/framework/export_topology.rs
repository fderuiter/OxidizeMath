//! Deferred `.dot`, `.graphml`, `TikZ` (`.tex`), `.svg`, and `.csv` file and clipboard dispatch
//! and action-bar UI controls for [`super::InteractiveTool`] graph topology exports.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::export_ui::export_action_button;
use super::*;
use eframe::egui;

#[cfg(feature = "pure_math")]
use crate::web_io::OpenDialogConfig;
#[cfg(feature = "pure_math")]
use math_explorer::pure_math::graph_theory::graph_io::TopologyFormat;

/// Shared file-open dialog filter for `.dot`, `.gv`, `.graphml`, `.csv`, and `.tsv` topologies.
#[cfg(feature = "pure_math")]
pub const TOPOLOGY_FILE_DIALOG_CONFIG: OpenDialogConfig = OpenDialogConfig {
    filter_name: "Graph Topology (.dot, .graphml, .csv, .tsv)",
    extensions: &["dot", "gv", "graphml", "csv", "tsv"],
    accept: ".dot,.gv,.graphml,.csv,.tsv,text/vnd.graphviz,application/graphml+xml,text/csv,text/tab-separated-values",
    read_error_prefix: "Failed to read topology file",
};

const TOPOLOGY_MENU_SPECS: &[(ExportFormat, &str, &str, &str, &str)] = &[
    (
        ExportFormat::TopologyDot,
        "Export DOT (.dot)",
        "Download Graphviz DOT topology (.dot)",
        "Copy DOT",
        "Copy Graphviz DOT topology to clipboard",
    ),
    (
        ExportFormat::TopologyGraphMl,
        "Export GraphML (.graphml)",
        "Download Gephi/Cytoscape GraphML network (.graphml)",
        "Copy GraphML",
        "Copy GraphML XML topology to clipboard",
    ),
    (
        ExportFormat::TopologyTikz,
        "Export TikZ (.tex)",
        "Download standalone compilable LaTeX TikZ diagram (.tex)",
        "Copy TikZ",
        "Copy standalone LaTeX TikZ diagram to clipboard",
    ),
    (
        ExportFormat::TopologySvg,
        "Export Topology SVG",
        "Download vector SVG network diagram (.svg)",
        "Copy Topology SVG",
        "Copy vector SVG network diagram to clipboard",
    ),
    (
        ExportFormat::TopologyCsv,
        "Export Topology CSV",
        "Download canonical node/edge CSV topology table (.csv)",
        "Copy Topology CSV",
        "Copy canonical node/edge CSV topology table to clipboard",
    ),
];

impl SimulationFramework {
    /// Queues a direct graph topology clipboard copy for deferred end-of-frame dispatch.
    pub(crate) fn queue_topology_clipboard_export(&mut self, format: ExportFormat) {
        self.export_queue
            .push_target(format, ExportTarget::Clipboard);
    }

    /// Renders `.dot`, `.graphml`, `TikZ`, `.svg`, and `.csv` topology export buttons when `caps.topology` is enabled.
    pub(super) fn render_topology_export_items(
        &mut self,
        ui: &mut egui::Ui,
        caps: ExportCapabilities,
        close_on_click: bool,
    ) {
        if !caps.topology {
            return;
        }
        if close_on_click {
            ui.separator();
            ui.label(egui::RichText::new("🕸 Topology Formats").strong());
            self.render_topology_buttons_flat(ui, true);
        } else {
            let resp = ui.menu_button("🕸 Topology ▾", |menu_ui| {
                self.render_topology_buttons_flat(menu_ui, true);
            });
            resp.response.accessible_hover_text(
                "Export or copy network topology as DOT, GraphML, LaTeX TikZ, SVG, or CSV",
            );
        }
    }

    fn render_topology_buttons_flat(&mut self, ui: &mut egui::Ui, close_on_click: bool) {
        for &(fmt, file_lbl, file_tip, copy_lbl, copy_tip) in TOPOLOGY_MENU_SPECS {
            ui.horizontal(|row| {
                if export_action_button(row, true, file_lbl, file_tip, close_on_click) {
                    self.queue_export(fmt);
                }
                if export_action_button(row, true, copy_lbl, copy_tip, close_on_click) {
                    self.queue_topology_clipboard_export(fmt);
                }
            });
        }
    }

    /// Dispatches a queued `ExportFormat::Topology*` file or clipboard export request.
    pub(super) fn dispatch_topology_export(
        &mut self,
        format: ExportFormat,
        target: ExportTarget,
        ctx: Option<&egui::Context>,
    ) {
        let Some(tool) = &self.active_tool else {
            return;
        };
        let caps = tool.export_capabilities();
        if !caps.topology {
            self.set_export_status(ExportStatusSeverity::Warning, caps.topology_disabled_reason);
            return;
        }
        let tool_name = tool.name().to_string();
        match serialize_active_tool_topology(tool.as_ref(), format) {
            Ok(serialized) if !serialized.trim().is_empty() => {
                self.finish_serialized_topology(format, target, &tool_name, serialized, ctx);
            }
            Ok(_) => self.set_export_status(
                ExportStatusSeverity::Error,
                format!("No topology content available to export from {tool_name}."),
            ),
            Err(err) => self.set_export_status(
                ExportStatusSeverity::Error,
                format!("Topology export failed: {err}"),
            ),
        }
    }

    fn finish_serialized_topology(
        &mut self,
        format: ExportFormat,
        target: ExportTarget,
        tool_name: &str,
        serialized: String,
        ctx: Option<&egui::Context>,
    ) {
        match target {
            ExportTarget::File => {
                let filename = Self::build_export_filename(tool_name, format);
                let bytes = serialized.as_bytes().to_vec();
                self.complete_file_export(
                    format,
                    &filename,
                    &bytes,
                    Some(serialized),
                    format.filter_name(),
                    "Topology export failed",
                );
            }
            ExportTarget::Clipboard => {
                self.dispatch_clipboard_text_request(format, serialized, ctx);
            }
        }
    }
}

#[cfg(feature = "pure_math")]
fn serialize_active_tool_topology(
    tool: &dyn InteractiveTool,
    format: ExportFormat,
) -> Result<String, String> {
    let codec_fmt = match format {
        ExportFormat::TopologyDot => TopologyFormat::Dot,
        ExportFormat::TopologyGraphMl => TopologyFormat::GraphMl,
        ExportFormat::TopologyTikz => TopologyFormat::Tikz,
        ExportFormat::TopologySvg => TopologyFormat::Svg,
        ExportFormat::TopologyCsv => TopologyFormat::Csv,
        _ => return Err("Unsupported topology export format".to_string()),
    };
    let doc = tool
        .export_topology_document()
        .ok_or_else(|| format!("{} did not provide a GraphTopologyDocument", tool.name()))??;
    doc.serialize(codec_fmt).map_err(|e| e.to_string())
}

#[cfg(not(feature = "pure_math"))]
fn serialize_active_tool_topology(
    _tool: &dyn InteractiveTool,
    _format: ExportFormat,
) -> Result<String, String> {
    Err("Topology codec requires the 'pure_math' Cargo feature".to_string())
}
