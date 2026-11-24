//! Deferred graph topology file download, clipboard copy, and action-bar UI controls.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::export_ui::export_action_button;
use super::*;
use eframe::egui;

#[cfg(feature = "pure_math")]
use math_explorer::pure_math::graph_theory::interchange::TopologyFormat;

#[cfg(feature = "pure_math")]
const fn map_topology_format(format: ExportFormat) -> Option<TopologyFormat> {
    match format {
        ExportFormat::TopologyDot => Some(TopologyFormat::Dot),
        ExportFormat::TopologyGraphMl => Some(TopologyFormat::GraphMl),
        ExportFormat::TopologyTikz => Some(TopologyFormat::Tikz),
        ExportFormat::TopologySvg => Some(TopologyFormat::Svg),
        ExportFormat::TopologyCsv => Some(TopologyFormat::Csv),
        _ => None,
    }
}

impl SimulationFramework {
    /// Queues a direct graph topology text clipboard copy for deferred end-of-frame dispatch.
    pub(crate) fn queue_topology_clipboard_export(&mut self, format: ExportFormat) {
        self.export_queue
            .push_target(format, ExportTarget::Clipboard);
    }

    /// Renders the `🕸 Topology ▾` export menu when the active tool advertises `caps.topology`.
    pub(super) fn render_topology_export_items(
        &mut self,
        ui: &mut egui::Ui,
        caps: ExportCapabilities,
        _close_on_click: bool,
    ) {
        if !caps.topology {
            return;
        }
        let resp = ui.menu_button("🕸 Topology ▾", |menu_ui| {
            self.render_topology_menu_entries(menu_ui);
        });
        resp.response.accessible_hover_text(
            "Export or copy structured network topology (.dot, .graphml, TikZ .tex, .svg, .csv)",
        );
    }

    fn render_topology_menu_entries(&mut self, ui: &mut egui::Ui) {
        let entries = [
            (
                "Export DOT (.dot)",
                "Download Graphviz DOT network topology (.dot)",
                ExportFormat::TopologyDot,
                false,
            ),
            (
                "Copy DOT",
                "Copy Graphviz DOT topology to clipboard",
                ExportFormat::TopologyDot,
                true,
            ),
            (
                "Export GraphML (.graphml)",
                "Download standard XML GraphML network (.graphml)",
                ExportFormat::TopologyGraphMl,
                false,
            ),
            (
                "Copy GraphML",
                "Copy XML GraphML network to clipboard",
                ExportFormat::TopologyGraphMl,
                true,
            ),
            (
                "Export TikZ (.tex)",
                "Download standalone compilable LaTeX TikZ figure (.tex)",
                ExportFormat::TopologyTikz,
                false,
            ),
            (
                "Copy TikZ",
                "Copy standalone LaTeX TikZ figure to clipboard",
                ExportFormat::TopologyTikz,
                true,
            ),
            (
                "Export Topology SVG (.svg)",
                "Download standalone vector network diagram (.svg)",
                ExportFormat::TopologySvg,
                false,
            ),
            (
                "Export Topology CSV (.csv)",
                "Download combined node and edge topology table (.csv)",
                ExportFormat::TopologyCsv,
                false,
            ),
        ];
        for (label, tip, fmt, to_clipboard) in entries {
            if export_action_button(ui, true, label, tip, true) {
                if to_clipboard {
                    self.queue_topology_clipboard_export(fmt);
                } else {
                    self.queue_export(fmt);
                }
            }
        }
    }

    #[cfg(feature = "pure_math")]
    fn serialize_active_topology(&mut self, topo_fmt: TopologyFormat) -> Option<(String, String)> {
        let tool = self.active_tool.as_ref()?;
        let caps = tool.export_capabilities();
        if !caps.topology {
            self.set_export_status(ExportStatusSeverity::Warning, caps.topology_disabled_reason);
            return None;
        }
        let tool_name = tool.name().to_string();
        match tool
            .export_topology_document()
            .and_then(|doc| doc.serialize(topo_fmt).map_err(|e| e.to_string()))
        {
            Ok(text) if !text.trim().is_empty() => Some((tool_name, text)),
            Ok(_) => {
                self.set_export_status(
                    ExportStatusSeverity::Error,
                    format!("No graph topology available to export from {tool_name}."),
                );
                None
            }
            Err(err) => {
                self.set_export_status(
                    ExportStatusSeverity::Error,
                    format!("Topology export failed: {err}"),
                );
                None
            }
        }
    }

    /// Dispatches a queued `Topology*` export request to file download or system clipboard.
    pub(super) fn dispatch_topology_export(
        &mut self,
        format: ExportFormat,
        target: ExportTarget,
        ctx: Option<&egui::Context>,
    ) {
        #[cfg(not(feature = "pure_math"))]
        {
            let _ = (format, target, ctx);
            self.set_export_status(
                ExportStatusSeverity::Warning,
                "Graph topology export requires the pure_math feature.",
            );
        }
        #[cfg(feature = "pure_math")]
        {
            let Some(topo_fmt) = map_topology_format(format) else {
                return;
            };
            let Some((tool_name, serialized)) = self.serialize_active_topology(topo_fmt) else {
                return;
            };
            match target {
                ExportTarget::File => {
                    let filename = Self::build_export_filename(&tool_name, format);
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
}
