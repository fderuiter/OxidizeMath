#[path = "export_queue_models.rs"]
mod export_queue_models;
use export_queue_models::format_clipboard_error;
pub(crate) use export_queue_models::{
    ActionBarExportStatus, CustomClipboardWriter, CustomExportWriter, ExportFormat, ExportRequest,
    ExportTarget,
};

use super::*;
use std::path::PathBuf;

/// Manages pending dataset and figure export requests deferred to the end of the rendering frame.
#[derive(Default)]
pub(crate) struct ExportQueue {
    target_requests: Vec<ExportRequest>,
    /// Optional directory to directly write exported files (useful in headless/automated test runs).
    pub(crate) destination_dir: Option<PathBuf>,
    /// Optional custom sink callback to receive exported filenames and file bytes.
    pub(crate) custom_writer: Option<CustomExportWriter>,
    /// Optional custom clipboard sink callback to receive serialized text content.
    pub(crate) custom_clipboard_writer: Option<CustomClipboardWriter>,
    /// Retains the last exported filename (for inspection and verification).
    pub(crate) last_exported_filename: Option<String>,
    /// Retains the last exported file or clipboard text content (for CSV/JSON/SVG inspection).
    pub(crate) last_exported_content: Option<String>,
    /// Retains the last exported raw byte payload (for PNG/GIF/SVG/CSV/JSON verification).
    pub(crate) last_exported_bytes: Option<Vec<u8>>,
    /// Retains the last content copied directly to the clipboard.
    pub(crate) last_clipboard_content: Option<String>,
    /// Current inline status feedback for action-bar export actions.
    pub(crate) status: Option<ActionBarExportStatus>,
}

impl std::fmt::Debug for ExportQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportQueue")
            .field("target_requests", &self.target_requests)
            .field("destination_dir", &self.destination_dir)
            .field("last_exported_filename", &self.last_exported_filename)
            .field("last_exported_content", &self.last_exported_content)
            .field(
                "last_exported_bytes_len",
                &self.last_exported_bytes.as_ref().map(Vec::len),
            )
            .field("last_clipboard_content", &self.last_clipboard_content)
            .field("status", &self.status)
            .finish()
    }
}

impl ExportQueue {
    /// Queues a file export request, preventing duplicates from rapid repeated clicks within the frame.
    pub(crate) fn push(&mut self, format: ExportFormat) -> bool {
        self.push_target(format, ExportTarget::File)
    }

    /// Queues an export request for a specific output target, deduplicating identical `(format, target)` pairs.
    pub(crate) fn push_target(&mut self, format: ExportFormat, target: ExportTarget) -> bool {
        let req = ExportRequest { format, target };
        if self.target_requests.contains(&req) {
            return false;
        }
        self.target_requests.push(req);
        true
    }

    /// Queues a direct CSV dataset export to the system clipboard, deduplicating rapid clicks.
    pub(crate) fn queue_clipboard_export(&mut self) -> bool {
        self.push_target(ExportFormat::Csv, ExportTarget::Clipboard)
    }

    /// Returns `true` if there are no pending export requests.
    pub(crate) fn is_empty(&self) -> bool {
        self.target_requests.is_empty()
    }

    /// Returns the number of pending export requests across all targets.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.target_requests.len()
    }

    /// Returns a slice of all pending target-aware export requests.
    #[cfg(test)]
    pub(crate) fn export_requests(&self) -> &[ExportRequest] {
        &self.target_requests
    }

    /// Drains all pending target-aware export requests from the queue.
    pub(crate) fn drain_requests(&mut self) -> Vec<ExportRequest> {
        std::mem::take(&mut self.target_requests)
    }
}

impl SimulationFramework {
    /// Records an export status message with visual severity styling and dispatches a screen-reader announcement.
    pub(crate) fn set_export_status(
        &mut self,
        severity: ExportStatusSeverity,
        message: impl Into<String>,
    ) {
        let msg = message.into();
        crate::accessibility::announce_status_with_priority(&msg, severity.aria_priority());
        self.export_status_severity = severity;
        self.export_status_message = Some(msg.clone());
        self.export_queue.status = Some(ActionBarExportStatus {
            message: msg,
            is_error: severity != ExportStatusSeverity::Success,
            expires_at_time: None,
        });
    }

    /// Queues a dataset or figure file export request for deferred end-of-frame dispatch.
    pub(crate) fn queue_export(&mut self, format: ExportFormat) {
        self.export_queue.push(format);
    }

    /// Resolves export datasets for the active tool, falling back to `current_valid_plot_datasets`
    /// when thread-local plot datasets were cleared prior to modal rendering or across panes.
    pub(crate) fn export_datasets_with_fallback(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let Some(tool) = &self.active_tool else {
            return Vec::new();
        };
        let datasets = tool.export_datasets();
        if datasets.is_empty()
            && self.sync_state.datasets_owner_tool == Some(tool.name())
            && !self.current_valid_plot_datasets.is_empty()
        {
            self.current_valid_plot_datasets.clone()
        } else {
            datasets
        }
    }

    pub(crate) fn export_datasets_3d_with_fallback(&self) -> Vec<(String, Vec<[f64; 3]>)> {
        let Some(tool) = &self.active_tool else {
            return Vec::new();
        };
        let datasets = tool.export_datasets_3d();
        if datasets.is_empty()
            && self.sync_state.datasets_owner_tool == Some(tool.name())
            && !self.current_valid_plot_datasets_3d.is_empty()
        {
            self.current_valid_plot_datasets_3d.clone()
        } else {
            datasets
        }
    }

    /// Queues a direct CSV dataset clipboard export for deferred end-of-frame dispatch.
    pub(crate) fn queue_clipboard_export(&mut self) {
        self.export_queue.queue_clipboard_export();
    }

    /// Dispatches all pending export requests with an `egui::Context` for repaint scheduling and async callbacks.
    pub(crate) fn dispatch_pending_exports_with_ctx(&mut self, ctx: &egui::Context) {
        if self.export_queue.is_empty() {
            return;
        }
        let requests = self.export_queue.drain_requests();
        if self.active_tool.is_none() {
            self.set_export_status(
                ExportStatusSeverity::Warning,
                "No active simulation tool available for export.",
            );
            ctx.request_repaint();
            return;
        }
        for req in requests {
            self.dispatch_single_request(req, Some(ctx));
        }
        ctx.request_repaint();
    }

    fn dispatch_single_request(&mut self, req: ExportRequest, ctx: Option<&egui::Context>) {
        match (req.format, req.target) {
            (ExportFormat::Csv | ExportFormat::Json, target) => {
                self.dispatch_dataset_export(req.format, target, ctx);
            }
            (ExportFormat::Png, ExportTarget::File) => self.dispatch_png_export(),
            (ExportFormat::Png, ExportTarget::Clipboard) => self.dispatch_png_clipboard_export(ctx),
            (ExportFormat::Svg, _) => self.dispatch_svg_export(),
            (ExportFormat::Gif, _) => self.dispatch_gif_export(),
            (ExportFormat::GlslWebGl2 | ExportFormat::GlslShadertoy, target) => {
                self.dispatch_shader_export(req.format, target, ctx);
            }
        }
    }

    fn serialize_active_dataset_export(
        &self,
        tool_name: &str,
        format: ExportFormat,
    ) -> Result<String, String> {
        let datasets_3d: Vec<(String, Vec<[f64; 3]>)> = self
            .export_datasets_3d_with_fallback()
            .into_iter()
            .filter(|(_, pts)| {
                pts.iter()
                    .any(|p| p[0].is_finite() && p[1].is_finite() && p[2].is_finite())
            })
            .collect();
        if !datasets_3d.is_empty() {
            return match format {
                ExportFormat::Csv => Ok(egui_plot::export::format_csv_3d(&datasets_3d)),
                ExportFormat::Json => Ok(egui_plot::export::format_json_3d(&datasets_3d)),
                _ => Err("Unsupported dataset format".to_string()),
            };
        }
        let datasets: Vec<(String, Vec<[f64; 2]>)> = self
            .export_datasets_with_fallback()
            .into_iter()
            .filter(|(_, pts)| pts.iter().any(|p| p[0].is_finite() && p[1].is_finite()))
            .collect();
        if !egui_plot::export_primitives::has_finite_dataset_points(&datasets) {
            return Err(format!(
                "No dataset points available to export from {tool_name}."
            ));
        }
        match format {
            ExportFormat::Csv => Ok(egui_plot::export::format_csv(&datasets)),
            ExportFormat::Json => Ok(egui_plot::export::format_json(&datasets)),
            _ => Err("Unsupported dataset format".to_string()),
        }
    }

    fn dispatch_dataset_export(
        &mut self,
        format: ExportFormat,
        target: ExportTarget,
        ctx: Option<&egui::Context>,
    ) {
        let Some(tool) = &self.active_tool else {
            return;
        };
        let caps = tool.export_capabilities();
        if !caps.dataset {
            self.set_export_status(ExportStatusSeverity::Warning, caps.dataset_disabled_reason);
            return;
        }
        let tool_name = tool.name().to_string();
        let content = match self.serialize_active_dataset_export(&tool_name, format) {
            Ok(s) => s,
            Err(err) => {
                self.set_export_status(ExportStatusSeverity::Error, err);
                return;
            }
        };
        match target {
            ExportTarget::File => {
                let filename = Self::build_export_filename(&tool_name, format);
                let bytes = content.as_bytes().to_vec();
                self.complete_file_export(
                    format,
                    &filename,
                    &bytes,
                    Some(content),
                    format.filter_name(),
                    "Export failed",
                );
            }
            ExportTarget::Clipboard => self.dispatch_clipboard_text_request(format, content, ctx),
        }
    }

    pub(super) fn dispatch_clipboard_text_request(
        &mut self,
        format: ExportFormat,
        content: String,
        ctx: Option<&egui::Context>,
    ) {
        let label = format.label();
        let noun = format.clipboard_item_noun();
        let res = if let Some(ref writer) = self.export_queue.custom_clipboard_writer {
            writer(&content)
        } else {
            self.write_system_text_clipboard(label, noun, &content, ctx)
        };
        match res {
            Ok(()) => {
                self.export_queue.last_exported_content = Some(content.clone());
                self.export_queue.last_clipboard_content = Some(content);
                let msg = if cfg!(target_arch = "wasm32")
                    && self.export_queue.custom_clipboard_writer.is_none()
                {
                    format!("Copying {label} {noun} to clipboard...")
                } else {
                    format!("Copied {label} {noun} to clipboard!")
                };
                self.set_export_status(ExportStatusSeverity::Success, msg);
            }
            Err(err) => {
                self.set_export_status(ExportStatusSeverity::Error, format_clipboard_error(&err));
            }
        }
    }

    fn write_system_text_clipboard(
        &self,
        label: &'static str,
        noun: &'static str,
        content: &str,
        ctx: Option<&egui::Context>,
    ) -> Result<(), String> {
        #[cfg(test)]
        {
            let _ = (label, noun, content, ctx);
            Ok(())
        }
        #[cfg(not(test))]
        {
            if let Some(egui_ctx) = ctx.cloned() {
                let status_id =
                    egui::Id::new(("framework_action_bar_clipboard_status", self.instance_id));
                egui_plot::export::copy_text_to_clipboard_with_callback(content, move |cb| {
                    egui_ctx.data_mut(|d| {
                        let status = match cb {
                            Ok(()) => ActionBarExportStatus {
                                message: format!("Copied {label} {noun} to clipboard!"),
                                is_error: false,
                                expires_at_time: None,
                            },
                            Err(err) => ActionBarExportStatus {
                                message: format_clipboard_error(&err),
                                is_error: true,
                                expires_at_time: None,
                            },
                        };
                        d.insert_temp(status_id, status);
                    });
                    egui_ctx.request_repaint();
                })
            } else {
                egui_plot::export::copy_text_to_clipboard(content)
            }
        }
    }

    pub(super) fn complete_file_export(
        &mut self,
        format: ExportFormat,
        filename: &str,
        bytes: &[u8],
        text_content: Option<String>,
        ok_label: &str,
        err_prefix: &str,
    ) {
        match self.persist_export_bytes(format, filename, bytes, text_content) {
            Ok(()) => self.set_export_status(
                ExportStatusSeverity::Success,
                format!("{ok_label} saved successfully."),
            ),
            Err(err) => {
                self.set_export_status(ExportStatusSeverity::Error, format!("{err_prefix}: {err}"))
            }
        }
    }

    pub(super) fn build_export_filename(tool_name: &str, format: ExportFormat) -> String {
        let slug: String = tool_name
            .to_lowercase()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let trimmed = slug.trim_matches('_');
        let base = if trimmed.is_empty() {
            "simulation"
        } else {
            trimmed
        };
        format!("{base}_{}.{}", format.stem_suffix(), format.extension())
    }

    fn persist_export_bytes(
        &mut self,
        format: ExportFormat,
        filename: &str,
        bytes: &[u8],
        text_content: Option<String>,
    ) -> Result<(), String> {
        if bytes.is_empty() {
            return Err("Refusing to write empty export payload".to_string());
        }
        self.export_queue.last_exported_filename = Some(filename.to_string());
        self.export_queue.last_exported_content = text_content;
        self.export_queue.last_exported_bytes = Some(bytes.to_vec());

        if let Some(ref dir) = self.export_queue.destination_dir {
            let path = dir.join(filename);
            std::fs::write(&path, bytes)
                .map_err(|e| format!("Failed to write {}: {e}", path.display()))?;
        } else if let Some(ref writer) = self.export_queue.custom_writer {
            writer(filename, bytes);
        } else {
            #[cfg(not(test))]
            egui_plot::export::save_file_dialog(
                filename,
                bytes,
                format.filter_name(),
                &[format.extension()],
            );
            #[cfg(test)]
            let _ = format;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "deferred_export_tests.rs"]
mod deferred_export_tests;

#[cfg(test)]
#[path = "deferred_export_audit_tests.rs"]
mod deferred_export_audit_tests;
