use super::*;
use std::path::PathBuf;
use std::sync::Arc;

/// Supported format for deferred simulation dataset export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExportFormat {
    /// Comma-separated values format (.csv).
    Csv,
    /// JavaScript Object Notation format (.json).
    Json,
}

impl ExportFormat {
    /// Returns the standard file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Csv => "csv",
            Self::Json => "json",
        }
    }

    /// Returns the uppercase label for display messages.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Csv => "CSV",
            Self::Json => "JSON",
        }
    }

    /// Returns the file dialog filter description.
    pub fn filter_name(&self) -> &'static str {
        match self {
            Self::Csv => "CSV File",
            Self::Json => "JSON File",
        }
    }
}

/// Output destination target for deferred simulation dataset export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExportTarget {
    /// Save serialized dataset to disk via file save dialog or browser download.
    File,
    /// Copy serialized dataset text directly to the system clipboard.
    Clipboard,
}

/// A deferred dataset export request combining serialization format and output target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExportRequest {
    /// Dataset serialization format.
    pub format: ExportFormat,
    /// Output destination target.
    pub target: ExportTarget,
}

/// Inline status feedback for action-bar dataset export operations.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ActionBarExportStatus {
    /// Human-readable confirmation or error message.
    pub message: String,
    /// Whether this status indicates an error or warning condition.
    pub is_error: bool,
    /// Optional `egui` timestamp (`ctx.input(|i| i.time)`) when temporary confirmation expires.
    pub expires_at_time: Option<f64>,
}

/// Type alias for custom export file writer function.
pub type CustomExportWriter = Arc<dyn Fn(&str, &[u8]) + Send + Sync>;

/// Type alias for custom clipboard text writer function (useful for deterministic testing).
pub type CustomClipboardWriter = Arc<dyn Fn(&str) -> Result<(), String> + Send + Sync>;

/// Formats a clipboard failure error with guidance toward disk export.
pub(super) fn format_clipboard_error(err: &str) -> String {
    if err.contains("Export CSV") {
        format!("Clipboard unavailable: {err}")
    } else {
        format!("Clipboard unavailable ({err}). Please use Export CSV to save to disk.")
    }
}

/// Manages pending dataset export requests deferred to the end of the rendering frame.
#[derive(Default)]
pub struct ExportQueue {
    requests: Vec<ExportFormat>,
    target_requests: Vec<ExportRequest>,
    /// Optional directory to directly write exported files (useful in headless/automated test runs).
    pub destination_dir: Option<PathBuf>,
    /// Optional custom sink callback to receive exported filenames and file content.
    pub custom_writer: Option<CustomExportWriter>,
    /// Optional custom clipboard sink callback to receive serialized text content.
    pub custom_clipboard_writer: Option<CustomClipboardWriter>,
    /// Retains the last exported filename (for inspection and verification).
    pub last_exported_filename: Option<String>,
    /// Retains the last exported file or clipboard content string (for inspection and verification).
    pub last_exported_content: Option<String>,
    /// Retains the last content copied directly to the clipboard.
    pub last_clipboard_content: Option<String>,
    /// Current inline status feedback for action-bar export actions.
    pub status: Option<ActionBarExportStatus>,
}

impl std::fmt::Debug for ExportQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportQueue")
            .field("requests", &self.requests)
            .field("target_requests", &self.target_requests)
            .field("destination_dir", &self.destination_dir)
            .field("last_exported_filename", &self.last_exported_filename)
            .field("last_exported_content", &self.last_exported_content)
            .field("last_clipboard_content", &self.last_clipboard_content)
            .field("status", &self.status)
            .finish()
    }
}

impl ExportQueue {
    /// Queues a file export request, preventing duplicates from rapid repeated clicks within the frame.
    /// Returns `true` if the request was added, or `false` if already queued.
    pub fn push(&mut self, format: ExportFormat) -> bool {
        self.push_target(format, ExportTarget::File)
    }

    /// Queues an export request for a specific output target, deduplicating identical `(format, target)` pairs.
    pub fn push_target(&mut self, format: ExportFormat, target: ExportTarget) -> bool {
        let req = ExportRequest { format, target };
        if self.target_requests.contains(&req) {
            return false;
        }
        if target == ExportTarget::File {
            self.requests.push(format);
        }
        self.target_requests.push(req);
        true
    }

    /// Queues a direct CSV dataset export to the system clipboard, deduplicating rapid clicks.
    pub fn queue_clipboard_export(&mut self) -> bool {
        self.push_target(ExportFormat::Csv, ExportTarget::Clipboard)
    }

    /// Returns `true` if there are no pending export requests.
    pub fn is_empty(&self) -> bool {
        self.target_requests.is_empty()
    }

    /// Returns the number of pending export requests across all targets.
    pub fn len(&self) -> usize {
        self.target_requests.len()
    }

    /// Returns a slice of pending file-target export formats.
    pub fn requests(&self) -> &[ExportFormat] {
        &self.requests
    }

    /// Returns a slice of all pending target-aware export requests.
    pub fn export_requests(&self) -> &[ExportRequest] {
        &self.target_requests
    }

    /// Drains all pending file-target export formats and clears the queue.
    pub fn drain(&mut self) -> Vec<ExportFormat> {
        self.target_requests.clear();
        std::mem::take(&mut self.requests)
    }

    /// Drains all pending target-aware export requests from the queue.
    pub fn drain_requests(&mut self) -> Vec<ExportRequest> {
        self.requests.clear();
        std::mem::take(&mut self.target_requests)
    }

    /// Clears the queue without executing pending requests.
    pub fn clear(&mut self) {
        self.requests.clear();
        self.target_requests.clear();
    }
}

impl SimulationFramework {
    /// Queues a disk file dataset export request for deferred end-of-frame dispatch.
    pub fn queue_export(&mut self, format: ExportFormat) {
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

    /// Queues a dataset export request with an explicit destination target.
    pub fn queue_export_with_target(&mut self, format: ExportFormat, target: ExportTarget) {
        self.export_queue.push_target(format, target);
    }

    /// Queues a direct CSV dataset clipboard export for deferred end-of-frame dispatch.
    pub fn queue_clipboard_export(&mut self) {
        self.export_queue.queue_clipboard_export();
    }

    pub(super) fn set_action_bar_status(&mut self, message: String, is_error: bool) {
        self.export_status_message = Some(message.clone());
        self.export_queue.status = Some(ActionBarExportStatus {
            message,
            is_error,
            expires_at_time: None,
        });
    }

    pub(super) fn poll_action_bar_clipboard_status(&mut self, ctx: &egui::Context) {
        let status_id = egui::Id::new(("framework_action_bar_clipboard_status", self.instance_id));
        if let Some(async_status) =
            ctx.data_mut(|d| d.remove_temp::<ActionBarExportStatus>(status_id))
        {
            self.export_status_message = Some(async_status.message.clone());
            self.export_queue.status = Some(async_status);
        }
    }

    fn dispatch_file_request(&mut self, file_slug: &str, format: ExportFormat, content: String) {
        let filename = format!("{file_slug}_dataset.{}", format.extension());
        self.export_queue.last_exported_filename = Some(filename.clone());
        self.export_queue.last_exported_content = Some(content.clone());

        if let Some(ref dir) = self.export_queue.destination_dir {
            let path = dir.join(&filename);
            let _ = std::fs::write(path, content.as_bytes());
        } else if let Some(ref writer) = self.export_queue.custom_writer {
            writer(&filename, content.as_bytes());
        } else {
            #[cfg(not(test))]
            egui_plot::export::save_file_dialog(
                &filename,
                content.as_bytes(),
                format.filter_name(),
                &[format.extension()],
            );
            #[cfg(test)]
            {
                let _ = (&filename, &content, &format);
            }
        }
    }

    fn dispatch_clipboard_request(
        &mut self,
        format: ExportFormat,
        content: String,
        ctx: Option<&egui::Context>,
    ) {
        let label = format.label();
        let res = if let Some(ref writer) = self.export_queue.custom_clipboard_writer {
            writer(&content)
        } else {
            #[cfg(not(test))]
            {
                if let Some(egui_ctx) = ctx.cloned() {
                    let status_id =
                        egui::Id::new(("framework_action_bar_clipboard_status", self.instance_id));
                    egui_plot::export::copy_text_to_clipboard_with_callback(&content, move |cb| {
                        egui_ctx.data_mut(|d| {
                            let status = match cb {
                                Ok(()) => ActionBarExportStatus {
                                    message: format!("Copied {label} dataset to clipboard!"),
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
                    egui_plot::export::copy_text_to_clipboard(&content)
                }
            }
            #[cfg(test)]
            {
                let _ = ctx;
                Ok(())
            }
        };

        match res {
            Ok(()) => {
                self.export_queue.last_exported_content = Some(content.clone());
                self.export_queue.last_clipboard_content = Some(content);
                let msg = if cfg!(target_arch = "wasm32")
                    && self.export_queue.custom_clipboard_writer.is_none()
                {
                    format!("Copying {label} dataset to clipboard...")
                } else {
                    format!("Copied {label} dataset to clipboard!")
                };
                self.set_action_bar_status(msg, false);
            }
            Err(err) => {
                self.set_action_bar_status(format_clipboard_error(&err), true);
            }
        }
    }

    /// Dispatches all pending export requests accumulated in the action queue.
    pub fn dispatch_pending_exports(&mut self) {
        self.dispatch_pending_exports_inner(None);
    }

    /// Dispatches all pending export requests with an `egui::Context` for repaint scheduling and async callbacks.
    pub fn dispatch_pending_exports_with_ctx(&mut self, ctx: &egui::Context) {
        self.dispatch_pending_exports_inner(Some(ctx));
    }

    fn dispatch_pending_exports_inner(&mut self, ctx: Option<&egui::Context>) {
        if self.export_queue.is_empty() {
            return;
        }

        let requests = self.export_queue.drain_requests();

        let Some(tool_name) = self.active_tool.as_ref().map(|t| t.name()) else {
            self.set_action_bar_status(
                "No active simulation tool available for export.".to_string(),
                true,
            );
            if let Some(egui_ctx) = ctx {
                egui_ctx.request_repaint();
            }
            return;
        };
        let datasets = self.export_datasets_with_fallback();

        let has_points = datasets.iter().any(|(_, pts)| !pts.is_empty());
        if !has_points {
            self.set_action_bar_status(
                format!("No dataset points available to export from {tool_name}."),
                true,
            );
            if let Some(egui_ctx) = ctx {
                egui_ctx.request_repaint();
            }
            return;
        }

        let file_slug = tool_name.to_lowercase().replace(' ', "_");

        for req in requests {
            let content = match req.format {
                ExportFormat::Csv => egui_plot::export::format_csv(&datasets),
                ExportFormat::Json => egui_plot::export::format_json(&datasets),
            };

            match req.target {
                ExportTarget::File => self.dispatch_file_request(&file_slug, req.format, content),
                ExportTarget::Clipboard => {
                    self.dispatch_clipboard_request(req.format, content, ctx)
                }
            }
        }

        if let Some(egui_ctx) = ctx {
            egui_ctx.request_repaint();
        }
    }
}
