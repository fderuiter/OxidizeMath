use super::*;
use std::path::PathBuf;
use std::sync::Arc;

/// Supported formats for deferred end-of-frame dataset, figure, and 3D exports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ExportFormat {
    /// Comma-separated values dataset format (`.csv`).
    Csv,
    /// JavaScript Object Notation dataset format (`.json`).
    Json,
    /// High-resolution Portable Network Graphics figure format (`.png`).
    Png,
    /// Scalable Vector Graphics figure format (`.svg`).
    Svg,
    /// Animated Graphics Interchange Format (`.gif`).
    Gif,
    /// Binary Stereolithography watertight 3D mesh format (`.stl`).
    Stl,
    /// Polygon File Format for colored 3D meshes and 3DGS splats (`.ply`).
    Ply,
    /// Wavefront 3D mesh format (`.obj`).
    Obj,
    /// Standalone WebGL2 (`#version 300 es`) fragment shader (`.frag`).
    GlslWebGl2,
    /// Standalone Shadertoy (`mainImage`) fragment shader (`.frag`).
    GlslShadertoy,
}

impl ExportFormat {
    const fn meta(self) -> (&'static str, &'static str, &'static str, &'static str) {
        match self {
            Self::Csv => ("csv", "CSV", "CSV File", "dataset"),
            Self::Json => ("json", "JSON", "JSON File", "dataset"),
            Self::Png => ("png", "PNG", "PNG Image", "figure"),
            Self::Svg => ("svg", "SVG", "SVG Vector Image", "figure"),
            Self::Gif => ("gif", "GIF", "GIF Animation", "animation"),
            Self::Stl => ("stl", "STL", "STL 3D Mesh", "mesh"),
            Self::Ply => ("ply", "PLY", "PLY 3D Model", "mesh"),
            Self::Obj => ("obj", "OBJ", "Wavefront OBJ Mesh", "mesh"),
            Self::GlslWebGl2 => (
                "frag",
                "WebGL2 GLSL",
                "WebGL2 GLSL Fragment Shader",
                "webgl2",
            ),
            Self::GlslShadertoy => (
                "frag",
                "Shadertoy GLSL",
                "Shadertoy Fragment Shader",
                "shadertoy",
            ),
        }
    }

    pub(super) const fn clipboard_item_noun(self) -> &'static str {
        match self {
            Self::GlslWebGl2 | Self::GlslShadertoy => "shader",
            _ => "dataset",
        }
    }

    /// Returns the standard file extension for this format.
    pub(crate) const fn extension(&self) -> &'static str {
        self.meta().0
    }

    /// Returns the uppercase label for display messages.
    pub(crate) const fn label(&self) -> &'static str {
        self.meta().1
    }

    /// Returns the file dialog filter description.
    pub(crate) const fn filter_name(&self) -> &'static str {
        self.meta().2
    }

    /// Returns the filename stem suffix (`dataset`, `figure`, or `animation`).
    pub(crate) const fn stem_suffix(&self) -> &'static str {
        self.meta().3
    }
}

/// Output destination target for deferred simulation dataset and figure export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ExportTarget {
    /// Save serialized dataset or figure to disk via file save dialog or browser download.
    File,
    /// Copy serialized dataset text or rendered figure image directly to the system clipboard.
    Clipboard,
}

/// A deferred export request combining format and output destination target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ExportRequest {
    /// Dataset or figure serialization format.
    pub(crate) format: ExportFormat,
    /// Output destination target.
    pub(crate) target: ExportTarget,
}

/// Inline status feedback for action-bar export operations.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ActionBarExportStatus {
    /// Human-readable confirmation or error message.
    pub(crate) message: String,
    /// Whether this status indicates an error or warning condition.
    pub(crate) is_error: bool,
    /// Optional `egui` timestamp (`ctx.input(|i| i.time)`) when temporary confirmation expires.
    pub(crate) expires_at_time: Option<f64>,
}

/// Type alias for custom export file writer function.
pub(crate) type CustomExportWriter = Arc<dyn Fn(&str, &[u8]) + Send + Sync>;

/// Type alias for custom clipboard text writer function (useful for deterministic testing).
pub(crate) type CustomClipboardWriter = Arc<dyn Fn(&str) -> Result<(), String> + Send + Sync>;

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
            (ExportFormat::Stl | ExportFormat::Ply | ExportFormat::Obj, target) => {
                self.dispatch_3d_export(req.format, target, ctx);
            }
            (ExportFormat::GlslWebGl2 | ExportFormat::GlslShadertoy, target) => {
                self.dispatch_shader_export(req.format, target, ctx);
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
}

#[cfg(test)]
#[path = "deferred_export_tests.rs"]
mod deferred_export_tests;

#[cfg(test)]
#[path = "deferred_export_audit_tests.rs"]
mod deferred_export_audit_tests;
