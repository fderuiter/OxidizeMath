use super::*;
use std::path::PathBuf;
use std::sync::Arc;

/// Supported formats for deferred end-of-frame dataset and figure exports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExportFormat {
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
}

impl ExportFormat {
    /// Returns the standard file extension for this format.
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::Csv => "csv",
            Self::Json => "json",
            Self::Png => "png",
            Self::Svg => "svg",
            Self::Gif => "gif",
        }
    }

    /// Returns the file dialog filter description.
    pub const fn filter_name(&self) -> &'static str {
        match self {
            Self::Csv => "CSV File",
            Self::Json => "JSON File",
            Self::Png => "PNG Image",
            Self::Svg => "SVG Vector Image",
            Self::Gif => "GIF Animation",
        }
    }

    /// Returns the filename stem suffix (`dataset`, `figure`, or `animation`).
    pub const fn stem_suffix(&self) -> &'static str {
        match self {
            Self::Csv | Self::Json => "dataset",
            Self::Png | Self::Svg => "figure",
            Self::Gif => "animation",
        }
    }
}

/// Type alias for custom export file writer function.
pub type CustomExportWriter = Arc<dyn Fn(&str, &[u8]) + Send + Sync>;

/// Manages pending dataset and figure export requests deferred to the end of the rendering frame.
#[derive(Default)]
pub struct ExportQueue {
    requests: Vec<ExportFormat>,
    /// Optional directory to directly write exported files (useful in headless/automated test runs).
    pub destination_dir: Option<PathBuf>,
    /// Optional custom sink callback to receive exported filenames and file bytes.
    pub custom_writer: Option<CustomExportWriter>,
    /// Retains the last exported filename (for inspection and verification).
    pub last_exported_filename: Option<String>,
    /// Retains the last exported text file content (for CSV/JSON/SVG inspection).
    pub last_exported_content: Option<String>,
    /// Retains the last exported raw byte payload (for PNG/GIF/SVG/CSV/JSON verification).
    pub last_exported_bytes: Option<Vec<u8>>,
}

impl std::fmt::Debug for ExportQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportQueue")
            .field("requests", &self.requests)
            .field("destination_dir", &self.destination_dir)
            .field("last_exported_filename", &self.last_exported_filename)
            .field("last_exported_content", &self.last_exported_content)
            .field(
                "last_exported_bytes_len",
                &self.last_exported_bytes.as_ref().map(Vec::len),
            )
            .finish()
    }
}

impl ExportQueue {
    /// Queues an export request, preventing duplicates from rapid repeated clicks within the frame.
    pub fn push(&mut self, format: ExportFormat) -> bool {
        if self.requests.contains(&format) {
            return false;
        }
        self.requests.push(format);
        true
    }

    /// Returns `true` if there are no pending export requests.
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Returns the number of pending export requests.
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Returns a slice of the pending requests.
    pub fn requests(&self) -> &[ExportFormat] {
        &self.requests
    }

    /// Drains all pending export requests from the queue.
    pub fn drain(&mut self) -> Vec<ExportFormat> {
        std::mem::take(&mut self.requests)
    }

    /// Clears the queue without executing pending requests.
    pub fn clear(&mut self) {
        self.requests.clear();
    }
}

impl SimulationFramework {
    /// Records an export status message with visual severity styling and dispatches a screen-reader announcement.
    pub fn set_export_status(
        &mut self,
        severity: ExportStatusSeverity,
        message: impl Into<String>,
    ) {
        let msg = message.into();
        crate::accessibility::announce_status_with_priority(&msg, severity.aria_priority());
        self.export_status_severity = severity;
        self.export_status_message = Some(msg);
    }

    /// Queues a dataset or figure export request for deferred end-of-frame dispatch.
    pub fn queue_export(&mut self, format: ExportFormat) {
        self.export_queue.push(format);
    }

    /// Dispatches all pending export requests accumulated in the action queue after `tool.show` completes.
    pub fn dispatch_pending_exports(&mut self) {
        if self.export_queue.is_empty() {
            return;
        }
        let requests = self.export_queue.drain();
        if self.active_tool.is_none() {
            self.set_export_status(
                ExportStatusSeverity::Warning,
                "Export skipped: no active simulation tool selected.",
            );
            return;
        }
        for format in requests {
            self.dispatch_single_export(format);
        }
    }

    fn dispatch_single_export(&mut self, format: ExportFormat) {
        match format {
            ExportFormat::Csv | ExportFormat::Json => self.dispatch_dataset_export(format),
            ExportFormat::Png => self.dispatch_png_export(),
            ExportFormat::Svg => self.dispatch_svg_export(),
            ExportFormat::Gif => self.dispatch_gif_export(),
        }
    }

    fn dispatch_dataset_export(&mut self, format: ExportFormat) {
        let Some(tool) = &self.active_tool else {
            return;
        };
        let caps = tool.export_capabilities();
        if !caps.dataset {
            self.set_export_status(ExportStatusSeverity::Warning, caps.dataset_disabled_reason);
            return;
        }
        let tool_name = tool.name().to_string();
        let datasets: Vec<(String, Vec<[f64; 2]>)> = tool
            .export_datasets()
            .into_iter()
            .filter(|(_, pts)| pts.iter().any(|p| p[0].is_finite() && p[1].is_finite()))
            .collect();
        if !egui_plot::export_primitives::has_finite_dataset_points(&datasets) {
            self.set_export_status(
                ExportStatusSeverity::Error,
                format!(
                    "{} export rejected: {} produced no finite data points.",
                    format.filter_name(),
                    tool_name
                ),
            );
            return;
        }
        let content = match format {
            ExportFormat::Csv => egui_plot::export::format_csv(&datasets),
            ExportFormat::Json => egui_plot::export::format_json(&datasets),
            _ => return,
        };
        let filename = Self::build_export_filename(&tool_name, format);
        if let Err(err) =
            self.persist_export_bytes(format, &filename, content.as_bytes(), Some(content.clone()))
        {
            self.set_export_status(ExportStatusSeverity::Error, format!("Export failed: {err}"));
        } else {
            self.set_export_status(
                ExportStatusSeverity::Success,
                format!("{} saved successfully ({filename}).", format.filter_name()),
            );
        }
    }

    fn dispatch_png_export(&mut self) {
        let Some(tool) = &self.active_tool else {
            return;
        };
        let caps = tool.export_capabilities();
        if !caps.figure {
            self.set_export_status(ExportStatusSeverity::Warning, caps.figure_disabled_reason);
            return;
        }
        let filename = Self::build_export_filename(tool.name(), ExportFormat::Png);
        match render_tool_png(tool.as_ref(), &self.export_options) {
            Ok(png_bytes) => {
                if let Err(err) =
                    self.persist_export_bytes(ExportFormat::Png, &filename, &png_bytes, None)
                {
                    self.set_export_status(
                        ExportStatusSeverity::Error,
                        format!("Export failed: {err}"),
                    );
                } else {
                    self.set_export_status(
                        ExportStatusSeverity::Success,
                        format!("PNG image saved successfully ({filename})."),
                    );
                }
            }
            Err(err) => {
                self.set_export_status(ExportStatusSeverity::Error, format!("Export failed: {err}"))
            }
        }
    }

    fn dispatch_svg_export(&mut self) {
        let Some(tool) = &self.active_tool else {
            return;
        };
        let caps = tool.export_capabilities();
        if !caps.figure {
            self.set_export_status(ExportStatusSeverity::Warning, caps.figure_disabled_reason);
            return;
        }
        let filename = Self::build_export_filename(tool.name(), ExportFormat::Svg);
        match render_tool_svg(tool.as_ref(), &self.export_options) {
            Ok(svg_text) => {
                if let Err(err) = self.persist_export_bytes(
                    ExportFormat::Svg,
                    &filename,
                    svg_text.as_bytes(),
                    Some(svg_text.clone()),
                ) {
                    self.set_export_status(
                        ExportStatusSeverity::Error,
                        format!("Export failed: {err}"),
                    );
                } else {
                    self.set_export_status(
                        ExportStatusSeverity::Success,
                        format!("SVG image saved successfully ({filename})."),
                    );
                }
            }
            Err(err) => {
                self.set_export_status(ExportStatusSeverity::Error, format!("Export failed: {err}"))
            }
        }
    }

    fn dispatch_gif_export(&mut self) {
        let Some(tool) = &mut self.active_tool else {
            return;
        };
        let caps = tool.export_capabilities();
        if !caps.figure {
            let reason = caps.figure_disabled_reason;
            self.set_export_status(ExportStatusSeverity::Warning, reason);
            return;
        }
        let filename = Self::build_export_filename(tool.name(), ExportFormat::Gif);
        let frames = self.export_gif_frames as usize;
        let fps = self.export_gif_fps;
        let opts = self.export_options.clone();
        match render_tool_gif(tool.as_mut(), &opts, frames, fps) {
            Ok(gif_bytes) => {
                if let Err(err) =
                    self.persist_export_bytes(ExportFormat::Gif, &filename, &gif_bytes, None)
                {
                    self.set_export_status(
                        ExportStatusSeverity::Error,
                        format!("GIF export failed: {err}"),
                    );
                } else {
                    self.set_export_status(
                        ExportStatusSeverity::Success,
                        format!("GIF animation saved successfully ({filename})."),
                    );
                }
            }
            Err(err) => self.set_export_status(
                ExportStatusSeverity::Error,
                format!("GIF export failed: {err}"),
            ),
        }
    }

    fn build_export_filename(tool_name: &str, format: ExportFormat) -> String {
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
            {
                let _ = format;
            }
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
