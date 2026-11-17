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

    /// Returns the file dialog filter description.
    pub fn filter_name(&self) -> &'static str {
        match self {
            Self::Csv => "CSV File",
            Self::Json => "JSON File",
        }
    }
}

/// Type alias for custom export file writer function.
pub type CustomExportWriter = Arc<dyn Fn(&str, &[u8]) + Send + Sync>;

/// Manages pending dataset export requests deferred to the end of the rendering frame.
#[derive(Default)]
pub struct ExportQueue {
    requests: Vec<ExportFormat>,
    /// Optional directory to directly write exported files (useful in headless/automated test runs).
    pub destination_dir: Option<PathBuf>,
    /// Optional custom sink callback to receive exported filenames and file content.
    pub custom_writer: Option<CustomExportWriter>,
    /// Retains the last exported filename (for inspection and verification).
    pub last_exported_filename: Option<String>,
    /// Retains the last exported file content string (for inspection and verification).
    pub last_exported_content: Option<String>,
}

impl std::fmt::Debug for ExportQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportQueue")
            .field("requests", &self.requests)
            .field("destination_dir", &self.destination_dir)
            .field("last_exported_filename", &self.last_exported_filename)
            .field("last_exported_content", &self.last_exported_content)
            .finish()
    }
}

impl ExportQueue {
    /// Queues an export request, preventing duplicates from rapid repeated clicks within the frame.
    /// Returns `true` if the request was added, or `false` if already queued.
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
    /// Queues a dataset export request for deferred end-of-frame dispatch.
    pub fn queue_export(&mut self, format: ExportFormat) {
        self.export_queue.push(format);
    }

    /// Dispatches all pending export requests accumulated in the action queue.
    /// This is executed at the end of the frame after active simulation tools have rendered
    /// and their plot datasets have been fully populated into `egui_plot`.
    pub fn dispatch_pending_exports(&mut self) {
        if self.export_queue.is_empty() {
            return;
        }

        let requests = self.export_queue.drain();

        let (tool_name, datasets) = if let Some(tool) = &self.active_tool {
            (tool.name(), tool.export_datasets())
        } else {
            return;
        };

        let file_slug = tool_name.to_lowercase().replace(' ', "_");

        for format in requests {
            let filename = format!("{file_slug}_dataset.{}", format.extension());
            let content = match format {
                ExportFormat::Csv => egui_plot::export::format_csv(&datasets),
                ExportFormat::Json => egui_plot::export::format_json(&datasets),
            };

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
    }
}
