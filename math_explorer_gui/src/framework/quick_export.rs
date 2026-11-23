#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{ExportFormat, SimulationFramework};
use eframe::egui;

/// Programmatic export actions triggered from the global Command Palette on the active pane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuickExportAction {
    /// Queue CSV dataset export for the active tool.
    ExportCsv,
    /// Copy active dataset points as CSV to the clipboard.
    CopyCsv,
    /// Queue JSON dataset export for the active tool.
    ExportJson,
    /// Open high-resolution PNG/SVG/GIF figure export modal for the active tool.
    ExportPng,
    /// Export the active tool's parameter preset JSON envelope.
    ExportPresetJson,
}

impl SimulationFramework {
    /// Executes a Command Palette export action on the active simulation tool.
    pub fn trigger_quick_export(
        &mut self,
        ctx: &egui::Context,
        action: QuickExportAction,
    ) -> Result<String, String> {
        let Some(tool) = &self.active_tool else {
            return Err("No active simulation tool selected".to_string());
        };
        let tool_name = tool.name();
        let (file_slug, caps) = self.active_tool_slug_and_caps();

        match action {
            QuickExportAction::ExportCsv => {
                if !caps.dataset {
                    return Err(caps.dataset_disabled_reason.to_string());
                }
                self.queue_export(ExportFormat::Csv);
                ctx.request_repaint();
                Ok(format!("Queued CSV dataset export for {tool_name}"))
            }
            QuickExportAction::CopyCsv => {
                if !caps.dataset {
                    return Err(caps.dataset_disabled_reason.to_string());
                }
                self.queue_clipboard_export();
                ctx.request_repaint();
                Ok(format!("Queued CSV clipboard copy for {tool_name}"))
            }
            QuickExportAction::ExportJson => {
                if !caps.dataset {
                    return Err(caps.dataset_disabled_reason.to_string());
                }
                self.queue_export(ExportFormat::Json);
                ctx.request_repaint();
                Ok(format!("Queued JSON dataset export for {tool_name}"))
            }
            QuickExportAction::ExportPng => {
                if !caps.figure {
                    return Err(caps.figure_disabled_reason.to_string());
                }
                self.open_export_modal_for_active_tool();
                ctx.request_repaint();
                Ok(format!("Opened figure export dialog for {tool_name}"))
            }
            QuickExportAction::ExportPresetJson => {
                self.export_active_preset_json(&file_slug);
                ctx.request_repaint();
                if let Some(ref err) = self.preset_dialog.toast_error {
                    return Err(err.clone());
                }
                Ok(format!("Triggered preset JSON export for {tool_name}"))
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

    pub(super) fn persist_export_bytes(
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
