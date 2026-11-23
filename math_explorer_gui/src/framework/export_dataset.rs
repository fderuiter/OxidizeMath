//! Tabular 2D/3D dataset serialization and text clipboard dispatch for [`crate::framework::SimulationFramework`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use eframe::egui;

/// Formats a clipboard failure error with guidance toward disk export.
fn format_clipboard_error(err: &str) -> String {
    if err.contains("Export CSV") {
        format!("Clipboard unavailable: {err}")
    } else {
        format!("Clipboard unavailable ({err}). Please use Export CSV to save to disk.")
    }
}

impl SimulationFramework {
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

    pub(super) fn dispatch_dataset_export(
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
}
