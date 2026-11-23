//! Dataset (`CSV`, `JSON`) and text-clipboard export serialization for [`super::SimulationFramework`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use eframe::egui;

pub(super) struct FrameworkExportSource<'a> {
    pub(super) active_tool: &'a mut Option<Box<dyn InteractiveTool>>,
    pub(super) cached_tool_name: Option<&'static str>,
    pub(super) cached_plot_datasets: &'a [(String, Vec<[f64; 2]>)],
    pub(super) export_queue: &'a mut ExportQueue,
    pub(super) status_color: egui::Color32,
    pub(super) ctx: egui::Context,
}

impl egui_plot::export_modal::ExportDataSource for FrameworkExportSource<'_> {
    fn export_datasets(&mut self) -> Vec<egui_plot::export::ExportDataset> {
        let Some(tool) = self.active_tool.as_ref() else {
            return Vec::new();
        };
        let datasets = tool.export_datasets();
        if datasets.is_empty()
            && self.cached_tool_name == Some(tool.name())
            && !self.cached_plot_datasets.is_empty()
        {
            self.cached_plot_datasets.to_vec()
        } else {
            datasets
        }
    }

    fn sample_gif_frames(&mut self, num_frames: usize) -> Vec<egui_plot::export::DatasetFrame> {
        let step_count = num_frames.max(1);
        let Some(tool) = self.active_tool.as_mut() else {
            return Vec::new();
        };
        let frames = tool.sample_animation_frames(step_count);
        if frames.is_empty()
            && self.cached_tool_name == Some(tool.name())
            && !self.cached_plot_datasets.is_empty()
        {
            egui_plot::export::sample_progressive_frames(self.cached_plot_datasets, step_count)
        } else {
            frames
        }
    }

    fn figure_capability(&self) -> (bool, &'static str) {
        if let Some(tool) = self.active_tool.as_ref() {
            let caps = tool.export_capabilities();
            (caps.figure, caps.figure_disabled_reason)
        } else {
            let caps = ExportCapabilities::none();
            (caps.figure, caps.figure_disabled_reason)
        }
    }

    fn status_color(&self) -> Option<egui::Color32> {
        Some(self.status_color)
    }

    fn intercept_action(&mut self, action: egui_plot::ExportModalAction) -> bool {
        let (format, target) = match action {
            egui_plot::ExportModalAction::SavePng => (ExportFormat::Png, ExportTarget::File),
            egui_plot::ExportModalAction::SaveSvg => (ExportFormat::Svg, ExportTarget::File),
            egui_plot::ExportModalAction::SaveGif => (ExportFormat::Gif, ExportTarget::File),
            egui_plot::ExportModalAction::CopyClipboard => {
                (ExportFormat::Png, ExportTarget::Clipboard)
            }
        };
        self.export_queue.push_target(format, target);
        self.ctx.request_repaint();
        true
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

    pub(super) fn serialize_active_dataset_export(
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
}
