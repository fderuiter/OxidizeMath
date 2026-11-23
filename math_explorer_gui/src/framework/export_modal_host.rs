#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use eframe::egui;

struct FrameworkExportSource<'a> {
    active_tool: &'a mut Option<Box<dyn InteractiveTool>>,
    cached_tool_name: Option<&'static str>,
    cached_plot_datasets: &'a [(String, Vec<[f64; 2]>)],
    export_queue: &'a mut ExportQueue,
    status_color: egui::Color32,
    ctx: egui::Context,
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
    pub(super) fn open_export_modal_for_active_tool(&mut self) {
        if let Some(tool) = &self.active_tool {
            if let Some(custom_opts) = tool.export_figure_options() {
                self.export_bg_dark = custom_opts.background_color != egui::Color32::WHITE;
                self.export_options = custom_opts;
            }
        }
        self.show_export_modal = true;
    }

    pub(super) fn export_filenames_for_tool(tool_name: &str) -> (String, String, String) {
        (
            Self::build_export_filename(tool_name, ExportFormat::Png),
            Self::build_export_filename(tool_name, ExportFormat::Svg),
            Self::build_export_filename(tool_name, ExportFormat::Gif),
        )
    }

    pub(super) fn export_modal_scope_id(&self, id_source: &str) -> egui::Id {
        egui::Id::new(("framework_export", id_source, self.instance_id))
    }

    fn poll_export_modal_clipboard_status(&mut self, ctx: &egui::Context, scope_id: egui::Id) {
        if let Some(msg) =
            ctx.data_mut(|d| d.remove_temp::<String>(egui_plot::clipboard_status_id(scope_id)))
        {
            let sev = if msg.contains("failed") || msg.contains("unavailable") {
                ExportStatusSeverity::Error
            } else {
                ExportStatusSeverity::Success
            };
            self.set_export_status(sev, msg);
        }
    }

    pub(super) fn show_export_modal_dialog(&mut self, ctx: &egui::Context, id_source: &str) {
        let scope_id = self.export_modal_scope_id(id_source);
        self.sync_state.last_modal_scope_id = Some(scope_id);
        self.poll_export_modal_clipboard_status(ctx, scope_id);

        let tool_name = self
            .active_tool
            .as_ref()
            .map_or("simulation", |tool| tool.name());
        let (png_filename, svg_filename, gif_filename) = Self::export_filenames_for_tool(tool_name);
        let window_title = format!("Export High-Res Plot - {id_source}");
        let config = egui_plot::export_modal::ExportModalConfig {
            window_title: &window_title,
            id: scope_id,
            png_filename: &png_filename,
            svg_filename: &svg_filename,
            gif_filename: &gif_filename,
        };
        let status_color = self
            .export_status_severity
            .palette_color(&ctx.style().visuals);
        let bindings = egui_plot::export_modal::ExportModalBindings {
            open: &mut self.show_export_modal,
            options: &mut self.export_options,
            bg_dark: &mut self.export_bg_dark,
            gif_fps: &mut self.export_gif_fps,
            gif_frames: &mut self.export_gif_frames,
            status_message: &mut self.export_status_message,
        };
        let mut source = FrameworkExportSource {
            active_tool: &mut self.active_tool,
            cached_tool_name: self.sync_state.datasets_owner_tool,
            cached_plot_datasets: &self.current_valid_plot_datasets,
            export_queue: &mut self.export_queue,
            status_color,
            ctx: ctx.clone(),
        };
        egui_plot::export_modal::show_export_modal(ctx, bindings, &config, &mut source);
    }
}
