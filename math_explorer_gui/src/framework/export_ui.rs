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

    pub(super) fn show_export_action_bar(&mut self, ctx: &egui::Context, id_source: &str) {
        if self.active_tool.is_none() {
            return;
        }

        let imported = self
            .active_tool
            .as_mut()
            .is_some_and(|tool| self.preset_dialog.poll_import_tool(tool.as_mut()));
        if imported {
            self.sync_active_baseline();
        }

        egui::TopBottomPanel::top(format!("{}_export_bar", id_source))
            .exact_height(32.0)
            .show(ctx, |ui| {
                self.render_export_bar_row(ui, ctx);
            });
    }

    fn render_export_bar_row(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 8.0;
            let file_slug = self.render_dataset_and_figure_buttons(ui);
            self.render_action_bar_export_status(ctx, ui);
            ui.separator();
            self.render_starter_gallery_toggle(ctx, ui);
            self.render_preset_action_bar_controls(ui, &file_slug);
        });
    }

    fn render_dataset_and_figure_buttons(&mut self, ui: &mut egui::Ui) -> String {
        ui.label(egui::RichText::new("📊 Export:").strong());
        let (tool_name, caps) = if let Some(tool) = &self.active_tool {
            (tool.name(), tool.export_capabilities())
        } else {
            ("simulation", ExportCapabilities::none())
        };
        let file_slug = tool_name.to_lowercase().replace(' ', "_");

        let csv_tip = if caps.dataset {
            "Export dataset points to CSV file"
        } else {
            caps.dataset_disabled_reason
        };
        if ui
            .add_enabled(caps.dataset, egui::Button::new("Export CSV"))
            .accessible_hover_text(csv_tip)
            .clicked()
        {
            self.queue_export(ExportFormat::Csv);
        }

        let copy_tip = if caps.dataset {
            "Copy dataset points as CSV to clipboard"
        } else {
            caps.dataset_disabled_reason
        };
        if ui
            .add_enabled(caps.dataset, egui::Button::new("Copy CSV"))
            .accessible_hover_text(copy_tip)
            .clicked()
        {
            self.queue_clipboard_export();
        }

        let json_tip = if caps.dataset {
            "Export dataset points to JSON file"
        } else {
            caps.dataset_disabled_reason
        };
        if ui
            .add_enabled(caps.dataset, egui::Button::new("Export JSON"))
            .accessible_hover_text(json_tip)
            .clicked()
        {
            self.queue_export(ExportFormat::Json);
        }

        let fig_tip = if caps.figure {
            "Configure and render high-resolution PNG figure"
        } else {
            caps.figure_disabled_reason
        };
        if ui
            .add_enabled(caps.figure, egui::Button::new("Export PNG Image"))
            .accessible_hover_text(fig_tip)
            .clicked()
        {
            self.open_export_modal_for_active_tool();
        }
        file_slug
    }

    fn render_starter_gallery_toggle(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        if ui
            .button("🎨 Starter Gallery")
            .accessible_hover_text("Toggle starter template gallery overlay")
            .clicked()
        {
            self.show_starter_gallery = !self.show_starter_gallery;
            ctx.data_mut(|d| {
                d.insert_temp(
                    egui::Id::new("SHOW_STARTER_GALLERY"),
                    self.show_starter_gallery,
                )
            });
        }
    }

    fn poll_action_bar_clipboard_status(&mut self, ctx: &egui::Context) {
        let status_id = egui::Id::new(("framework_action_bar_clipboard_status", self.instance_id));
        if let Some(async_status) =
            ctx.data_mut(|d| d.remove_temp::<ActionBarExportStatus>(status_id))
        {
            let sev = if async_status.is_error {
                ExportStatusSeverity::Error
            } else {
                ExportStatusSeverity::Success
            };
            self.set_export_status(sev, async_status.message);
        }
    }

    fn render_action_bar_export_status(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        self.poll_action_bar_clipboard_status(ctx);

        let now = ctx.input(|i| i.time);
        let mut clear_status = false;
        let mut dismissed_by_user = false;

        if let Some(ref mut status) = self.export_queue.status {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            if !status.is_error {
                let exp = *status.expires_at_time.get_or_insert(now + 3.0);
                if now > 0.0 && now >= exp {
                    clear_status = true;
                } else {
                    let remaining = (exp - now).max(0.05);
                    ctx.request_repaint_after(std::time::Duration::from_secs_f64(remaining));
                    if !self.show_export_modal {
                        ui.colored_label(palette.success, format!("✓ {}", status.message));
                        if ui
                            .small_button("✖")
                            .accessible_hover_text("Dismiss notification")
                            .clicked()
                        {
                            clear_status = true;
                            dismissed_by_user = true;
                        }
                    }
                }
            } else if !self.show_export_modal {
                let err_color = if self.export_status_severity == ExportStatusSeverity::Warning {
                    palette.warning
                } else {
                    palette.error
                };
                ui.colored_label(err_color, format!("⚠️ {}", status.message));
                if ui
                    .small_button("✖")
                    .accessible_hover_text("Dismiss warning")
                    .clicked()
                {
                    clear_status = true;
                    dismissed_by_user = true;
                }
            }
        }

        if clear_status {
            self.export_queue.status = None;
            if dismissed_by_user {
                self.export_status_message = None;
            }
        }
    }

    fn export_active_preset_json(&mut self, file_slug: &str) {
        if let Some(tool) = &self.active_tool {
            match tool.export_preset_json() {
                Ok(json_str) => {
                    let filename = format!("{}_preset.json", file_slug);
                    self.preset_dialog.trigger_export(json_str, &filename);
                }
                Err(err) => {
                    self.preset_dialog.record_import_error(err);
                }
            }
        }
    }

    fn render_preset_action_bar_controls(&mut self, ui: &mut egui::Ui, file_slug: &str) {
        ui.separator();
        ui.label(egui::RichText::new("⚙ Presets:").strong());

        if ui
            .button("Import Preset JSON")
            .accessible_hover_text("Import parameter preset from a JSON file")
            .clicked()
        {
            self.preset_dialog.trigger_import();
        }

        if ui
            .button("Export Preset JSON")
            .accessible_hover_text("Export current parameter preset to a JSON file")
            .clicked()
        {
            self.export_active_preset_json(file_slug);
        }

        if ui
            .button("↻ Reset Parameters")
            .accessible_hover_text("Restore default baseline parameters")
            .clicked()
        {
            self.reset_parameters();
            crate::accessibility::announce_status("Parameters reset to baseline defaults");
            ui.ctx().request_repaint();
        }

        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        if let Some(ref msg) = self.preset_dialog.toast_info {
            ui.separator();
            ui.colored_label(palette.success, format!("ℹ️ {}", msg));
            if ui
                .small_button("✖")
                .accessible_hover_text("Dismiss notification")
                .clicked()
            {
                self.preset_dialog.toast_info = None;
            }
        }

        if let Some(ref err) = self.preset_dialog.toast_error {
            ui.separator();
            ui.colored_label(palette.error, format!("⚠️ {}", err));
            if ui
                .small_button("✖")
                .accessible_hover_text("Dismiss warning")
                .clicked()
            {
                self.preset_dialog.toast_error = None;
            }
        }
    }

    pub(super) fn show_export_modal_dialog(&mut self, ctx: &egui::Context, id_source: &str) {
        let scope_id = self.export_modal_scope_id(id_source);
        self.sync_state.last_modal_scope_id = Some(scope_id);

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
