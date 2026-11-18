use super::*;
use eframe::egui;

struct FrameworkExportSource<'a> {
    active_tool: &'a mut Option<Box<dyn InteractiveTool>>,
    cached_tool_name: Option<&'static str>,
    cached_plot_datasets: &'a [(String, Vec<[f64; 2]>)],
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
        let file_slug = tool_name.to_lowercase().replace(' ', "_");
        (
            format!("{file_slug}_figure.png"),
            format!("{file_slug}_figure.svg"),
            format!("{file_slug}_animation.gif"),
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
            self.sync_active_tool_state_after_preset();
            ctx.request_repaint();
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
            ui.label(egui::RichText::new("📊 Export:").strong());

            let tool_name = self
                .active_tool
                .as_ref()
                .map_or("simulation", |tool| tool.name());
            let file_slug = tool_name.to_lowercase().replace(' ', "_");

            self.render_dataset_export_buttons(ui);
            self.render_action_bar_export_status(ctx, ui);

            ui.separator();
            if ui
                .button("🎨 Starter Gallery")
                .on_hover_text("Toggle starter template gallery overlay")
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

            self.render_preset_action_bar_controls(ui, &file_slug);
        });
    }

    fn render_dataset_export_buttons(&mut self, ui: &mut egui::Ui) {
        if ui
            .button("Export CSV")
            .on_hover_text("Export dataset points to CSV file")
            .clicked()
        {
            self.queue_export(ExportFormat::Csv);
        }

        if ui
            .button("Copy CSV")
            .on_hover_text("Copy dataset points as CSV to clipboard")
            .clicked()
        {
            self.queue_clipboard_export();
        }

        if ui
            .button("Export JSON")
            .on_hover_text("Export dataset points to JSON file")
            .clicked()
        {
            self.queue_export(ExportFormat::Json);
        }

        if ui
            .button("Export PNG Image")
            .on_hover_text("Configure and render high-resolution PNG figure")
            .clicked()
        {
            self.open_export_modal_for_active_tool();
        }
    }

    fn render_action_bar_export_status(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        self.poll_action_bar_clipboard_status(ctx);

        let now = ctx.input(|i| i.time);
        let mut clear_status = false;

        if let Some(ref mut status) = self.export_queue.status {
            if !status.is_error {
                let exp = *status.expires_at_time.get_or_insert(now + 3.0);
                if now > 0.0 && now >= exp {
                    clear_status = true;
                } else {
                    let remaining = (exp - now).max(0.05);
                    ctx.request_repaint_after(std::time::Duration::from_secs_f64(remaining));
                    ui.colored_label(egui::Color32::LIGHT_GREEN, format!("✓ {}", status.message));
                    if ui
                        .small_button("✖")
                        .on_hover_text("Dismiss notification")
                        .clicked()
                    {
                        clear_status = true;
                    }
                }
            } else {
                ui.colored_label(egui::Color32::RED, format!("⚠️ {}", status.message));
                if ui
                    .small_button("✖")
                    .on_hover_text("Dismiss warning")
                    .clicked()
                {
                    clear_status = true;
                }
            }
        }

        if clear_status {
            self.export_queue.status = None;
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
            .on_hover_text("Import parameter preset from a JSON file")
            .clicked()
        {
            self.preset_dialog.trigger_import();
        }

        if ui
            .button("Export Preset JSON")
            .on_hover_text("Export current parameter preset to a JSON file")
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

        if let Some(ref msg) = self.preset_dialog.toast_info {
            ui.separator();
            ui.colored_label(egui::Color32::LIGHT_GREEN, format!("ℹ️ {}", msg));
            if ui
                .small_button("✖")
                .on_hover_text("Dismiss notification")
                .clicked()
            {
                self.preset_dialog.toast_info = None;
            }
        }

        if let Some(ref err) = self.preset_dialog.toast_error {
            ui.separator();
            ui.colored_label(egui::Color32::RED, format!("⚠️ {}", err));
            if ui
                .small_button("✖")
                .on_hover_text("Dismiss warning")
                .clicked()
            {
                self.preset_dialog.toast_error = None;
            }
        }
    }

    pub(super) fn show_export_modal_dialog(&mut self, ctx: &egui::Context, id_source: &str) {
        let tool_name = self
            .active_tool
            .as_ref()
            .map_or("simulation", |tool| tool.name());
        let (png_filename, svg_filename, gif_filename) = Self::export_filenames_for_tool(tool_name);
        let window_title = format!("Export High-Res Plot - {id_source}");
        let scope_id = self.export_modal_scope_id(id_source);
        let config = egui_plot::export_modal::ExportModalConfig {
            window_title: &window_title,
            id: scope_id,
            png_filename: &png_filename,
            svg_filename: &svg_filename,
            gif_filename: &gif_filename,
        };
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
        };
        egui_plot::export_modal::show_export_modal(ctx, bindings, &config, &mut source);
    }
}
