use super::*;
use eframe::egui;

struct FrameworkExportSource<'a>(&'a mut Option<Box<dyn InteractiveTool>>);

impl egui_plot::export_modal::ExportDataSource for FrameworkExportSource<'_> {
    fn export_datasets(&mut self) -> Vec<egui_plot::export::ExportDataset> {
        self.0
            .as_ref()
            .map(|tool| tool.export_datasets())
            .unwrap_or_default()
    }

    fn sample_gif_frames(&mut self, num_frames: usize) -> Vec<egui_plot::export::DatasetFrame> {
        self.0
            .as_mut()
            .map(|tool| tool.sample_animation_frames(num_frames))
            .unwrap_or_default()
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

    pub(super) fn export_modal_scope_id(id_source: &str) -> egui::Id {
        egui::Id::new(id_source).with("framework_export")
    }

    pub(super) fn show_export_action_bar(&mut self, ctx: &egui::Context, id_source: &str) {
        if self.active_tool.is_none() {
            return;
        }

        if let Some(tool) = &mut self.active_tool {
            if self.preset_dialog.poll_import_tool(tool.as_mut()) {
                if let Some(state) = tool.save_state() {
                    self.saved_tool_states
                        .insert(tool.name().to_string(), state);
                }
            }
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

            if ui
                .button("Export CSV")
                .on_hover_text("Export dataset points to CSV file")
                .clicked()
            {
                self.queue_export(ExportFormat::Csv);
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

    fn export_active_preset_json(&mut self, file_slug: &str) {
        if let Some(tool) = &self.active_tool {
            match tool.export_preset_json() {
                Ok(json_str) => {
                    let filename = format!("{}_preset.json", file_slug);
                    self.preset_dialog.trigger_export(json_str, &filename);
                }
                Err(err) => {
                    federated_registry::global_registry().emit(
                        federated_registry::TelemetryEvent {
                            source: "PresetSerialization".to_string(),
                            severity: federated_registry::Severity::Warning,
                            message: err.clone(),
                            metadata: std::collections::HashMap::new(),
                            thread_name: None,
                        },
                    );
                    self.preset_dialog.toast_error = Some(err);
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
        let config = egui_plot::export_modal::ExportModalConfig {
            window_title: &window_title,
            id: Self::export_modal_scope_id(id_source),
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
        let mut source = FrameworkExportSource(&mut self.active_tool);
        egui_plot::export_modal::show_export_modal(ctx, bindings, &config, &mut source);
    }
}
