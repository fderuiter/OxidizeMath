use super::*;
use eframe::egui;

impl SimulationFramework {
    #[allow(clippy::too_many_lines)]
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
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 8.0;
                    ui.label(egui::RichText::new("📊 Export:").strong());

                    let (tool_name, caps) = if let Some(tool) = &self.active_tool {
                        (tool.name(), tool.export_capabilities())
                    } else {
                        ("simulation", ExportCapabilities::none())
                    };
                    let file_slug = tool_name.to_lowercase().replace(' ', "_");

                    if ui
                        .add_enabled(caps.dataset, egui::Button::new("Export CSV"))
                        .on_hover_text("Export dataset points to CSV file")
                        .on_disabled_hover_text(caps.dataset_disabled_reason)
                        .clicked()
                    {
                        self.queue_export(ExportFormat::Csv);
                    }

                    if ui
                        .add_enabled(caps.dataset, egui::Button::new("Export JSON"))
                        .on_hover_text("Export dataset points to JSON file")
                        .on_disabled_hover_text(caps.dataset_disabled_reason)
                        .clicked()
                    {
                        self.queue_export(ExportFormat::Json);
                    }

                    if ui
                        .add_enabled(caps.figure, egui::Button::new("Export PNG Image"))
                        .on_hover_text("Configure and render high-resolution figure")
                        .on_disabled_hover_text(caps.figure_disabled_reason)
                        .clicked()
                    {
                        if let Some(tool) = &self.active_tool {
                            if let Some(custom_opts) = tool.export_figure_options() {
                                self.export_options = custom_opts;
                            }
                        }
                        self.show_export_modal = true;
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
                    self.render_inline_export_status(ui);
                });
            });
    }

    fn render_inline_export_status(&mut self, ui: &mut egui::Ui) {
        if self.show_export_modal {
            return;
        }
        if let Some(ref msg) = self.export_status_message {
            ui.separator();
            ui.colored_label(self.export_status_severity.color(), msg);
            if ui
                .small_button("✖")
                .on_hover_text("Dismiss export notification")
                .clicked()
            {
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

    fn render_modal_aspect_combo(&mut self, ui: &mut egui::Ui, id_source: &str) {
        ui.horizontal(|ui| {
            ui.label("Aspect Ratio:");
            egui::ComboBox::from_id_salt(format!("{}_export_aspect", id_source))
                .selected_text(match self.export_options.aspect_ratio {
                    egui_plot::export::AspectRatioOption::Free => "Custom / Free",
                    egui_plot::export::AspectRatioOption::SixteenNine => "16:9 (Widescreen)",
                    egui_plot::export::AspectRatioOption::FourThree => "4:3 (Standard)",
                    egui_plot::export::AspectRatioOption::OneOne => "1:1 (Square)",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.export_options.aspect_ratio,
                        egui_plot::export::AspectRatioOption::SixteenNine,
                        "16:9 (Widescreen)",
                    );
                    ui.selectable_value(
                        &mut self.export_options.aspect_ratio,
                        egui_plot::export::AspectRatioOption::FourThree,
                        "4:3 (Standard)",
                    );
                    ui.selectable_value(
                        &mut self.export_options.aspect_ratio,
                        egui_plot::export::AspectRatioOption::OneOne,
                        "1:1 (Square)",
                    );
                    ui.selectable_value(
                        &mut self.export_options.aspect_ratio,
                        egui_plot::export::AspectRatioOption::Free,
                        "Custom / Free",
                    );
                });
        });
    }

    fn render_export_modal_settings(&mut self, ui: &mut egui::Ui, id_source: &str) {
        ui.horizontal(|ui| {
            ui.label("Resolution Scale:");
            ui.selectable_value(&mut self.export_options.scale, 1.0, "1x (1080p)");
            ui.selectable_value(&mut self.export_options.scale, 2.0, "2x (4K)");
            ui.selectable_value(&mut self.export_options.scale, 4.0, "4x");
        });

        self.render_modal_aspect_combo(ui, id_source);

        ui.horizontal(|ui| {
            ui.label("Line Thickness:");
            ui.add(egui::Slider::new(
                &mut self.export_options.line_thickness,
                0.5..=10.0,
            ));
        });

        ui.horizontal(|ui| {
            ui.label("Background Opacity:");
            ui.add(egui::Slider::new(
                &mut self.export_options.background_opacity,
                0.0..=1.0,
            ));
        });

        ui.horizontal(|ui| {
            ui.label("Background Color:");
            ui.selectable_value(&mut self.export_bg_dark, true, "Dark");
            ui.selectable_value(&mut self.export_bg_dark, false, "Light");
        });

        self.export_options.background_color = if self.export_bg_dark {
            egui::Color32::from_rgb(18, 18, 18)
        } else {
            egui::Color32::WHITE
        };

        ui.horizontal(|ui| {
            ui.label("GIF FPS:");
            ui.add(egui::Slider::new(&mut self.export_gif_fps, 1..=60));
        });

        ui.horizontal(|ui| {
            ui.label("GIF Frame Steps:");
            ui.add(egui::Slider::new(&mut self.export_gif_frames, 5..=60));
        });
    }

    fn queue_modal_figure_export(&mut self, ui: &egui::Ui, format: ExportFormat) {
        let Some(tool) = &self.active_tool else {
            self.set_export_status(
                ExportStatusSeverity::Warning,
                "Export skipped: no active simulation tool selected.",
            );
            return;
        };
        let caps = tool.export_capabilities();
        if !caps.figure {
            self.set_export_status(ExportStatusSeverity::Warning, caps.figure_disabled_reason);
            return;
        }
        self.queue_export(format);
        ui.ctx().request_repaint();
    }

    fn render_modal_save_buttons(&mut self, ui: &mut egui::Ui, caps: ExportCapabilities) {
        if ui
            .add_enabled(caps.figure, egui::Button::new("💾 Save PNG"))
            .on_hover_text("Queue high-resolution PNG render after active frame completes")
            .on_disabled_hover_text(caps.figure_disabled_reason)
            .clicked()
        {
            self.queue_modal_figure_export(ui, ExportFormat::Png);
        }

        if ui
            .add_enabled(caps.figure, egui::Button::new("💾 Save SVG"))
            .on_hover_text("Queue vector SVG render after active frame completes")
            .on_disabled_hover_text(caps.figure_disabled_reason)
            .clicked()
        {
            self.queue_modal_figure_export(ui, ExportFormat::Svg);
        }

        if ui
            .add_enabled(caps.figure, egui::Button::new("🎬 Save GIF"))
            .on_hover_text("Queue multi-frame GIF animation export after active frame completes")
            .on_disabled_hover_text(caps.figure_disabled_reason)
            .clicked()
        {
            self.queue_modal_figure_export(ui, ExportFormat::Gif);
        }
    }

    fn render_modal_copy_clipboard(&mut self, ui: &mut egui::Ui, caps: ExportCapabilities) {
        if !ui
            .add_enabled(caps.figure, egui::Button::new("📋 Copy to Clipboard"))
            .on_disabled_hover_text(caps.figure_disabled_reason)
            .clicked()
        {
            return;
        }
        let Some(tool) = &self.active_tool else {
            return;
        };
        match render_tool_color_image(tool.as_ref(), &self.export_options) {
            Ok(color_img) => {
                let ctx = ui.ctx().clone();
                let status_id = egui::Id::new("framework_export_clipboard_status");
                match egui_plot::export::copy_image_to_clipboard_with_callback(
                    &color_img,
                    move |res| {
                        ctx.data_mut(|d| {
                            let item = match res {
                                Ok(_) => (
                                    ExportStatusSeverity::Success,
                                    "Copied plot image to clipboard!".to_string(),
                                ),
                                Err(e) => (
                                    ExportStatusSeverity::Error,
                                    format!("Clipboard copy failed: {e}"),
                                ),
                            };
                            d.insert_temp(status_id, item);
                        });
                        ctx.request_repaint();
                    },
                ) {
                    Ok(_) => {
                        #[cfg(not(target_arch = "wasm32"))]
                        self.set_export_status(
                            ExportStatusSeverity::Success,
                            "Copied plot image to clipboard!",
                        );
                        #[cfg(target_arch = "wasm32")]
                        self.set_export_status(
                            ExportStatusSeverity::Success,
                            "Copying plot image to clipboard...",
                        );
                    }
                    Err(e) => self.set_export_status(
                        ExportStatusSeverity::Error,
                        format!("Clipboard copy failed: {e}"),
                    ),
                }
            }
            Err(e) => {
                self.set_export_status(ExportStatusSeverity::Error, format!("Render failed: {e}"))
            }
        }
    }

    fn render_export_modal_actions(&mut self, ui: &mut egui::Ui) {
        let status_id = egui::Id::new("framework_export_clipboard_status");
        if let Some((sev, msg)) = ui
            .ctx()
            .data_mut(|d| d.remove_temp::<(ExportStatusSeverity, String)>(status_id))
        {
            self.set_export_status(sev, msg);
        }
        let base_w = self.export_options.width.max(1);
        let base_h = self.export_options.height.max(1);
        let calc_w = (base_w as f32 * self.export_options.scale).round() as u32;
        let mut calc_h = (base_h as f32 * self.export_options.scale).round() as u32;
        if let Some(ratio) = self
            .export_options
            .aspect_ratio
            .ratio()
            .filter(|&r| r > 0.0)
        {
            calc_h = (calc_w as f32 / ratio).round() as u32;
        }

        if calc_w > egui_plot::export::MAX_EXPORT_DIMENSION
            || calc_h > egui_plot::export::MAX_EXPORT_DIMENSION
        {
            ui.label(
                egui::RichText::new(format!(
                    "⚠️ Capped at max dimension {}x{} px",
                    egui_plot::export::MAX_EXPORT_DIMENSION,
                    egui_plot::export::MAX_EXPORT_DIMENSION
                ))
                .color(egui::Color32::YELLOW),
            );
        } else {
            ui.label(format!("Target Canvas Size: {} x {} px", calc_w, calc_h));
        }

        if let Some(msg) = &self.export_status_message {
            ui.label(egui::RichText::new(msg).color(self.export_status_severity.color()));
        }

        ui.separator();

        let caps = self
            .active_tool
            .as_ref()
            .map(|t| t.export_capabilities())
            .unwrap_or_else(ExportCapabilities::none);

        ui.horizontal(|ui| {
            self.render_modal_save_buttons(ui, caps);
            self.render_modal_copy_clipboard(ui, caps);
        });
    }

    pub(super) fn show_export_modal_dialog(&mut self, ctx: &egui::Context, id_source: &str) {
        if !self.show_export_modal {
            return;
        }

        let mut modal_open = self.show_export_modal;
        egui::Window::new(format!("Export High-Res Plot - {}", id_source))
            .open(&mut modal_open)
            .resizable(true)
            .collapsible(false)
            .default_width(380.0)
            .show(ctx, |ui| {
                ui.label(egui::RichText::new("Offscreen High-Res Render Settings").strong());
                ui.separator();
                self.render_export_modal_settings(ui, id_source);
                ui.separator();
                self.render_export_modal_actions(ui);
            });

        self.show_export_modal = modal_open;
    }
}
