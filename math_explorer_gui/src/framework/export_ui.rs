use super::*;
use eframe::egui;

impl SimulationFramework {
    pub(super) fn show_export_action_bar(&mut self, ctx: &egui::Context, id_source: &str) {
        if self.active_tool.is_none() {
            return;
        }

        egui::TopBottomPanel::top(format!("{}_export_bar", id_source))
            .exact_height(32.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 8.0;
                    ui.label(egui::RichText::new("📊 Export:").strong());

                    let tool_name = if let Some(tool) = &self.active_tool {
                        tool.name()
                    } else {
                        "simulation"
                    };

                    let file_slug = tool_name.to_lowercase().replace(' ', "_");

                    if ui
                        .button("Export CSV")
                        .on_hover_text("Export dataset points to CSV file")
                        .clicked()
                    {
                        if let Some(tool) = &self.active_tool {
                            let datasets = tool.export_datasets();
                            let filename = format!("{}_dataset.csv", file_slug);
                            egui_plot::export::export_csv_filename(&datasets, &filename);
                        }
                    }

                    if ui
                        .button("Export JSON")
                        .on_hover_text("Export dataset points to JSON file")
                        .clicked()
                    {
                        if let Some(tool) = &self.active_tool {
                            let datasets = tool.export_datasets();
                            let filename = format!("{}_dataset.json", file_slug);
                            egui_plot::export::export_json_filename(&datasets, &filename);
                        }
                    }

                    if ui
                        .button("Export PNG Image")
                        .on_hover_text("Configure and render high-resolution PNG figure")
                        .clicked()
                    {
                        if let Some(tool) = &self.active_tool {
                            if let Some(custom_opts) = tool.export_figure_options() {
                                self.export_options = custom_opts;
                            }
                        }
                        self.show_export_modal = true;
                    }
                });
            });
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

    fn render_modal_save_png(&mut self, ui: &mut egui::Ui) {
        if ui.button("💾 Save PNG").clicked() {
            if let Some(tool) = &self.active_tool {
                let datasets = tool.export_datasets();
                let file_slug = tool.name().to_lowercase().replace(' ', "_");
                let filename = format!("{}_figure.png", file_slug);
                match egui_plot::export::export_custom_png(&datasets, &self.export_options) {
                    Ok(png_bytes) => {
                        egui_plot::export::save_file_dialog(
                            &filename,
                            &png_bytes,
                            "PNG Image",
                            &["png"],
                        );
                        self.export_status_message =
                            Some("PNG image saved successfully.".to_string());
                    }
                    Err(e) => {
                        self.export_status_message = Some(format!("Export failed: {e}"));
                    }
                }
            }
        }
    }

    fn render_modal_save_gif(&mut self, ui: &mut egui::Ui) {
        if ui.button("🎥 Save GIF").clicked() {
            if let Some(tool) = &mut self.active_tool {
                let frames = tool.sample_animation_frames(self.export_gif_frames as usize);
                let file_slug = tool.name().to_lowercase().replace(' ', "_");
                let filename = format!("{}_animation.gif", file_slug);
                match egui_plot::export::export_custom_gif(
                    &frames,
                    &self.export_options,
                    self.export_gif_fps,
                ) {
                    Ok(gif_bytes) => {
                        egui_plot::export::save_file_dialog(
                            &filename,
                            &gif_bytes,
                            "GIF Animation",
                            &["gif"],
                        );
                        self.export_status_message =
                            Some("GIF animation saved successfully.".to_string());
                    }
                    Err(e) => {
                        self.export_status_message = Some(format!("GIF export failed: {e}"));
                    }
                }
            }
        }
    }

    fn render_modal_copy_clipboard(&mut self, ui: &mut egui::Ui) {
        if ui.button("📋 Copy to Clipboard").clicked() {
            if let Some(tool) = &self.active_tool {
                let datasets = tool.export_datasets();
                match egui_plot::export::export_custom_image(&datasets, &self.export_options) {
                    Ok(color_img) => match egui_plot::export::copy_image_to_clipboard(&color_img) {
                        Ok(_) => {
                            self.export_status_message =
                                Some("Copied plot image to clipboard!".to_string());
                        }
                        Err(e) => {
                            self.export_status_message =
                                Some(format!("Clipboard copy failed: {e}"));
                        }
                    },
                    Err(e) => {
                        self.export_status_message = Some(format!("Render failed: {e}"));
                    }
                }
            }
        }
    }

    fn render_export_modal_actions(&mut self, ui: &mut egui::Ui) {
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
            ui.label(egui::RichText::new(msg).color(egui::Color32::GREEN));
        }

        ui.separator();

        ui.horizontal(|ui| {
            self.render_modal_save_png(ui);
            self.render_modal_save_gif(ui);
            self.render_modal_copy_clipboard(ui);
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
