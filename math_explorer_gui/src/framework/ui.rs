use super::*;
use eframe::egui;

impl SimulationFramework {
    pub(super) fn show_theory_portal(&self, ctx: &egui::Context, id_source: &str) {
        if !self.show_theory_portal {
            return;
        }
        egui::SidePanel::right(format!("{}_theory_portal", id_source))
            .resizable(true)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    if let Some(tool) = &self.active_tool {
                        ui.heading(format!("Theory: {}", tool.name()));
                        ui.separator();

                        let theory = tool.theory();
                        // Requirement 4 & Acceptance Criteria 3: On-demand registration of bibliographic details when active/loaded
                        scientific_metadata::citation_registry::CitationRegistry::register(
                            tool.name().to_string(),
                            theory.theory_citation(),
                        );
                        // Requirement 5: Screen readers can successfully navigate the theory text and citations
                        use crate::accessibility::AccessibleHoverText;
                        use crate::reflective_ui::render_theory_summary_with_export;

                        render_theory_summary_with_export(ui, &theory.theory_description(), None)
                            .accessible_hover_text("Theoretical background description");

                        ui.separator();
                        ui.heading("Citations");
                        render_theory_summary_with_export(ui, &theory.theory_citation(), None)
                            .accessible_hover_text("Academic citations");

                        let available = theory.available_descriptions();
                        if !available.is_empty() {
                            ui.separator();
                            ui.heading("Additional Context");
                            for (key, desc) in available {
                                let context_entry = format!("{}: {}", key, desc);
                                render_theory_summary_with_export(ui, &context_entry, None)
                                    .accessible_hover_text(format!(
                                        "Additional context for {}",
                                        key
                                    ));
                            }
                        }
                    }
                });
            });
    }

    pub(super) fn render_search_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let text_edit =
                egui::TextEdit::singleline(&mut self.search_query).hint_text("🔍 Filter tools...");
            let response = ui.add(text_edit);

            if !self.search_query.is_empty()
                && ui.button("❌").on_hover_text("Clear search").clicked()
            {
                self.search_query.clear();
            }

            if ui.input(|i| i.key_pressed(egui::Key::Escape)) && response.has_focus() {
                self.search_query.clear();
            }
        });
    }

    #[allow(clippy::too_many_lines)]
    pub(super) fn render_filtered_tools(&mut self, ui: &mut egui::Ui) {
        let query = self.search_query.trim().to_lowercase();
        let filtered_tools: Vec<(usize, &&'static ToolMetadata)> = self
            .available_tools
            .iter()
            .enumerate()
            .filter(|(_, meta)| {
                if query.is_empty() {
                    true
                } else {
                    meta.name.to_lowercase().contains(&query)
                        || meta.tags.iter().any(|t| t.to_lowercase().contains(&query))
                }
            })
            .collect();

        let count = filtered_tools.len();

        if self.last_announced_count != Some(count) || self.last_announced_query != query {
            if !query.is_empty() {
                let msg = if count == 1 {
                    "1 tool matches filter".to_string()
                } else {
                    format!("{} tools match filter", count)
                };
                crate::accessibility::announce_status(&msg);
            }
            self.last_announced_count = Some(count);
            self.last_announced_query = query;
        }

        if filtered_tools.is_empty() {
            ui.label(format!("No tools match '{}'", self.search_query));
            if ui.button("Clear search").clicked() {
                self.search_query.clear();
            }
        } else {
            let mut tool_to_select = None;
            let mut open_portal_for_tool = None;
            egui::ScrollArea::vertical().show(ui, |ui| {
                for &(i, meta) in &filtered_tools {
                    ui.horizontal(|ui| {
                        let is_selected = self.selected_tool_index == Some(i);
                        if ui.selectable_label(is_selected, meta.name).clicked() {
                            tool_to_select = Some(i);
                        }
                        let tool_help_btn = ui
                            .add(egui::Button::new("❓").small())
                            .accessible_hover_text(format!(
                                "Open Theory Context Portal for {}",
                                meta.name
                            ));
                        if tool_help_btn.clicked() {
                            open_portal_for_tool = Some(i);
                        }
                    });
                }
            });
            if let Some(idx) = open_portal_for_tool {
                self.select_tool(idx);
                self.show_theory_portal = true;
                ui.ctx().data_mut(|d| {
                    d.insert_temp(egui::Id::new("SHOW_THEORY_PORTAL"), true);
                });
            } else if let Some(idx) = tool_to_select {
                self.select_tool(idx);
            }
        }
    }

    pub(super) fn show_side_panel(&mut self, ctx: &egui::Context, id_source: &str) {
        egui::SidePanel::right(format!("{}_tool_selector", id_source))
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Tools");
                    let help_btn = ui
                        .add(egui::Button::new("❓").small())
                        .accessible_hover_text("Toggle Theory Context Portal");
                    if help_btn.clicked() {
                        self.show_theory_portal = !self.show_theory_portal;
                        ctx.data_mut(|d| {
                            d.insert_temp(
                                egui::Id::new("SHOW_THEORY_PORTAL"),
                                self.show_theory_portal,
                            )
                        });
                    }
                });
                self.render_search_bar(ui);
                ui.separator();
                self.render_filtered_tools(ui);
                ui.separator();
                if ui
                    .checkbox(&mut self.show_theory_portal, "Theory Context Portal")
                    .changed()
                {
                    ctx.data_mut(|d| {
                        d.insert_temp(egui::Id::new("SHOW_THEORY_PORTAL"), self.show_theory_portal)
                    });
                }
            });
    }

    pub(super) fn show_quick_start_cards(&mut self, ui: &mut egui::Ui) {
        if self.available_tools.is_empty() {
            ui.vertical_centered(|ui| {
                ui.add_space(40.0);
                ui.heading("No Tools Available");
                ui.label("There are currently no interactive tools available for this domain.");
            });
            return;
        }

        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.vertical(|ui| {
                ui.add_space(20.0);
                ui.heading("Quick-Start Launchpad");
                ui.label("Select a tool below to begin interactive simulation and analysis:");
                ui.add_space(10.0);
                ui.separator();
                ui.add_space(15.0);

                let mut tool_to_select = None;

                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(12.0, 12.0);
                    for (i, meta) in self.available_tools.iter().enumerate() {
                        egui::Frame::group(ui.style())
                            .inner_margin(12.0)
                            .corner_radius(8.0)
                            .show(ui, |ui| {
                                ui.set_width(220.0);
                                ui.vertical(|ui| {
                                    ui.heading(meta.name);
                                    ui.add_space(6.0);
                                    if !meta.tags.is_empty() {
                                        ui.horizontal_wrapped(|ui| {
                                            for tag in meta.tags {
                                                let text = egui::RichText::new(format!("#{}", tag))
                                                    .small()
                                                    .color(ui.visuals().weak_text_color());
                                                ui.label(text);
                                            }
                                        });
                                        ui.add_space(8.0);
                                    }
                                    if ui.button("Launch Tool").clicked() {
                                        tool_to_select = Some(i);
                                    }
                                });
                            });
                    }
                });

                if let Some(idx) = tool_to_select {
                    self.select_tool(idx);
                }
            });
        });
    }

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
