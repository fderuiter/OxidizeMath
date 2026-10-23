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
}
