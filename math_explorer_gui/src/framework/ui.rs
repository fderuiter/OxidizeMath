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

                        let theory_desc = theory.theory_description();
                        let latex_eq = theory.theory_latex_equation();
                        render_theory_summary_with_export(ui, &theory_desc, latex_eq.as_deref())
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

    pub(super) fn sync_active_tool_state_after_preset(&mut self) {
        let Some(tool) = &self.active_tool else {
            return;
        };
        egui_plot::clear_last_plot_datasets();
        let active_state = tool.save_state();
        if let Some(ref state_str) = active_state {
            self.saved_tool_states
                .insert(tool.name().to_string(), state_str.clone());
        }
        self.baseline_tool_state = active_state.clone();
        self.is_diverged = false;
        self.diagnostic_report = None;
        self.step_counter = 0;
        let datasets = tool.export_datasets();
        self.current_valid_plot_datasets = datasets.clone();
        self.last_valid_snapshot = Some(StateSnapshot {
            tool_state: active_state,
            plot_datasets: datasets,
            step_index: 0,
            timestamp: crate::web_io::PlatformInstant::now(),
        });
    }

    /// Launches a Starter Gallery card by selecting its target tool, importing
    /// and validating its preset JSON, updating framework state snapshots, or
    /// surfacing an error toast + telemetry warning on failure.
    pub fn apply_starter_card(
        &mut self,
        card_title: &str,
        tool_name: Option<&str>,
        preset_json: Option<&str>,
    ) -> Result<(), String> {
        if let Some(target_name) = tool_name {
            if let Some(pos) = self
                .available_tools
                .iter()
                .position(|t| t.name == target_name)
            {
                self.select_tool(pos);
            } else {
                let err = format!(
                    "Starter preset '{}' targets unavailable tool '{}'",
                    card_title, target_name
                );
                self.preset_dialog.record_import_error(err.clone());
                return Err(err);
            }
        }
        let Some(json_str) = preset_json else {
            return Ok(());
        };
        let Some(tool) = &mut self.active_tool else {
            let err = format!("No active tool available to launch '{}'", card_title);
            self.preset_dialog.record_import_error(err.clone());
            return Err(err);
        };
        match tool.import_preset_json(json_str) {
            Ok(()) => {
                self.preset_dialog.toast_error = None;
                self.sync_active_tool_state_after_preset();
                Ok(())
            }
            Err(err) => {
                let msg = format!("Failed to launch '{}': {}", card_title, err);
                self.preset_dialog.record_import_error(msg.clone());
                Err(msg)
            }
        }
    }

    pub(super) fn show_starter_gallery_if_open(&mut self, ctx: &egui::Context, id_source: &str) {
        if self.show_starter_gallery {
            let cards = crate::starter_gallery::get_starter_cards_for_domain(
                id_source,
                &self.available_tools,
            );
            let mut is_open = self.show_starter_gallery;
            if let Some(card) = crate::starter_gallery::show_starter_gallery_overlay(
                ctx,
                id_source,
                &cards,
                &mut is_open,
            ) {
                let _ = self.apply_starter_card(card.title, card.tool_name, card.preset_json);
                is_open = false;
                ctx.request_repaint();
            }
            self.show_starter_gallery = is_open;
            ctx.data_mut(|d| {
                d.insert_temp(
                    egui::Id::new("SHOW_STARTER_GALLERY"),
                    self.show_starter_gallery,
                )
            });
        }
    }

    #[allow(missing_docs, clippy::too_many_lines)]
    pub fn show(&mut self, ctx: &egui::Context, id_source: &str) {
        egui_plot::clear_last_plot_datasets();

        // Display recovery banner when solver divergence is detected
        if self.is_diverged {
            egui::TopBottomPanel::top(format!("{}_recovery_banner", id_source)).show(ctx, |ui| {
                if let Some(action) = render_state_guard_banner(
                    ui,
                    self.diagnostic_report.as_ref(),
                    self.last_valid_snapshot.is_some(),
                ) {
                    self.handle_state_guard_action(action);
                    ui.ctx().request_repaint();
                }
            });
        }

        // Update global input mode
        ctx.input(|i| {
            if i.any_touches() || i.multi_touch().is_some() {
                self.input_mode = InputMode::Touch;
            } else if i.pointer.is_moving() && !i.any_touches() {
                // Heuristic: if pointer is moving but no touches, likely mouse
                self.input_mode = InputMode::Mouse;
            }
        });
        ctx.data_mut(|d| {
            d.insert_temp(egui::Id::new("INPUT_MODE"), self.input_mode);
            d.insert_temp(
                egui::Id::new("INPUT_MODE_TOUCH"),
                self.input_mode == InputMode::Touch,
            );
            // Clear CMD_REGISTRY at start of frame
            d.insert_temp(
                egui::Id::new("CMD_REGISTRY"),
                egui_plot::commands::CommandRegistryData::default(),
            );
        });

        if let Some(portal_state) =
            ctx.data(|d| d.get_temp::<bool>(egui::Id::new("SHOW_THEORY_PORTAL")))
        {
            self.show_theory_portal = portal_state;
        } else {
            ctx.data_mut(|d| {
                d.insert_temp(egui::Id::new("SHOW_THEORY_PORTAL"), self.show_theory_portal)
            });
        }

        if let Some(gallery_state) =
            ctx.data(|d| d.get_temp::<bool>(egui::Id::new("SHOW_STARTER_GALLERY")))
        {
            self.show_starter_gallery = gallery_state;
        } else {
            ctx.data_mut(|d| {
                d.insert_temp(
                    egui::Id::new("SHOW_STARTER_GALLERY"),
                    self.show_starter_gallery,
                )
            });
        }

        self.show_side_panel(ctx, id_source);
        self.show_theory_portal(ctx, id_source);
        self.show_export_action_bar(ctx, id_source);
        self.show_export_modal_dialog(ctx, id_source);

        if !self.is_diverged {
            self.cache_valid_snapshot();
            if let Some(tool) = &mut self.active_tool {
                tool.show(ctx);
            }
            let _ = self.inspect_runtime_state();
            if self.is_diverged {
                ctx.request_repaint();
            }
        } else if let Some(tool) = &mut self.active_tool {
            // Solver execution halted: keep last valid plot data active in egui_plot
            if let Some(ref snapshot) = self.last_valid_snapshot {
                egui_plot::clear_last_plot_datasets();
                egui_plot::append_last_plot_datasets(&snapshot.plot_datasets);
            }
            tool.show(ctx);
        } else {
            egui::CentralPanel::default().show(ctx, |ui| {
                if self.available_tools.is_empty() {
                    ui.vertical_centered(|ui| {
                        ui.add_space(40.0);
                        ui.heading("No Tools Available");
                        ui.label(
                            "There are currently no interactive tools available for this domain.",
                        );
                    });
                }
            });
        }

        self.show_starter_gallery_if_open(ctx, id_source);

        if let Some(portal_state) =
            ctx.data(|d| d.get_temp::<bool>(egui::Id::new("SHOW_THEORY_PORTAL")))
        {
            self.show_theory_portal = portal_state;
        }

        self.dispatch_pending_exports();
    }
}
