use super::*;
use eframe::egui;

pub(crate) fn format_theory_context_key(key: &str) -> String {
    match key.trim() {
        "g_na" => "Sodium Conductance".to_string(),
        "g_k" => "Potassium Conductance".to_string(),
        "g_l" => "Leak Conductance".to_string(),
        "i_ext" => "External Stimulus Current".to_string(),
        "potential_type" => "Potential Profile".to_string(),
        "dt" => "Time Step".to_string(),
        "params" => "Model Parameters".to_string(),
        "viscosity" => "Kinematic Viscosity".to_string(),
        other if other.contains('_') => other
            .split('_')
            .filter(|s| !s.is_empty())
            .map(|w| {
                let mut chars = w.chars();
                match chars.next() {
                    Some(c) => format!("{}{}", c.to_uppercase(), chars.as_str()),
                    None => String::new(),
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
        other => {
            let mut chars = other.chars();
            match chars.next() {
                Some(c) if c.is_lowercase() => format!("{}{}", c.to_uppercase(), chars.as_str()),
                _ => other.to_string(),
            }
        }
    }
}

fn has_interactive_ui_input(ctx: &egui::Context) -> bool {
    ctx.input(|i| {
        i.pointer.any_down()
            || i.pointer.any_pressed()
            || i.pointer.any_released()
            || i.any_touches()
            || i.raw_scroll_delta != egui::Vec2::ZERO
            || i.smooth_scroll_delta != egui::Vec2::ZERO
            || (i.zoom_delta() - 1.0).abs() > f32::EPSILON
            || i.events.iter().any(|e| {
                !matches!(
                    e,
                    egui::Event::PointerMoved(_)
                        | egui::Event::PointerGone
                        | egui::Event::WindowFocused(_)
                )
            })
    })
}

impl SimulationFramework {
    pub(super) fn show_theory_portal(&self, ui: &mut egui::Ui, id_source: &str) {
        if !self.show_theory_portal {
            return;
        }
        let panel_id = scoped_panel_id(ui, format!("{id_source}_theory_portal"));
        PaneSidePanel::right(panel_id)
            .resizable(true)
            .show_inside(ui, |ui| {
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

                    let mut entries: Vec<(String, String)> = theory
                        .available_descriptions()
                        .into_iter()
                        .filter(|(k, _)| !k.trim().eq_ignore_ascii_case("default"))
                        .collect();
                    entries.sort_by(|a, b| a.0.cmp(&b.0));
                    if !entries.is_empty() {
                        ui.separator();
                        ui.heading("Additional Context");
                        for (key, desc) in entries {
                            let label = format_theory_context_key(&key);
                            let context_entry = format!("{label}: {desc}");
                            render_theory_summary_with_export(ui, &context_entry, None)
                                .accessible_hover_text(format!("Additional context for {label}"));
                        }
                    }
                }
            });
    }

    pub(super) fn render_search_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let text_edit =
                egui::TextEdit::singleline(&mut self.search_query).hint_text("🔍 Filter tools...");
            let response = ui.add(text_edit);

            if !self.search_query.is_empty()
                && ui
                    .button("❌")
                    .accessible_hover_text("Clear search")
                    .clicked()
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
                    meta.display_name.to_lowercase().contains(&query)
                        || meta.id.to_lowercase().contains(&query)
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
            for &(i, meta) in &filtered_tools {
                ui.horizontal(|ui| {
                    let is_selected = self.selected_tool_index == Some(i);
                    if ui
                        .selectable_label(is_selected, meta.display_name)
                        .clicked()
                    {
                        tool_to_select = Some(i);
                    }
                    let tool_help_btn = ui
                        .add(egui::Button::new("❓").small())
                        .accessible_hover_text(format!(
                            "Open Theory Context Portal for {}",
                            meta.display_name
                        ));
                    if tool_help_btn.clicked() {
                        open_portal_for_tool = Some(i);
                    }
                });
            }
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

    pub(super) fn show_side_panel(&mut self, ui: &mut egui::Ui, id_source: &str) {
        if self.available_tools.len() <= 1 {
            return;
        }
        let ctx = ui.ctx().clone();
        let panel_id = scoped_panel_id(ui, format!("{id_source}_tool_selector"));
        PaneSidePanel::right(panel_id)
            .resizable(false)
            .show_inside(ui, |ui| {
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
                                    ui.heading(meta.display_name);
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

    pub(crate) fn apply_starter_gallery_outcome(
        &mut self,
        ctx: &egui::Context,
        id_source: &str,
        is_open: bool,
        selected: Option<crate::starter_gallery::StarterCard>,
    ) {
        let sel = selected.map(crate::starter_gallery::GallerySelection::Starter);
        self.apply_starter_gallery_selection(ctx, id_source, is_open, sel);
    }

    pub(crate) fn apply_starter_gallery_selection(
        &mut self,
        ctx: &egui::Context,
        id_source: &str,
        mut is_open: bool,
        selected: Option<crate::starter_gallery::GallerySelection>,
    ) {
        match selected {
            Some(crate::starter_gallery::GallerySelection::Starter(card)) => {
                let _ = self.launch_starter_card(&card);
                is_open = false;
                ctx.request_repaint();
            }
            Some(crate::starter_gallery::GallerySelection::Story(story)) => {
                let _ = self.launch_story(story, 0);
                self.flush_hud_announcements(ctx);
                is_open = false;
                ctx.request_repaint();
            }
            Some(crate::starter_gallery::GallerySelection::Challenge(ch)) => {
                let _ = self.launch_challenge_by_id(ch.id, Some(ctx));
                is_open = false;
                ctx.request_repaint();
            }
            Some(crate::starter_gallery::GallerySelection::GoalChallenge(goal)) => {
                let _ = self.launch_challenge(goal);
                self.flush_hud_announcements(ctx);
                is_open = false;
                ctx.request_repaint();
            }
            None => {}
        }
        self.show_starter_gallery = is_open;
        self.sync_starter_gallery_state(ctx, id_source);
    }

    pub(super) fn show_starter_gallery_if_open(&mut self, ctx: &egui::Context, id_source: &str) {
        if !self.show_starter_gallery {
            return;
        }
        let cards =
            crate::starter_gallery::get_starter_cards_for_domain(id_source, &self.available_tools);
        let challenges = crate::starter_gallery::get_challenge_cards_for_domain(id_source);
        let modal_source = format!("{id_source}_{}", self.instance_id);
        if crate::starter_gallery::try_defer_starter_gallery(
            ctx,
            id_source,
            &modal_source,
            cards.clone(),
            challenges.clone(),
        ) {
            return;
        }
        let mut is_open = self.show_starter_gallery;
        let selected = crate::starter_gallery::show_starter_gallery_with_challenges(
            ctx,
            &modal_source,
            &cards,
            &challenges,
            &mut is_open,
        );
        self.apply_starter_gallery_selection(ctx, id_source, is_open, selected);
    }

    #[allow(missing_docs)]
    pub fn show(&mut self, ctx: &egui::Context, id_source: &str) {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                self.show_inside(ui, id_source);
            });
    }

    #[allow(missing_docs, clippy::too_many_lines)]
    pub fn show_inside(&mut self, ui: &mut egui::Ui, id_source: &str) {
        let ctx = ui.ctx().clone();
        egui_plot::clear_last_plot_datasets();
        egui_plot::clear_last_plot_primitives();
        let interactive_input = has_interactive_ui_input(&ctx);
        if interactive_input {
            self.rev_cache.untracked_tool_revision =
                self.rev_cache.untracked_tool_revision.wrapping_add(1);
        }
        let sweep_mut_before = crate::reflective_ui::slider::live_sweep_mutation_count(&ctx);

        // Display recovery banner when solver divergence is detected
        if self.is_diverged {
            let banner_id = scoped_panel_id(ui, format!("{id_source}_recovery_banner"));
            egui::TopBottomPanel::top(banner_id).show_inside(ui, |ui| {
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
                self.input_mode = InputMode::Mouse;
            }
        });
        let inst_portal = egui::Id::new(("SHOW_THEORY_PORTAL_INST", id_source, self.instance_id));
        ctx.data_mut(|d| {
            d.insert_temp(egui::Id::new("INPUT_MODE"), self.input_mode);
            d.insert_temp(
                egui::Id::new("INPUT_MODE_TOUCH"),
                self.input_mode == InputMode::Touch,
            );
            d.insert_temp(egui::Id::new("CURRENT_THEORY_PORTAL_INST_ID"), inst_portal);
            d.insert_temp(
                egui::Id::new("SCOPED_ANIMATED_SLIDER_OWNER"),
                self.instance_id,
            );
        });

        self.sync_overlays_before_frame(&ctx, id_source);

        self.show_export_action_bar(ui, id_source);
        self.show_challenge_hud(ui, id_source);
        self.render_challenge_hud_panel(ui, &ctx, id_source);
        self.show_side_panel(ui, id_source);
        self.show_theory_portal(ui, id_source);

        if !self.is_diverged {
            self.cache_valid_snapshot();
            if let Some(tool) = &mut self.active_tool {
                tool.show_inside(ui);
            }
            let _ = self.inspect_runtime_state();
            if self.is_diverged {
                crate::audio::notify_solver_diverged();
                ctx.request_repaint();
            } else {
                let _ = self.evaluate_challenge_predicates(Some(&ctx));
            }
        } else if let Some(tool) = &mut self.active_tool {
            // Solver execution halted: keep last valid plot data active in egui_plot
            if let Some(ref snapshot) = self.last_valid_snapshot {
                egui_plot::clear_last_plot_datasets();
                egui_plot::clear_last_plot_primitives();
                egui_plot::append_last_plot_datasets(&snapshot.plot_datasets);
            }
            tool.show_inside(ui);
        } else {
            egui::CentralPanel::default().show_inside(ui, |ui| {
                self.show_quick_start_cards(ui);
            });
        }

        if !interactive_input
            && crate::reflective_ui::slider::live_sweep_mutation_count(&ctx) > sweep_mut_before
        {
            self.rev_cache.untracked_tool_revision =
                self.rev_cache.untracked_tool_revision.wrapping_add(1);
        }

        ctx.data_mut(|d| {
            d.remove::<egui::Id>(egui::Id::new("CURRENT_THEORY_PORTAL_INST_ID"));
            d.remove::<u64>(egui::Id::new("SCOPED_ANIMATED_SLIDER_OWNER"));
        });

        self.show_export_modal_dialog(&ctx, id_source);
        self.show_starter_gallery_if_open(&ctx, id_source);
        self.sync_overlays_after_frame(&ctx, id_source);

        self.dispatch_pending_exports_with_ctx(&ctx);
    }
}
