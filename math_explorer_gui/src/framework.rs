use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use scientific_metadata::theory::TheoryDescribable;

pub struct ToolMetadata {
    pub name: &'static str,
    pub domain: &'static str,
    pub tags: &'static [&'static str],
    pub build: fn() -> Box<dyn InteractiveTool>,
}

inventory::collect!(ToolMetadata);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputMode {
    Mouse,
    Touch,
}

/// Event context provided to interaction hooks.
#[allow(missing_docs)]
pub struct InteractionContext<'a> {
    pub pointer_pos: Option<egui::Pos2>,
    pub delta: egui::Vec2,
    pub is_dragging: bool,
    pub is_clicked: bool,
    pub response: &'a egui::Response,
    pub input_mode: InputMode,
    pub multi_touch: Option<egui::MultiTouchInfo>,
    pub keys_down: std::collections::HashSet<egui::Key>,
    pub modifiers: egui::Modifiers,
}

pub trait InteractiveTool {
    fn name(&self) -> &'static str;

    /// Provide theoretical context.
    fn theory(&self) -> &dyn TheoryDescribable;

    /// Show the tool. Tools can implement this to take full control over rendering.
    /// The default implementation delegates to `show_ui` and then sets up a CentralPanel
    /// with an allocated painter to call the normalized event hooks and `draw`.
    fn show(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left(format!("{}_controls", self.name())).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading(self.name());
                let help_btn = ui
                    .add(egui::Button::new("❓").small())
                    .accessible_hover_text(format!(
                        "Toggle Theory Context Portal for {}",
                        self.name()
                    ));
                if help_btn.clicked() {
                    let current = ctx.data(|d| {
                        d.get_temp::<bool>(egui::Id::new("SHOW_THEORY_PORTAL"))
                            .unwrap_or(false)
                    });
                    ctx.data_mut(|d| d.insert_temp(egui::Id::new("SHOW_THEORY_PORTAL"), !current));
                }
            });
            ui.separator();
            self.show_ui(ui);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());

            let multi_touch = ui.input(|i| i.multi_touch());

            let input_mode = ctx.data(|d| {
                d.get_temp(egui::Id::new("INPUT_MODE"))
                    .unwrap_or(InputMode::Mouse)
            });

            let keys_down = ui.input(|i| i.keys_down.clone());
            let modifiers = ui.input(|i| i.modifiers);

            let interaction_ctx = InteractionContext {
                pointer_pos: response.interact_pointer_pos(),
                delta: response.drag_delta(),
                is_dragging: response.dragged(),
                is_clicked: response.clicked(),
                response: &response,
                input_mode,
                multi_touch,
                keys_down,
                modifiers,
            };

            if interaction_ctx.response.has_focus() {
                self.on_keyboard(&interaction_ctx);
            }

            if interaction_ctx.multi_touch.is_some() {
                self.on_gesture(&interaction_ctx);
            } else if let Some(_pos) = interaction_ctx.pointer_pos {
                if interaction_ctx.is_dragging {
                    self.on_drag(&interaction_ctx);
                    self.on_brush(&interaction_ctx);
                } else if interaction_ctx.is_clicked {
                    self.on_click(&interaction_ctx);
                } else {
                    self.on_hover(&interaction_ctx);
                }
            }

            self.draw(ui, &response, &painter);
        });
    }

    /// Optional: UI rendering for the tool's specific side-panel controls.
    fn show_ui(&mut self, _ui: &mut egui::Ui) {}

    /// Optional: Context-based drawing.
    fn draw(&mut self, _ui: &mut egui::Ui, _response: &egui::Response, _painter: &egui::Painter) {}

    // Normalized event hooks
    #[allow(missing_docs)]
    fn on_hover(&mut self, _ctx: &InteractionContext) {}
    #[allow(missing_docs)]
    fn on_drag(&mut self, _ctx: &InteractionContext) {}
    #[allow(missing_docs)]
    fn on_click(&mut self, _ctx: &InteractionContext) {}
    #[allow(missing_docs)]
    fn on_brush(&mut self, _ctx: &InteractionContext) {}
    #[allow(missing_docs)]
    fn on_gesture(&mut self, _ctx: &InteractionContext) {}
    #[allow(missing_docs)]
    fn on_keyboard(&mut self, _ctx: &InteractionContext) {}

    /// Optional hook to serialize tool-specific parameters into a String (e.g. JSON).
    fn save_state(&self) -> Option<String> {
        None
    }

    /// Optional hook to deserialize and restore tool-specific parameters from a String.
    fn load_state(&mut self, _state: &str) {}
}

#[derive(serde::Serialize, serde::Deserialize)]
struct FrameworkState {
    selected_tool_index: Option<usize>,
    selected_tool_name: Option<String>,
    show_theory_portal: bool,
    tool_states: std::collections::HashMap<String, String>,
}

#[allow(missing_docs)]
pub struct SimulationFramework {
    #[allow(missing_docs)]
    pub available_tools: Vec<&'static ToolMetadata>,
    #[allow(missing_docs)]
    pub active_tool: Option<Box<dyn InteractiveTool>>,
    #[allow(missing_docs)]
    pub selected_tool_index: Option<usize>,
    #[allow(missing_docs)]
    pub input_mode: InputMode,
    #[allow(missing_docs)]
    pub show_theory_portal: bool,
    #[allow(missing_docs)]
    pub saved_tool_states: std::collections::HashMap<String, String>,
    pub search_query: String,
    last_announced_count: Option<usize>,
    last_announced_query: String,
}

impl SimulationFramework {
    #[allow(missing_docs)]
    pub fn new(domain: &str) -> Self {
        let mut available_tools: Vec<&'static ToolMetadata> = inventory::iter::<ToolMetadata>
            .into_iter()
            .filter(|t| t.domain == domain)
            .collect();

        // Sort by name for deterministic order
        available_tools.sort_by_key(|t| t.name);

        let mut framework = Self {
            available_tools,
            active_tool: None,
            selected_tool_index: None,
            input_mode: InputMode::Mouse,
            show_theory_portal: false,
            saved_tool_states: std::collections::HashMap::new(),
            search_query: String::new(),
            last_announced_count: None,
            last_announced_query: String::new(),
        };

        if !framework.available_tools.is_empty() {
            framework.select_tool(0);
        }

        framework
    }

    /// Selects and activates a tool by index in `available_tools`.
    pub fn select_tool(&mut self, index: usize) {
        if index >= self.available_tools.len() {
            return;
        }
        if self.selected_tool_index == Some(index) && self.active_tool.is_some() {
            return;
        }
        if let Some(old_tool) = &self.active_tool {
            if let Some(state) = old_tool.save_state() {
                self.saved_tool_states
                    .insert(old_tool.name().to_string(), state);
            }
        }
        self.selected_tool_index = Some(index);
        let meta = self.available_tools[index];
        let mut new_tool = (meta.build)();
        if let Some(state) = self.saved_tool_states.get(new_tool.name()) {
            new_tool.load_state(state);
        }
        scientific_metadata::citation_registry::CitationRegistry::register(
            new_tool.name().to_string(),
            new_tool.theory().theory_citation(),
        );
        self.active_tool = Some(new_tool);
    }

    fn show_theory_portal(&self, ctx: &egui::Context, id_source: &str) {
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

    fn render_search_bar(&mut self, ui: &mut egui::Ui) {
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
    fn render_filtered_tools(&mut self, ui: &mut egui::Ui) {
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

    fn show_side_panel(&mut self, ctx: &egui::Context, id_source: &str) {
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

    /// Serializes the framework state (selected tool index, active tool state, theory portal state) to JSON.
    pub fn save_state(&self) -> Option<String> {
        let mut tool_states = self.saved_tool_states.clone();
        if let Some(active_tool) = &self.active_tool {
            if let Some(state) = active_tool.save_state() {
                tool_states.insert(active_tool.name().to_string(), state);
            }
        }
        let selected_tool_name = self.selected_tool_index.and_then(|idx| {
            self.available_tools
                .get(idx)
                .map(|meta| meta.name.to_string())
        });

        let state = FrameworkState {
            selected_tool_index: self.selected_tool_index,
            selected_tool_name,
            show_theory_portal: self.show_theory_portal,
            tool_states,
        };
        serde_json::to_string(&state).ok()
    }

    /// Deserializes and restores the framework state from JSON.
    pub fn load_state(&mut self, state_str: &str) {
        let state: FrameworkState = match serde_json::from_str(state_str) {
            Ok(s) => s,
            Err(_) => return,
        };
        self.show_theory_portal = state.show_theory_portal;
        self.saved_tool_states = state.tool_states;

        let tool_idx = if let Some(ref name) = state.selected_tool_name {
            self.available_tools
                .iter()
                .position(|m| m.name == *name)
                .or(state.selected_tool_index)
        } else {
            state.selected_tool_index
        };

        if let Some(idx) = tool_idx {
            if idx < self.available_tools.len() {
                self.select_tool(idx);
            }
        }
    }

    fn show_quick_start_cards(&mut self, ui: &mut egui::Ui) {
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

    #[allow(missing_docs)]
    pub fn show(&mut self, ctx: &egui::Context, id_source: &str) {
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

        self.show_side_panel(ctx, id_source);
        self.show_theory_portal(ctx, id_source);

        if let Some(tool) = &mut self.active_tool {
            tool.show(ctx);
        } else {
            egui::CentralPanel::default().show(ctx, |ui| {
                self.show_quick_start_cards(ui);
            });
        }

        if let Some(portal_state) =
            ctx.data(|d| d.get_temp::<bool>(egui::Id::new("SHOW_THEORY_PORTAL")))
        {
            self.show_theory_portal = portal_state;
        }
    }
}

pub use crate::camera::{Camera3D, CoordinateMapper};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_framework_state_roundtrip() {
        let mut framework = SimulationFramework::new("analysis");
        framework.show_theory_portal = true;
        if !framework.available_tools.is_empty() {
            framework.selected_tool_index = Some(0);
            let meta = framework.available_tools[0];
            framework.active_tool = Some((meta.build)());
        }

        let saved_json = framework
            .save_state()
            .expect("Must serialize framework state");

        let mut framework2 = SimulationFramework::new("analysis");
        framework2.load_state(&saved_json);

        assert!(framework2.show_theory_portal);
        if !framework.available_tools.is_empty() {
            assert_eq!(framework2.selected_tool_index, Some(0));
            assert!(framework2.active_tool.is_some());
        }
    }

    #[test]
    fn test_framework_show_theory_portal_toggle() {
        let ctx = egui::Context::default();
        let mut framework = SimulationFramework::new("analysis");
        assert!(!framework.show_theory_portal);

        // Frame 1
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            framework.show(ctx, "test_framework");
        });
        assert_eq!(
            ctx.data(|d| d.get_temp::<bool>(egui::Id::new("SHOW_THEORY_PORTAL"))),
            Some(false)
        );

        // Mutate context temp data (simulating help button click)
        ctx.data_mut(|d| d.insert_temp(egui::Id::new("SHOW_THEORY_PORTAL"), true));

        // Frame 2: framework syncs from context temp data
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            framework.show(ctx, "test_framework");
        });
        assert!(framework.show_theory_portal);
    }
}
