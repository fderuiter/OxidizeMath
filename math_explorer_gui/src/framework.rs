use eframe::egui;
use scientific_metadata::theory::TheoryDescribable;

#[allow(missing_docs)]
pub struct ToolMetadata {
    #[allow(missing_docs)]
    pub name: &'static str,
    #[allow(missing_docs)]
    pub domain: &'static str,
    #[allow(missing_docs)]
    pub tags: &'static [&'static str],
    #[allow(missing_docs)]
    pub build: fn() -> Box<dyn InteractiveTool>,
}

inventory::collect!(ToolMetadata);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum InputMode {
    #[allow(missing_docs)]
    Mouse,
    #[allow(missing_docs)]
    Touch,
}

/// Event context provided to interaction hooks.
pub struct InteractionContext<'a> {
    #[allow(missing_docs)]
    pub pointer_pos: Option<egui::Pos2>,
    #[allow(missing_docs)]
    pub delta: egui::Vec2,
    #[allow(missing_docs)]
    pub is_dragging: bool,
    #[allow(missing_docs)]
    pub is_clicked: bool,
    #[allow(missing_docs)]
    pub response: &'a egui::Response,
    #[allow(missing_docs)]
    pub input_mode: InputMode,
    #[allow(missing_docs)]
    pub multi_touch: Option<egui::MultiTouchInfo>,
    #[allow(missing_docs)]
    pub keys_down: std::collections::HashSet<egui::Key>,
    #[allow(missing_docs)]
    pub modifiers: egui::Modifiers,
}

#[allow(missing_docs)]
pub trait InteractiveTool {
    #[allow(missing_docs)]
    fn name(&self) -> &'static str;

    /// Provide theoretical context.
    fn theory(&self) -> &dyn TheoryDescribable;

    /// Show the tool. Tools can implement this to take full control over rendering.
    /// The default implementation delegates to `show_ui` and then sets up a CentralPanel
    /// with an allocated painter to call the normalized event hooks and `draw`.
    fn show(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left(format!("{}_controls", self.name())).show(ctx, |ui| {
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

        Self {
            available_tools,
            active_tool: None,
            selected_tool_index: None,
            input_mode: InputMode::Mouse,
            show_theory_portal: false,
            saved_tool_states: std::collections::HashMap::new(),
        }
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

                        ui.label(theory.theory_description())
                            .accessible_hover_text("Theoretical background description");

                        ui.separator();
                        ui.heading("Citations");
                        ui.label(theory.theory_citation())
                            .accessible_hover_text("Academic citations");

                        let available = theory.available_descriptions();
                        if !available.is_empty() {
                            ui.separator();
                            ui.heading("Additional Context");
                            for (key, desc) in available {
                                ui.label(format!("{}: {}", key, desc))
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

    fn show_side_panel(&mut self, ctx: &egui::Context, id_source: &str) {
        egui::SidePanel::right(format!("{}_tool_selector", id_source))
            .resizable(false)
            .show(ctx, |ui| {
                ui.heading("Tools");
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for (i, meta) in self.available_tools.iter().enumerate() {
                        if ui
                            .selectable_label(self.selected_tool_index == Some(i), meta.name)
                            .clicked()
                            && self.selected_tool_index != Some(i)
                        {
                            if let Some(old_tool) = &self.active_tool {
                                if let Some(state) = old_tool.save_state() {
                                    self.saved_tool_states
                                        .insert(old_tool.name().to_string(), state);
                                }
                            }
                            self.selected_tool_index = Some(i);
                            let mut new_tool = (meta.build)();
                            if let Some(state) = self.saved_tool_states.get(new_tool.name()) {
                                new_tool.load_state(state);
                            }
                            // Requirement 4 & Acceptance Criteria 3: On-demand registration of bibliographic details when active/loaded
                            scientific_metadata::citation_registry::CitationRegistry::register(
                                new_tool.name().to_string(),
                                new_tool.theory().theory_citation(),
                            );
                            self.active_tool = Some(new_tool);
                        }
                    }
                });
                ui.separator();
                ui.checkbox(&mut self.show_theory_portal, "Theory Context Portal");
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
                self.selected_tool_index = Some(idx);
                let meta = self.available_tools[idx];
                let mut tool = (meta.build)();
                if let Some(saved) = self.saved_tool_states.get(tool.name()) {
                    tool.load_state(saved);
                }
                scientific_metadata::citation_registry::CitationRegistry::register(
                    tool.name().to_string(),
                    tool.theory().theory_citation(),
                );
                self.active_tool = Some(tool);
            }
        }
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

        self.show_side_panel(ctx, id_source);
        self.show_theory_portal(ctx, id_source);

        if let Some(tool) = &mut self.active_tool {
            tool.show(ctx);
        } else {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.centered_and_justified(|ui| {
                    ui.label("No tool selected");
                });
            });
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
}
