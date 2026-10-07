mod ui;

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
