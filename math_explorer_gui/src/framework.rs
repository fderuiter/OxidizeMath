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
                            self.selected_tool_index = Some(i);
                            let new_tool = (meta.build)();
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

#[allow(missing_docs)]
pub struct CoordinateMapper {
    #[allow(missing_docs)]
    pub screen_rect: egui::Rect,
    #[allow(missing_docs)]
    pub sim_rect: egui::Rect,
}

#[derive(Clone, Copy, Debug)]
#[allow(missing_docs)]
pub struct Camera3D {
    #[allow(missing_docs)]
    pub pitch: f32,
    #[allow(missing_docs)]
    pub yaw: f32,
    #[allow(missing_docs)]
    pub zoom: f32,
}

impl Default for Camera3D {
    fn default() -> Self {
        Self {
            pitch: 0.5,
            yaw: 0.5,
            zoom: 1.0,
        }
    }
}

impl Camera3D {
    #[allow(missing_docs)]
    pub fn new(pitch: f32, yaw: f32, zoom: f32) -> Self {
        Self { pitch, yaw, zoom }
    }

    #[allow(missing_docs)]
    pub fn handle_interaction(&mut self, response: &egui::Response, ui: &egui::Ui) {
        let multi_touch = ui.input(|i| i.multi_touch());

        if let Some(touch) = multi_touch {
            if touch.zoom_delta != 1.0 {
                let zoom_factor = 1.0 + (touch.zoom_delta - 1.0) * 0.5;
                self.zoom *= zoom_factor;
                self.zoom = self.zoom.clamp(0.01, 100.0);
            }
            if touch.translation_delta != egui::Vec2::ZERO {
                self.yaw -= touch.translation_delta.x * 0.01;
                self.pitch -= touch.translation_delta.y * 0.01;
            }
        } else if response.dragged() {
            let delta = response.drag_delta();
            self.yaw -= delta.x * 0.01;
            self.pitch -= delta.y * 0.01;
        }

        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0 && response.hovered() && multi_touch.is_none() {
            self.zoom *= 1.0 + (scroll * 0.001);
            self.zoom = self.zoom.clamp(0.01, 100.0);
        }

        if response.has_focus() {
            self.handle_keyboard_interaction(ui);
        }
    }

    fn handle_keyboard_interaction(&mut self, ui: &egui::Ui) {
        let mut yaw_delta = 0.0;
        let mut pitch_delta = 0.0;
        let mut zoom_factor = 1.0;
        if ui.input(|i| i.key_down(egui::Key::A)) {
            yaw_delta += 0.05;
        }
        if ui.input(|i| i.key_down(egui::Key::D)) {
            yaw_delta -= 0.05;
        }
        if ui.input(|i| i.key_down(egui::Key::W)) {
            pitch_delta += 0.05;
        }
        if ui.input(|i| i.key_down(egui::Key::S)) {
            pitch_delta -= 0.05;
        }
        if ui.input(|i| i.key_down(egui::Key::Q)) {
            zoom_factor *= 1.05;
        }
        if ui.input(|i| i.key_down(egui::Key::E)) {
            zoom_factor /= 1.05;
        }

        self.yaw -= yaw_delta;
        self.pitch -= pitch_delta;
        self.zoom *= zoom_factor;
        self.zoom = self.zoom.clamp(0.01, 100.0);

        self.register_camera_commands(ui);
    }

    fn register_camera_commands(&self, ui: &egui::Ui) {
        ui.ctx().data_mut(|d| {
            let mut registry = d
                .get_temp::<egui_plot::commands::CommandRegistryData>(egui::Id::new("CMD_REGISTRY"))
                .unwrap_or_default();

            let commands = [
                (egui::Key::W, "Pitch Up", "Rotate camera pitch upwards"),
                (egui::Key::S, "Pitch Down", "Rotate camera pitch downwards"),
                (egui::Key::A, "Yaw Left", "Rotate camera yaw left"),
                (egui::Key::D, "Yaw Right", "Rotate camera yaw right"),
                (egui::Key::Q, "Zoom Out", "Zoom camera out"),
                (egui::Key::E, "Zoom In", "Zoom camera in"),
            ];

            for (key, name, desc) in commands {
                if !registry
                    .commands
                    .iter()
                    .any(|c| c.name == name && c.context == "Camera 3D")
                {
                    registry
                        .commands
                        .push(egui_plot::commands::CommandMetadata {
                            name: name.to_string(),
                            description: desc.to_string(),
                            trigger: egui_plot::commands::CommandTrigger::Key(key),
                            desktop_only: true,
                            context: "Camera 3D".to_string(),
                        });
                }
            }
            d.insert_temp(egui::Id::new("CMD_REGISTRY"), registry);
        });
    }

    #[allow(missing_docs)]
    pub fn project(&self, point: &[f64; 3]) -> [f64; 2] {
        let cy = (self.yaw as f64).cos();
        let sy = (self.yaw as f64).sin();
        let cp = (self.pitch as f64).cos();
        let sp = (self.pitch as f64).sin();

        // Apply yaw (rotate around Z)
        let x1 = point[0] * cy - point[1] * sy;
        let y1 = point[0] * sy + point[1] * cy;
        let z1 = point[2];

        // Apply pitch (rotate around X)
        let x2 = x1;
        let y2 = y1 * cp - z1 * sp;

        [x2 * (self.zoom as f64), y2 * (self.zoom as f64)]
    }

    #[allow(missing_docs)]
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Yaw");
        ui.drag_angle(&mut self.yaw);

        ui.label("Pitch");
        ui.drag_angle(&mut self.pitch);

        ui.add(egui::Slider::new(&mut self.zoom, 0.1..=5.0).text("Zoom"));
    }
}

impl CoordinateMapper {
    #[allow(missing_docs)]
    pub fn new(screen_rect: egui::Rect, sim_rect: egui::Rect) -> Self {
        Self {
            screen_rect,
            sim_rect,
        }
    }

    #[allow(missing_docs)]
    pub fn screen_to_sim(&self, pos: egui::Pos2) -> egui::Pos2 {
        let x_norm = (pos.x - self.screen_rect.min.x) / self.screen_rect.width();
        let y_norm = (pos.y - self.screen_rect.min.y) / self.screen_rect.height();

        egui::Pos2::new(
            self.sim_rect.min.x + x_norm * self.sim_rect.width(),
            self.sim_rect.min.y + y_norm * self.sim_rect.height(),
        )
    }

    #[allow(missing_docs)]
    pub fn sim_to_screen(&self, pos: egui::Pos2) -> egui::Pos2 {
        let x_norm = (pos.x - self.sim_rect.min.x) / self.sim_rect.width();
        let y_norm = (pos.y - self.sim_rect.min.y) / self.sim_rect.height();

        egui::Pos2::new(
            self.screen_rect.min.x + x_norm * self.screen_rect.width(),
            self.screen_rect.min.y + y_norm * self.screen_rect.height(),
        )
    }
}
