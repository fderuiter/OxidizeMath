pub mod callout_overlay;
pub mod clamp;
pub mod export_queue;
#[cfg(test)]
mod export_queue_tests;
mod export_ui;
#[cfg(test)]
mod export_ui_tests;
pub mod math_input;
#[cfg(test)]
mod metadata_label_tests;
mod overlay_sync;
mod persistence;
pub mod recovery;
#[cfg(test)]
mod recovery_tests;
pub mod state_guard;
#[cfg(test)]
mod state_guard_tests;
pub mod trace_overlay;
mod ui;

pub use callout_overlay::*;
pub use clamp::*;
pub use export_queue::*;
pub use math_input::*;
pub use recovery::*;
pub use state_guard::*;
pub use trace_overlay::*;

use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use scientific_metadata::theory::TheoryDescribable;

pub struct ToolMetadata {
    pub id: &'static str,
    pub display_name: &'static str,
    pub domain: &'static str,
    pub tags: &'static [&'static str],
    pub build: fn() -> Box<dyn InteractiveTool>,
}

inventory::collect!(ToolMetadata);
pub use crate::starter_gallery::StarterPresetMetadata;

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

    /// Optional hook to clamp unstable parameters to safe bounds, returning true if handled.
    fn clamp_parameters(&mut self) -> bool {
        false
    }

    /// Domain category of the tool (e.g. "analysis", "biology", "game_theory").
    fn domain(&self) -> &'static str {
        ""
    }

    /// Serializes active parameter preset state to a formatted JSON string with metadata envelope.
    fn export_preset_json(&self) -> Result<String, String> {
        if let Some(state_str) = self.save_state() {
            let config: serde_json::Value = serde_json::from_str(&state_str)
                .map_err(|e| format!("Serialization error: {}", e))?;
            let envelope = crate::presets::PresetWrapper {
                metadata: crate::presets::PresetMetadata {
                    schema_version: "1.0".to_string(),
                    title: self.name().to_string(),
                    domain: self.domain().to_string(),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                },
                config,
            };
            serde_json::to_string_pretty(&envelope)
                .map_err(|e| format!("Serialization error: {}", e))
        } else {
            Err(format!("Preset export not supported for {}", self.name()))
        }
    }

    /// Parses and validates a JSON preset string against this tool's domain, applying configuration if valid.
    fn import_preset_json(&mut self, json_str: &str) -> Result<(), String> {
        let envelope: crate::presets::PresetWrapper<serde_json::Value> =
            serde_json::from_str(json_str)
                .map_err(|e| format!("Invalid JSON preset schema: {}", e))?;
        let expected_domain = self.domain();
        if !expected_domain.is_empty() && envelope.metadata.domain != expected_domain {
            return Err(format!(
                "Domain mismatch: preset is for domain '{}', but active tool is '{}'",
                envelope.metadata.domain, expected_domain
            ));
        }
        let config_str = serde_json::to_string(&envelope.config)
            .map_err(|e| format!("Config serialization error: {}", e))?;
        self.load_state(&config_str);
        Ok(())
    }

    /// Optional hook to export tool datasets.
    /// Default implementation extracts datasets recorded from embedded `egui_plot::Plot`s if available.
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        egui_plot::get_last_plot_datasets()
    }

    /// Optional hook to provide custom figure export options.
    fn export_figure_options(&self) -> Option<egui_plot::export::ExportOptions> {
        None
    }

    /// Optional hook to step simulation logic during frame sampling for GIF animation export.
    fn step_simulation(&mut self) {}

    /// Optional hook to apply a preset JSON configuration envelope (via InteractivePreset).
    fn apply_preset_json(&mut self, _json_str: &str) -> Result<(), String> {
        Ok(())
    }

    /// Optional hook to sample multi-frame animation datasets for GIF export.
    fn sample_animation_frames(
        &mut self,
        num_frames: usize,
    ) -> Vec<egui_plot::export::DatasetFrame> {
        let count = num_frames.max(1);
        let mut frames = Vec::with_capacity(count);

        for _ in 0..count {
            self.step_simulation();
            let ds = self.export_datasets();
            if !ds.is_empty() {
                frames.push(ds);
            }
        }

        if frames.is_empty() {
            let ds = self.export_datasets();
            if !ds.is_empty() {
                frames = egui_plot::export::sample_progressive_frames(&ds, count);
            }
        }

        frames
    }
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
    pub show_starter_gallery: bool,
    #[allow(missing_docs)]
    pub saved_tool_states: std::collections::HashMap<String, String>,
    pub search_query: String,
    last_announced_count: Option<usize>,
    last_announced_query: String,
    pub show_export_modal: bool,
    pub export_options: egui_plot::export::ExportOptions,
    pub export_bg_dark: bool,
    pub export_gif_fps: u32,
    pub export_gif_frames: u32,
    pub export_status_message: Option<String>,
    pub domain: String,
    pub preset_dialog: crate::presets::PresetDialogState,
    pub export_queue: ExportQueue,

    /// Cached state snapshot before divergence for step-back rollback.
    pub last_valid_snapshot: Option<StateSnapshot>,
    /// Baseline parameters snapshot on tool load.
    pub baseline_tool_state: Option<String>,
    /// Last validated finite plot datasets.
    pub current_valid_plot_datasets: Vec<(String, Vec<[f64; 2]>)>,
    /// Whether divergence has been detected in the active tool.
    pub is_diverged: bool,
    /// Diagnostic report for divergence.
    pub diagnostic_report: Option<DiagnosticReport>,
    /// Simulation frame / step counter.
    pub step_counter: usize,
    /// Start time of current active session.
    pub start_time: crate::web_io::PlatformInstant,
    pub(super) instance_id: u64,
    pub(super) overlay_sync: (Option<bool>, Option<bool>, Option<&'static str>),
}

impl SimulationFramework {
    #[allow(missing_docs)]
    pub fn new(domain: &str) -> Self {
        let mut available_tools: Vec<&'static ToolMetadata> = inventory::iter::<ToolMetadata>
            .into_iter()
            .filter(|t| t.domain == domain)
            .collect();

        // Sort by display name (then id) for deterministic order
        available_tools.sort_by_key(|t| (t.display_name, t.id));

        let mut framework = Self {
            available_tools,
            active_tool: None,
            selected_tool_index: None,
            input_mode: InputMode::Mouse,
            show_theory_portal: false,
            show_starter_gallery: false,
            saved_tool_states: std::collections::HashMap::new(),
            search_query: String::new(),
            last_announced_count: None,
            last_announced_query: String::new(),
            show_export_modal: false,
            export_options: egui_plot::export::ExportOptions::default(),
            export_bg_dark: true,
            export_gif_fps: 15,
            export_gif_frames: 20,
            export_status_message: None,
            domain: domain.to_string(),
            preset_dialog: crate::presets::PresetDialogState::default(),
            export_queue: ExportQueue::default(),
            last_valid_snapshot: None,
            baseline_tool_state: None,
            current_valid_plot_datasets: Vec::new(),
            is_diverged: false,
            diagnostic_report: None,
            step_counter: 0,
            start_time: crate::web_io::PlatformInstant::now(),
            instance_id: overlay_sync::next_framework_id(),
            overlay_sync: (None, None, None),
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
        let factory_baseline_state = new_tool.save_state();
        if let Some(state) = self
            .saved_tool_states
            .get(new_tool.name())
            .or_else(|| self.saved_tool_states.get(meta.id))
        {
            new_tool.load_state(state);
        }
        self.is_diverged = false;
        self.diagnostic_report = None;
        self.step_counter = 0;
        self.baseline_tool_state = factory_baseline_state;
        let active_tool_state = new_tool.save_state();
        let prev_datasets = egui_plot::get_last_plot_datasets();
        egui_plot::clear_last_plot_datasets();
        let initial_datasets = new_tool.export_datasets();
        if !prev_datasets.is_empty() {
            egui_plot::append_last_plot_datasets(&prev_datasets);
        }
        (self.overlay_sync.2, self.current_valid_plot_datasets) =
            (Some(new_tool.name()), initial_datasets.clone());
        self.last_valid_snapshot = Some(StateSnapshot {
            tool_state: active_tool_state,
            plot_datasets: initial_datasets,
            step_index: 0,
            timestamp: crate::web_io::PlatformInstant::now(),
        });
        scientific_metadata::citation_registry::CitationRegistry::register(
            new_tool.name().to_string(),
            new_tool.theory().theory_citation(),
        );
        self.active_tool = Some(new_tool);
    }

    /// Selects and activates a tool by ID, display name, or tag in `available_tools` (case-insensitively).
    pub fn select_tool_by_name(&mut self, name: &str) -> bool {
        let name_trimmed = name.trim();
        if let Some(index) = self.available_tools.iter().position(|t| {
            t.id.eq_ignore_ascii_case(name_trimmed)
                || t.display_name.eq_ignore_ascii_case(name_trimmed)
                || t.tags
                    .iter()
                    .any(|tag| tag.eq_ignore_ascii_case(name_trimmed))
        }) {
            self.select_tool(index);
            true
        } else {
            false
        }
    }
}

pub use crate::camera::Camera3D;
