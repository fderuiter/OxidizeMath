pub mod callout_overlay;
pub mod clamp;
#[cfg(test)]
mod clamp_tests;
pub mod export_figure;
pub mod export_queue;
#[cfg(test)]
mod export_queue_tests;
pub mod export_types;
mod export_ui;
#[cfg(test)]
mod export_ui_tests;
pub mod math_input;
#[cfg(test)]
mod metadata_label_tests;
mod overlay_sync;
#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod perf_guard_tests;
mod persistence;
pub mod playback;
#[cfg(test)]
mod playback_tests;
pub mod recovery;
#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod recovery_tests;
pub mod state_guard;
#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod state_guard_tests;
pub mod trace_overlay;
mod ui;

pub use callout_overlay::*;
pub use clamp::*;
pub use egui_plot::{ModalFrameContext, ModalLifecycle};
pub use export_figure::*;
pub use export_queue::*;
pub use export_types::*;
pub use math_input::*;
pub use playback::*;
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
    ///
    /// # Accessibility Requirements for Custom `show` Overrides
    ///
    /// The default implementation allocates its central canvas with [`egui::Sense::click_and_drag`]
    /// (which includes [`egui::Sense::FOCUSABLE`]) and dispatches [`Self::on_keyboard`] when
    /// `response.has_focus()` is true. When overriding `show` directly:
    ///
    /// - Allocate custom painter regions with [`egui::Sense::focusable_noninteractive`] (or
    ///   [`egui::Sense::click`] / [`egui::Sense::click_and_drag`] for interactive canvases) instead
    ///   of `Sense::hover()` so `Tab` traversal reaches the visualization.
    /// - Render raster textures via `ui.add(egui::Image::new(texture).alt_text(...).sense(...))`
    ///   instead of raw `ui.image(texture)` calls so `AccessKit` exposes `Role::Image` and non-empty
    ///   alternative text.
    /// - Populate `AccessKit` metadata (`Role`, non-empty `label`, and dynamic `value` summary) via
    ///   `ctx.accesskit_node_builder(response.id, ...)` (or
    ///   `crate::accessibility::register_canvas_accessibility`) and draw a high-contrast focus ring
    ///   when `response.has_focus()`.
    /// - If the canvas consumes arrow keys, call `Memory::set_focus_lock_filter` with
    ///   `EventFilter { tab: false, horizontal_arrows: true, vertical_arrows: true, escape: false }`
    ///   strictly while `response.has_focus()` so `Tab` and `Shift+Tab` never trap keyboard users.
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
            } else if interaction_ctx.pointer_pos.is_some() {
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

    /// Revision counter for persisted tool state (`0` falls back to framework UI revision tracking).
    fn state_revision(&self) -> u64 {
        0
    }

    /// Returns true if this tool maintains local undo history with available entries.
    fn can_undo(&self) -> bool {
        false
    }

    /// Returns true if this tool maintains local redo history with available entries.
    fn can_redo(&self) -> bool {
        false
    }

    /// Attempts to undo the latest local tool edit, returning true if handled.
    fn try_undo(&mut self) -> bool {
        false
    }

    /// Attempts to redo the latest locally undone tool edit, returning true if handled.
    fn try_redo(&mut self) -> bool {
        false
    }

    /// Returns the interactive canvas `Id` when this tool owns non-text canvas focus.
    fn focused_canvas_id(&self) -> Option<egui::Id> {
        None
    }

    /// Direct numeric parameter finiteness hook without JSON serialization.
    fn has_finite_parameters(&self) -> bool {
        true
    }

    /// Optional hook to serialize tool-specific parameters into a String (e.g. JSON).
    fn save_state(&self) -> Option<String> {
        None
    }

    /// Lightweight parameter-only snapshot hook used by `cache_valid_snapshot`.
    fn save_recovery_state(&self) -> Option<String> {
        self.save_state()
    }

    /// Restores parameter-only recovery state without clobbering pinned comparison overlays.
    fn load_recovery_state(&mut self, state: &str) {
        self.load_state(state);
    }

    /// Fallibly deserializes and restores tool-specific parameters from a JSON state string.
    fn try_load_state(&mut self, _state: &str) -> Result<(), String> {
        Err(format!(
            "Preset state deserialization not supported for {}",
            self.name()
        ))
    }

    /// Optional hook to deserialize and restore tool-specific parameters from a String.
    fn load_state(&mut self, state: &str) {
        let _ = self.try_load_state(state);
    }

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
                .map_err(|e| format!("Serialization error: {e}"))?;
            crate::presets::serialize_preset_envelope(self.name(), self.domain(), config)
        } else {
            Err(format!("Preset export not supported for {}", self.name()))
        }
    }

    /// Parses and validates a JSON preset string against this tool's domain, applying configuration if valid.
    fn import_preset_json(&mut self, json_str: &str) -> Result<(), String> {
        let config: serde_json::Value =
            crate::presets::deserialize_preset_envelope(json_str, self.domain(), self.name())?;
        let config_str = serde_json::to_string(&config)
            .map_err(|e| format!("Config serialization error: {e}"))?;
        self.try_load_state(&config_str)
    }

    /// Declares which dataset and figure export formats this tool supports.
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    /// Optional hook to export tool datasets.
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        egui_plot::get_last_plot_datasets()
    }

    /// Optional hook to extract offscreen drawable plot primitives (`Bar`, `Arrow`, `BoxElem`, `Polyline`, `Points`).
    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        egui_plot::get_last_plot_primitives()
    }

    /// Optional hook for texture-backed tools to expose their rendered 2D raster field for PNG, SVG, and GIF exports.
    fn export_raster_image(&self) -> Option<egui::ColorImage> {
        None
    }

    /// Optional hook to provide custom figure export options.
    fn export_figure_options(&self) -> Option<egui_plot::export::ExportOptions> {
        None
    }

    /// Optional hook to step simulation logic during frame sampling for GIF animation export.
    fn step_simulation(&mut self) {}

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

    /// Returns a type-erased reference for test inspection of concrete tool state.
    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }

    /// Returns a mutable type-erased reference for test manipulation of concrete tool state.
    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
}

#[derive(Default, Clone, Copy)]
pub(super) struct FrameworkSyncState {
    pub(super) theory_portal: Option<bool>,
    pub(super) starter_gallery: Option<bool>,
    pub(super) datasets_owner_tool: Option<&'static str>,
    pub(super) last_modal_scope_id: Option<egui::Id>,
}

#[derive(Default)]
pub(super) struct FrameworkRevisionCache {
    pub(super) framework_revision: u64,
    pub(super) untracked_tool_revision: u64,
    pub(super) active_tool_json:
        std::cell::RefCell<Option<(&'static str, u64, std::sync::Arc<str>)>>,
    pub(super) last_snapshot_rev: Option<(&'static str, u64)>,
}

#[allow(missing_docs)]
pub struct SimulationFramework {
    pub available_tools: Vec<&'static ToolMetadata>,
    pub active_tool: Option<Box<dyn InteractiveTool>>,
    pub selected_tool_index: Option<usize>,
    pub input_mode: InputMode,
    pub show_theory_portal: bool,
    pub show_starter_gallery: bool,
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
    pub export_status_severity: ExportStatusSeverity,
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
    pub(super) sync_state: FrameworkSyncState,
    pub(super) rev_cache: FrameworkRevisionCache,
}

impl SimulationFramework {
    #[allow(missing_docs)]
    pub fn new(domain: &str) -> Self {
        let mut available_tools: Vec<&'static ToolMetadata> = inventory::iter::<ToolMetadata>
            .into_iter()
            .filter(|t| t.domain == domain)
            .collect();

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
            export_status_severity: ExportStatusSeverity::Success,
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
            sync_state: FrameworkSyncState::default(),
            rev_cache: FrameworkRevisionCache::default(),
        };

        if !framework.available_tools.is_empty() {
            framework.select_tool(0);
        }

        framework
    }
}

pub use crate::camera::{Camera3D, Plot3DViewport, CAMERA_3D_HELPER_TEXT};
