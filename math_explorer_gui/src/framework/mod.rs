pub mod callout_overlay;
pub mod challenge;
mod challenge_hud;
pub mod clamp;
#[cfg(test)]
mod clamp_tests;
pub mod decimate;
#[cfg(all(test, feature = "ai", feature = "physics", feature = "pure_math"))]
mod drag_bounds_and_rollback_tests;
mod export_3d;
mod export_dataset;
mod export_figure;
mod export_queue;
#[cfg(test)]
mod export_queue_tests;
mod export_shader;
#[cfg(test)]
mod export_shader_tests;
mod export_spatial;
#[cfg(test)]
mod export_spatial_tests;
pub mod export_types;
mod export_ui;
mod export_ui_modal;
#[cfg(test)]
mod export_ui_tests;
pub mod fit_bridge;
mod init;
pub mod math_input;
#[cfg(test)]
mod metadata_label_tests;
mod overlay_sync;
mod pane_scope;
#[cfg(test)]
mod perf_guard_tests;
mod persistence;
pub mod playback;
#[cfg(test)]
mod playback_tests;
pub mod plot_persistence;
#[cfg(test)]
mod plot_persistence_session_tests;
#[cfg(test)]
mod plot_persistence_tests;
pub mod recovery;
#[cfg(test)]
mod recovery_tests;
pub mod side_panel;
pub mod state_bounds;
#[cfg(test)]
mod state_bounds_tests;
pub mod state_guard;
#[cfg(test)]
mod state_guard_tests;
pub(crate) mod theory_links;
#[cfg(test)]
mod theory_links_tests;
pub mod trace_overlay;
mod ui;
mod url_share;

pub use callout_overlay::*;
pub use challenge::ChallengeEvaluation;
pub use clamp::*;
pub use decimate::*;
pub use egui_plot::{ModalFrameContext, ModalLifecycle};
pub(crate) use export_queue::*;
pub use export_types::*;
pub use fit_bridge::ensure_custom_fit_bridge;
pub use math_input::*;
pub use pane_scope::{
    is_pane_activation_shortcut_allowed, is_pane_shortcut_allowed, scoped_panel_id,
};
pub use playback::*;
pub use recovery::*;
pub use side_panel::{
    clear_pane_side_panel_records, pane_side_panel_records, PaneSidePanel, PaneSidePanelRecord,
    MAX_PANE_SIDE_PANEL_WIDTH_RATIO,
};
pub use state_bounds::*;
pub use state_guard::*;
pub use trace_overlay::*;

pub(super) use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use init::FrameworkRevisionCache;
use overlay_sync::FrameworkSyncState;
use scientific_metadata::theory::TheoryDescribable;

pub struct ToolMetadata {
    pub id: &'static str,
    pub display_name: &'static str,
    pub domain: &'static str,
    pub tags: &'static [&'static str],
    pub build: fn() -> Box<dyn InteractiveTool>,
}

inventory::collect!(ToolMetadata);
pub use crate::starter_gallery::{ChallengeMetadata, StarterPresetMetadata};

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

    /// Show the tool at the root `egui::Context` level by delegating to [`Self::show_inside`].
    fn show(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                self.show_inside(ui);
            });
    }

    /// Show the tool inside a scoped parent `egui::Ui` (supporting multi-pane layouts on a single root `Context`).
    ///
    /// # Accessibility Requirements for Custom `show_inside` Overrides
    ///
    /// The default implementation allocates its central canvas with [`egui::Sense::click_and_drag`]
    /// (which includes [`egui::Sense::FOCUSABLE`]) and dispatches [`Self::on_keyboard`] when
    /// `response.has_focus()` is true. When overriding `show_inside` directly:
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
    fn show_inside(&mut self, ui: &mut egui::Ui) {
        pane_scope::render_default_tool_inside(self, ui);
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

    /// Fallibly restores parameter-only recovery state without clobbering pinned comparison overlays.
    fn try_load_recovery_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        self.try_load_state(state)
    }

    /// Restores parameter-only recovery state without clobbering pinned comparison overlays.
    fn load_recovery_state(&mut self, state: &str) {
        let _ = self.try_load_recovery_state(state);
    }

    /// Structural and byte-size budget enforced before JSON state deserialization.
    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::DEFAULT
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
        if self.state_bounds().validate_json(state).is_err() {
            return;
        }
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

    /// Activates or clears an inverse challenge contract (`None` exits challenge mode).
    fn apply_challenge(&mut self, _challenge_id: Option<&str>) -> Result<(), String> {
        Ok(())
    }

    /// Returns the canonical ID of the active challenge on this tool, if any.
    fn active_challenge_id(&self) -> Option<&'static str> {
        None
    }

    /// Evaluates live `0..=100` score and pass/residual status for the active inverse challenge.
    fn evaluate_challenge(&self) -> Option<ChallengeEvaluation> {
        None
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
        let bounds = self.state_bounds();
        let config: Box<serde_json::value::RawValue> =
            crate::presets::deserialize_preset_envelope_bounded(
                json_str,
                self.domain(),
                self.name(),
                bounds,
            )?;
        let config_str = config.get();
        bounds.validate_json(config_str)?;
        self.try_load_state(config_str)
    }

    /// Declares which dataset and figure export formats this tool supports.
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    /// Optional hook to export tool datasets.
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        egui_plot::get_last_plot_datasets()
    }

    /// Optional hook to export 3D world-coordinate `(X, Y, Z)` datasets.
    fn export_datasets_3d(&self) -> Vec<(String, Vec<[f64; 3]>)> {
        egui_plot::get_last_plot_datasets_3d()
    }

    /// Optional hook to extract offscreen drawable plot primitives (`Bar`, `Arrow`, `BoxElem`, `Polyline`, `Points`).
    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        egui_plot::get_last_plot_primitives()
    }

    /// Optional hook for texture-backed tools to expose their rendered 2D raster field for PNG and GIF exports.
    fn export_raster_image(&self) -> Option<egui::ColorImage> {
        None
    }

    /// Optional hook for 2D spatial tools to expose continuous or lattice scalar fields for Marching Squares contours and 16-bit textures.
    fn export_scalar_field(
        &self,
    ) -> Option<math_commons::math_kernel::marching_squares::ScalarField2D> {
        None
    }

    /// Optional resolution-aware scalar-field hook used when baking borderless power-of-two 16-bit heightmaps or normal maps.
    fn export_scalar_field_at(
        &self,
        _width: usize,
        _height: usize,
    ) -> Option<math_commons::math_kernel::marching_squares::ScalarField2D> {
        self.export_scalar_field()
    }

    /// Optional hook for discrete lattice or matrix tools to serialize standalone vector `<path>` or `<rect>` SVG documents.
    fn export_vector_svg(
        &self,
        _options: &egui_plot::export::ExportOptions,
        _contour_levels: usize,
    ) -> Option<Result<String, String>> {
        None
    }

    /// Optional hook to provide custom figure export options.
    fn export_figure_options(&self) -> Option<egui_plot::export::ExportOptions> {
        None
    }

    /// Optional cold-path hook to build a watertight 3D triangle [`oxidize_core::mesh::Mesh`] for `.stl`, `.ply`, and `.obj` exports.
    fn export_mesh_3d(&self) -> Option<Result<oxidize_core::mesh::Mesh, String>> {
        None
    }

    /// Optional cold-path hook to build a per-vertex colored 3D [`oxidize_core::mesh::ColoredMesh`] for `.ply` exports.
    fn export_colored_mesh_3d(&self) -> Option<Result<oxidize_core::mesh::ColoredMesh, String>> {
        None
    }

    /// Optional cold-path hook to serialize an INRIA-standard 3D Gaussian Splatting binary `.ply` scene.
    fn export_splat_ply_3d(&self) -> Option<Result<Vec<u8>, String>> {
        None
    }

    /// Optional hook to emit a standalone GLSL fragment shader for the requested [`ShaderDialect`].
    fn export_glsl_shader(&self, _dialect: ShaderDialect) -> Result<String, String> {
        Err(format!(
            "GLSL shader export is not supported for {}",
            self.name()
        ))
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

#[allow(missing_docs)]
pub struct SimulationFramework {
    pub available_tools: Vec<&'static ToolMetadata>,
    pub active_tool: Option<Box<dyn InteractiveTool>>,
    pub selected_tool_index: Option<usize>,
    pub input_mode: InputMode,
    pub show_theory_portal: bool,
    pub show_starter_gallery: bool,
    pub saved_tool_states: std::collections::HashMap<String, String>,
    pub saved_plot_states: plot_persistence::SavedPlotStatesMap,
    pub(super) plot_sync_generation: u64,
    pub search_query: String,
    pub(super) last_announced_count: Option<usize>,
    pub(super) last_announced_query: String,
    pub show_export_modal: bool,
    pub export_options: egui_plot::export::ExportOptions,
    pub export_bg_dark: bool,
    pub export_gif_fps: u32,
    pub export_gif_frames: u32,
    pub export_contour_levels: usize,
    pub export_texture_size: u32,
    pub export_normal_strength: f32,
    pub export_status_message: Option<String>,
    pub export_status_severity: ExportStatusSeverity,
    pub domain: String,
    pub preset_dialog: crate::presets::PresetDialogState,
    pub(crate) export_queue: ExportQueue,

    /// Cached state snapshot before divergence for step-back rollback.
    pub last_valid_snapshot: Option<StateSnapshot>,
    /// Baseline parameters snapshot on tool load.
    pub baseline_tool_state: Option<String>,
    /// Last validated finite plot datasets.
    pub current_valid_plot_datasets: Vec<(String, Vec<[f64; 2]>)>,
    /// Last validated finite 3D world-coordinate plot datasets.
    pub current_valid_plot_datasets_3d: Vec<(String, Vec<[f64; 3]>)>,
    /// Whether divergence has been detected in the active tool.
    pub is_diverged: bool,
    /// Diagnostic report for divergence.
    pub diagnostic_report: Option<DiagnosticReport>,
    /// Simulation frame / step counter.
    pub step_counter: usize,
    /// Start time of current active session.
    pub start_time: crate::web_io::PlatformInstant,
    /// Active inverse challenge contract displayed in the Challenge HUD, if any.
    pub active_challenge: Option<&'static ChallengeMetadata>,
    pub(super) challenge_was_passed: bool,
    pub(super) instance_id: u64,
    pub(super) sync_state: FrameworkSyncState,
    pub(super) rev_cache: FrameworkRevisionCache,
}

pub use crate::camera::{
    Camera3D, Plot3DViewport, ProjectionMode, Scene3DBuilder, CAMERA_3D_HELPER_TEXT,
    CAMERA_3D_MAX_PAN, CAMERA_3D_PERSPECTIVE_DISTANCE, CAMERA_3D_ZOOM_MAX, CAMERA_3D_ZOOM_MIN,
};
