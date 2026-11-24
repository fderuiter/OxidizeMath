pub mod atlas_host;
mod audio_sync;
pub mod brush_codec;
pub mod brush_dispatch;
#[cfg(all(test, feature = "physics", feature = "biology"))]
mod brush_dispatch_tests;
pub mod callout_overlay;
pub mod challenge;
mod challenge_hud;
#[cfg(test)]
mod challenge_hud_tests;
pub mod clamp;
#[cfg(test)]
mod clamp_tests;
pub mod code_export;
pub mod code_export_modal;
#[cfg(all(
    test,
    feature = "pure_math",
    feature = "physics",
    feature = "applied",
    feature = "epidemiology"
))]
mod code_export_tests;
pub mod counterfactual;
pub mod decimate;
#[cfg(all(test, feature = "ai", feature = "physics", feature = "pure_math"))]
mod drag_bounds_and_rollback_tests;
mod export_3d;
mod export_audio;
mod export_dataset;
mod export_figure;
mod export_gif_job;
mod export_modal_host;
mod export_persistence;
mod export_queue;
#[cfg(test)]
pub(crate) use export_queue::deferred_export_tests;
#[cfg(test)]
mod export_queue_tests;
pub(crate) mod export_shader;
#[cfg(test)]
mod export_shader_tests;
pub mod export_types;
mod export_ui;
#[cfg(test)]
mod export_ui_tests;
pub mod fit_bridge;
mod history_ui;
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
#[cfg(test)]
mod playback_time_travel_tests;
pub mod plot_persistence;
#[cfg(test)]
mod plot_persistence_session_tests;
#[cfg(test)]
mod plot_persistence_tests;
pub mod recovery;
#[cfg(test)]
mod recovery_tests;
#[cfg(test)]
mod shared_points_perf_tests;
pub mod side_panel;
pub mod spatial_brush;
#[cfg(all(test, feature = "physics", feature = "biology"))]
mod spatial_brush_integration_tests;
#[cfg(test)]
mod spatial_brush_tests;
pub mod state_bounds;
#[cfg(test)]
mod state_bounds_tests;
pub mod state_guard;
#[cfg(test)]
mod state_guard_tests;
pub mod story_hud;
pub(crate) mod theory_links;
#[cfg(test)]
mod theory_links_tests;
mod theory_portal;
pub mod trace_overlay;
mod ui;
mod url_share;

pub use brush_codec::*;
pub use brush_dispatch::{PlotBrushController, PlotBrushFrame};
pub use callout_overlay::*;
pub use challenge::ChallengeEvaluation;
pub use clamp::*;
pub use code_export::*;
pub use code_export_modal::CodeExportTab;
pub use counterfactual::*;
pub use decimate::*;
pub use egui_plot::{ModalFrameContext, ModalLifecycle};
pub(crate) use export_queue::*;
pub use export_types::*;
pub use fit_bridge::ensure_custom_fit_bridge;
pub use history_ui::render_workspace_history_buttons;
pub use math_input::*;
#[cfg(any(feature = "pure_math", feature = "applied"))]
pub(crate) use overlay_sync::record_theory_portal_toggle;
pub use pane_scope::{
    is_button_activation_focused, is_pane_activation_shortcut_allowed, is_pane_shortcut_allowed,
    is_text_edit_focused, record_focused_widget_type_from_output, scoped_panel_id, InputMode,
    InteractionContext,
};
pub(crate) use persistence::framework_json_semantically_eq;
pub use playback::*;
pub use recovery::*;
pub use side_panel::{
    clear_pane_side_panel_records, pane_side_panel_records, PaneSidePanel, PaneSidePanelRecord,
    MAX_PANE_SIDE_PANEL_WIDTH_RATIO,
};
pub use spatial_brush::*;
pub use state_bounds::*;
pub use state_guard::*;
pub use story_hud::*;
pub use trace_overlay::*;

pub(super) use crate::accessibility::AccessibleHoverText;
pub use crate::starter_gallery::{ChallengeMetadata, StarterPresetMetadata};
use eframe::egui;
pub use overlay_sync::ToolMetadata;
pub(super) use overlay_sync::{
    FrameworkRevisionCache, FrameworkSyncState, UntrackedUrlDigestCache,
};
use scientific_metadata::theory::TheoryDescribable;

pub trait InteractiveTool {
    fn name(&self) -> &'static str;

    /// Provide theoretical context.
    fn theory(&self) -> &dyn TheoryDescribable;

    /// Show the tool at the root `egui::Context` level by delegating to [`Self::show_inside`].
    fn show(&mut self, ctx: &egui::Context) {
        pane_scope::render_default_tool_root(self, ctx);
    }

    /// Show the tool inside a scoped parent `egui::Ui` (supporting multi-pane layouts on a single root `Context`).
    ///
    /// # Accessibility Requirements for Custom `show_inside` Overrides
    ///
    /// - Allocate custom painter regions with [`egui::Sense::focusable_noninteractive`], [`egui::Sense::click`],
    ///   or [`egui::Sense::click_and_drag`] instead of `Sense::hover()` so `Tab` traversal reaches the visualization.
    /// - Render raster textures via `ui.add(egui::Image::new(texture).alt_text(...).sense(...))` so `AccessKit`
    ///   exposes `Role::Image` and non-empty alternative text.
    /// - Populate `AccessKit` metadata via `crate::accessibility::register_canvas_accessibility` and draw a
    ///   high-contrast focus ring when `response.has_focus()`.
    /// - When consuming arrow keys, set `EventFilter { tab: false, horizontal_arrows: true, vertical_arrows: true, escape: false }`
    ///   strictly while `response.has_focus()` so `Tab` and `Shift+Tab` never trap keyboard users.
    fn show_inside(&mut self, ui: &mut egui::Ui) {
        pane_scope::render_default_tool_inside(self, ui);
    }
    /// Optional: UI rendering for the tool's specific side-panel controls.
    fn show_ui(&mut self, _ui: &mut egui::Ui) {}
    /// Optional: Context-based drawing.
    fn draw(&mut self, _ui: &mut egui::Ui, _response: &egui::Response, _painter: &egui::Painter) {}
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

    /// Parameter-only revision counter for live URL synchronization (defaults to [`Self::state_revision`]).
    fn url_state_revision(&self) -> u64 {
        self.state_revision()
    }

    /// Returns the interactive canvas `Id` when this tool owns non-text canvas focus.
    fn focused_canvas_id(&self) -> Option<egui::Id> {
        None
    }

    /// Direct numeric parameter finiteness hook without JSON serialization.
    fn has_finite_parameters(&self) -> bool {
        true
    }

    /// Named scalar observables exposed to the Challenge HUD and `math_commons::ast` evaluator.
    fn observables(&self) -> Vec<(&'static str, f64)> {
        Vec::new()
    }

    /// Synchronizes declarative guided story or challenge callout pins onto the tool's plot.
    fn sync_callouts(&mut self, _callouts: &[CalloutSpec]) {}

    /// Candidate `(x_key, y_key)` parameter pairs for the framework-hosted 2D parameter atlas.
    fn atlas_axis_pairs(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }

    /// Current `(x_val, y_val)` coordinates of the tool on the selected `axis_keys` plane.
    fn active_atlas_coords(&self, _axis_keys: (&'static str, &'static str)) -> (f64, f64) {
        (0.0, 0.0)
    }

    /// Hashable signature of non-axis context parameters that invalidate cached atlas slices.
    fn atlas_context_signature(&self, _axis_keys: (&'static str, &'static str)) -> u64 {
        self.state_revision()
    }

    /// Builds a zero-allocation [`crate::async_sim::atlas::RegimeCartographer`] for `axis_keys`.
    fn build_atlas_cartographer(
        &self,
        _axis_keys: (&'static str, &'static str),
    ) -> Option<std::sync::Arc<dyn crate::async_sim::atlas::RegimeCartographer>> {
        None
    }

    /// Applies an interactive click or keyboard teleport event from the framework-hosted 2D atlas.
    fn apply_atlas_teleport(&mut self, _ev: crate::async_sim::atlas::AtlasTeleportEvent) {}

    /// Optional hook to serialize tool-specific parameters into a String (e.g. JSON).
    fn save_state(&self) -> Option<String> {
        None
    }

    /// Lightweight parameter-only snapshot hook used by `cache_valid_snapshot`.
    fn save_recovery_state(&self) -> Option<String> {
        self.save_state()
    }

    /// Parameter-only state serialization hook for URL sharing (omits trace histories).
    fn save_url_state(&self) -> Option<String> {
        self.save_recovery_state()
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
        export_persistence::export_tool_preset_json(self)
    }

    /// Parses and validates a JSON preset string against this tool's domain, applying configuration if valid.
    fn import_preset_json(&mut self, json_str: &str) -> Result<(), String> {
        export_persistence::import_tool_preset_json(self, json_str)
    }

    /// Declares which dataset, figure, and code export formats this tool supports.
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    /// Optional cold-path hook returning calibrated LaTeX, Rust, and Python code artifacts.
    fn export_code_artifact(&self) -> Option<CodeArtifact> {
        None
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

    /// Optional hook for texture-backed tools to expose their rendered 2D raster field for PNG, SVG, and GIF exports.
    fn export_raster_image(&self) -> Option<egui::ColorImage> {
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

    /// Optional hook returning the active counterfactual divergence snapshot for `SimulationFramework`.
    fn counterfactual_snapshot(&self) -> Option<DivergenceSnapshot> {
        None
    }

    /// Optional hook applying a `CounterfactualBranchEngine` action (`Fork`, `Swap`, `Clear`, `SetPerturbation`).
    fn apply_counterfactual_action(&mut self, _action: BranchControlAction) -> bool {
        false
    }

    /// Optional hook to sample multi-frame animation datasets for GIF export.
    fn sample_animation_frames(
        &mut self,
        num_frames: usize,
    ) -> Vec<egui_plot::export::DatasetFrame> {
        export_figure::sample_tool_animation_frames(self, num_frames)
    }
    /// Optional hook emitting a zero-allocation [`crate::audio::SonificationFrame`] for live Web Audio and `.wav` export.
    fn sonification_frame(&self) -> Option<crate::audio::SonificationFrame> {
        None
    }
    /// Returns persisted non-mute [`crate::audio::AcousticSettings`] for presets and share URLs.
    fn acoustic_settings(&self) -> crate::audio::AcousticSettings {
        crate::audio::AcousticSettings::default()
    }
    /// Applies persisted non-mute [`crate::audio::AcousticSettings`] without unmuting playback.
    fn set_acoustic_settings(&mut self, _settings: crate::audio::AcousticSettings) {}
    /// Samples deterministic multi-step [`crate::audio::SonificationFrame`] trajectories for 16-bit `.wav` export.
    fn sample_sonification_frames(
        &mut self,
        num_frames: usize,
    ) -> Vec<crate::audio::SonificationFrame> {
        audio_sync::sample_tool_sonification_frames(self, num_frames)
    }

    /// Returns a type-erased reference for concrete tool state inspection.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }

    /// Returns a mutable type-erased reference for concrete tool state updates (e.g. cross-tab handoffs).
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
    pub show_code_export_modal: bool,
    pub code_export_tab: CodeExportTab,
    pub export_options: egui_plot::export::ExportOptions,
    pub export_bg_dark: bool,
    pub export_gif_fps: u32,
    pub export_gif_frames: u32,
    pub export_status_message: Option<String>,
    pub export_status_severity: ExportStatusSeverity,
    pub domain: String,
    pub preset_dialog: crate::presets::PresetDialogState,
    pub(crate) export_queue: ExportQueue,
    gif_job: Option<export_gif_job::FrameworkGifJob>,
    /// Cached state snapshot before divergence for step-back rollback.
    pub last_valid_snapshot: Option<StateSnapshot>,
    /// Baseline parameters snapshot on tool load.
    pub baseline_tool_state: Option<String>,
    /// Last validated finite plot datasets.
    pub current_valid_plot_datasets: Vec<(String, egui_plot::SharedPoints)>,
    /// Last validated finite 3D world-coordinate plot datasets.
    pub current_valid_plot_datasets_3d: Vec<(String, Vec<[f64; 3]>)>,
    /// Whether divergence has been detected in the active tool.
    pub is_diverged: bool,
    /// Diagnostic report for divergence.
    pub diagnostic_report: Option<DiagnosticReport>,
    /// Simulation frame / step counter.
    pub step_counter: usize,
    /// Counterfactual shadow twin engine state synchronized with the active tool.
    pub branch_engine: CounterfactualBranchEngine,
    /// Start time of current active session.
    pub start_time: crate::web_io::PlatformInstant,
    /// Scoped Web Audio / gain-ramp / divergence-silence controller for this framework pane.
    pub audio_controller: crate::audio::AudioEngineController,
    /// Declarative guided story and AST challenge HUD state.
    pub challenge_hud: ChallengeHudState,
    /// Active inverse challenge contract displayed in the Challenge HUD, if any.
    pub active_challenge: Option<&'static ChallengeMetadata>,
    pub(super) challenge_was_passed: bool,
    pub(super) instance_id: u64,
    pub(super) sync_state: FrameworkSyncState,
    pub(super) rev_cache: FrameworkRevisionCache,
    pub atlas_host: atlas_host::FrameworkAtlasHost,
}

pub use crate::camera::{
    Camera3D, Plot3DViewport, ProjectionMode, Scene3DBuilder, CAMERA_3D_HELPER_TEXT,
    CAMERA_3D_MAX_PAN, CAMERA_3D_PERSPECTIVE_DISTANCE, CAMERA_3D_ZOOM_MAX, CAMERA_3D_ZOOM_MIN,
};
