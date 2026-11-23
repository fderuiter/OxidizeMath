use eframe::egui;
use scientific_metadata::theory::TheoryDescribable;

use super::{
    export_figure, export_persistence, pane_scope, BranchControlAction, CalloutSpec,
    ChallengeEvaluation, CodeArtifact, DivergenceSnapshot, ExportCapabilities, InteractionContext,
    ShaderDialect, ToolStateBounds,
};

/// Core interactive visualization and simulation contract implemented by every tool in `math_explorer_gui`.
pub trait InteractiveTool {
    /// Human-readable display title of the tool.
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

    /// Normalized pointer hover hook invoked when the cursor moves across the tool canvas.
    fn on_hover(&mut self, _ctx: &InteractionContext) {}
    /// Normalized pointer drag hook invoked during active click-and-drag gestures.
    fn on_drag(&mut self, _ctx: &InteractionContext) {}
    /// Normalized pointer click hook invoked on primary or secondary canvas clicks.
    fn on_click(&mut self, _ctx: &InteractionContext) {}
    /// Normalized spatial brush hook invoked while painting continuous field perturbations.
    fn on_brush(&mut self, _ctx: &InteractionContext) {}
    /// Normalized multi-touch / trackpad zoom and pan gesture hook.
    fn on_gesture(&mut self, _ctx: &InteractionContext) {}
    /// Normalized keyboard event hook invoked while the tool canvas holds keyboard focus.
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

    /// Declares which dataset, figure, code, 3D, shader, and audio export formats this tool supports.
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

    /// Optional cold-path hook to synthesize deterministic stereo PCM16 `.wav` bytes (`1..=15 s` at `44.1`/`48 kHz`).
    fn export_wav_audio(&self, _sample_rate: u32, _duration_secs: f32) -> Result<Vec<u8>, String> {
        Err(format!(
            "WAV audio export is not supported for {}",
            self.name()
        ))
    }

    /// Optional cold-path hook to serialize a polyphonic SMF Format 1 `.mid` score (`1..=15 s`).
    fn export_midi_score(&self, _duration_secs: f32) -> Result<Vec<u8>, String> {
        Err(format!(
            "MIDI score export is not supported for {}",
            self.name()
        ))
    }

    /// Optional cold-path hook to serialize a Huygens-Fokker `(scl_text, kbm_text)` tuning pair.
    fn export_scala_tuning(&self) -> Result<(String, String), String> {
        Err(format!(
            "Scala tuning export is not supported for {}",
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

    /// Returns a type-erased reference for concrete tool state inspection.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }

    /// Returns a mutable type-erased reference for concrete tool state updates (e.g. cross-tab handoffs).
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
}
