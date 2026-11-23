#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod atlas;
mod canvas;
mod controls;
mod persistence;
pub mod sculpt;
pub mod tool;
pub use sculpt::UnifiedSimRunner;
pub use tool::*;

use crate::async_sim::{RegimeCartographer, SculptStatePayload, SimCommand, StateSnapshot};
use crate::framework::{
    BrushStroke, CalloutManager, ChallengeEvaluation, SpatialBrushKit, MAX_SYNTHESIZED_PARAMS,
};
use eframe::egui;
use math_commons::ast::CompiledExpr;
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;
use std::ops::RangeInclusive;
use std::sync::{Arc, RwLock};

/// Optional UI-side extension for [`UnifiedModel`] tools supporting inverse challenges,
/// custom raster sculpting state, keyboard canvas cursors, and automated plot callouts.
#[allow(missing_docs)]
pub trait UnifiedUiExtension: Send {
    fn save_extra_state(&self, _params: &HashMap<String, f64>) -> Option<String> {
        None
    }

    fn try_load_extra_state(
        &mut self,
        _state: &str,
        _params: &mut HashMap<String, f64>,
    ) -> Option<Result<Vec<SimCommand>, String>> {
        None
    }

    fn apply_challenge(
        &mut self,
        challenge_id: Option<&str>,
        _params: &mut HashMap<String, f64>,
    ) -> Result<Vec<SimCommand>, String> {
        if let Some(id) = challenge_id {
            Err(format!("Challenge '{id}' is not supported by this model"))
        } else {
            Ok(Vec::new())
        }
    }

    fn active_challenge_id(&self) -> Option<&'static str> {
        None
    }

    fn evaluate_challenge(
        &self,
        _latest_snapshot: Option<&StateSnapshot>,
    ) -> Option<ChallengeEvaluation> {
        None
    }

    fn is_parameter_locked(&self, _param_name: &str) -> bool {
        false
    }

    fn on_reset(&mut self, _params: &HashMap<String, f64>) {}

    fn on_brush(&mut self, _cx: i32, _cy: i32, _r: i32, _is_obstacle: bool) -> bool {
        true
    }

    fn on_brush_stroke(&mut self, _stroke: &BrushStroke) -> bool {
        true
    }

    fn sync_sculpt_payload(&self, _sculpt: &mut SculptStatePayload) {}

    fn on_custom_action(&mut self, _action: &str) {}

    fn draw_side_panel_controls(
        &mut self,
        _ui: &mut egui::Ui,
        _latest_snapshot: Option<&StateSnapshot>,
    ) -> Vec<SimCommand> {
        Vec::new()
    }

    fn handle_canvas_overlay(
        &mut self,
        _ctx: &egui::Context,
        _visuals: &egui::Visuals,
        _plot_ui: &mut egui_plot::PlotUi,
    ) -> Vec<SimCommand> {
        Vec::new()
    }

    fn sync_callouts(
        &self,
        _latest_snapshot: Option<&StateSnapshot>,
        _callouts: &mut CalloutManager,
    ) {
    }

    fn state_revision_salt(&self) -> u64 {
        0
    }
}

/// Specification for an opt-in custom formula input bar on a [`UnifiedModel`].
#[derive(Clone, Debug)]
pub struct UnifiedFormulaSpec {
    /// Left-hand side equation label (e.g. `"V(x)"`).
    pub lhs_label: &'static str,
    /// Initial default expression string.
    pub default_expr: &'static str,
    /// Coordinate axis variable names bound to leading bytecode slots (e.g. `&["x"]`).
    pub axes: &'static [&'static str],
    /// Curated `(display_name, expression)` presets.
    pub presets: &'static [(&'static str, &'static str)],
    /// Slider range bounds for auto-synthesized free parameters.
    pub param_range: RangeInclusive<f64>,
    /// Default value assigned to newly discovered free parameters.
    pub param_default: f64,
}

/// `Arc`-shared compiled formula payload streamed to [`UnifiedSimRunner`] without worker-side parsing.
#[derive(Clone, Debug)]
pub struct CompiledFormulaPayload {
    /// Monotonic revision incremented on formula or parameter slider edits.
    pub revision: u64,
    /// Compiled postfix AST bytecode.
    pub compiled: Arc<CompiledExpr>,
    /// Stack-allocated parameter values aligned with `compiled`.
    pub params: [f64; MAX_SYNTHESIZED_PARAMS],
    /// Number of active parameter slots in `params`.
    pub param_len: usize,
}

#[allow(missing_docs)]
pub trait UnifiedModel:
    Send
    + 'static
    + scientific_metadata::theory::TheoryDescribable
    + crate::framework::recovery::FiniteStateGuard
{
    /// Initialize the model given the starting parameters.
    fn new(params: &HashMap<String, f64>) -> Self
    where
        Self: Sized;

    /// Reset simulation state on [`SimCommand::Reset`]. Defaults to reconstructing via [`Self::new`].
    fn reset_model(&mut self, params: &HashMap<String, f64>)
    where
        Self: Sized,
    {
        *self = Self::new(params);
    }

    /// Optional UI extension for inverse challenges, keyboard sculpting, and custom serialization.
    fn create_ui_extension() -> Option<Box<dyn UnifiedUiExtension>>
    where
        Self: Sized,
    {
        None
    }

    /// Step the simulation forward one tick.
    fn step(&mut self, params: &HashMap<String, f64>);

    /// Extract a visual snapshot of the state.
    fn get_snapshot(&self) -> StateSnapshot;

    /// Process any custom commands (e.g. ApplyBrush). Default is no-op.
    fn process_command(&mut self, _cmd: SimCommand, _params: &HashMap<String, f64>) {}

    /// Return parameter definitions for automatic UI generation.
    fn parameters() -> HashMap<String, ParameterConstraint>
    where
        Self: Sized;

    /// Return a user-facing display label for a parameter key.
    fn parameter_label(name: &str) -> String
    where
        Self: Sized,
    {
        name.split('_')
            .filter(|part| !part.is_empty())
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
                    None => String::new(),
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Return discrete `(label, numeric_value)` choices when a parameter represents categorical options.
    fn parameter_choices(_name: &str) -> &'static [(&'static str, f64)]
    where
        Self: Sized,
    {
        &[]
    }

    /// Opt-in custom mathematical expression input bar specification.
    fn formula_spec() -> Option<UnifiedFormulaSpec>
    where
        Self: Sized,
    {
        None
    }

    /// Applies an updated [`CompiledFormulaPayload`] (or `None` when disabled) to rebuild model operators.
    fn apply_compiled_formula(
        &mut self,
        _formula: Option<&CompiledFormulaPayload>,
        _params: &HashMap<String, f64>,
    ) {
    }

    /// The name of the tool.
    fn name() -> &'static str
    where
        Self: Sized;

    /// Return the domain category of the model.
    fn domain() -> &'static str
    where
        Self: Sized,
    {
        ""
    }

    /// Return presets that the model supports.
    fn presets() -> Vec<(&'static str, HashMap<String, f64>)>
    where
        Self: Sized,
    {
        Vec::new()
    }

    /// Return custom actions that the model supports.
    fn custom_actions() -> Vec<&'static str>
    where
        Self: Sized,
    {
        Vec::new()
    }

    /// Whether this model produces a 2D raster pixel grid rather than tabular `(X, Y)` series.
    fn is_raster_model() -> bool
    where
        Self: Sized,
    {
        false
    }

    /// Optional hook to build typed [`crate::framework::ExportTable`]s from a [`StateSnapshot`] for raster models.
    fn export_tables_from_snapshot(
        _snapshot: &StateSnapshot,
        _params: &HashMap<String, f64>,
    ) -> Vec<crate::framework::ExportTable>
    where
        Self: Sized,
    {
        Vec::new()
    }

    /// Whether this model supports interactive [`SpatialBrushKit`] sculpting and persistence.
    fn supports_spatial_brush() -> bool
    where
        Self: Sized,
    {
        false
    }

    /// Returns the default [`SpatialBrushKit`] configuration for this model.
    fn default_brush_kit() -> SpatialBrushKit
    where
        Self: Sized,
    {
        SpatialBrushKit::default()
    }

    /// Renders optional brush controls and live diagnostic readouts in the left side panel.
    fn draw_brush_controls(
        _ui: &mut eframe::egui::Ui,
        _brush_kit: &mut SpatialBrushKit,
        _params: &Arc<RwLock<HashMap<String, f64>>>,
        _sculpt_state: &Arc<RwLock<SculptStatePayload>>,
        _snapshot: Option<&StateSnapshot>,
    ) -> Vec<SimCommand>
    where
        Self: Sized,
    {
        Vec::new()
    }

    /// Updates UI-side `sculpt_state` synchronously when a `BrushStroke` is applied so
    /// `save_state()` immediately reflects sculpted masks/potentials.
    fn record_ui_brush_stroke(
        _stroke: &BrushStroke,
        _params: &HashMap<String, f64>,
        _formula: Option<&CompiledFormulaPayload>,
        _sculpt_state: &mut SculptStatePayload,
    ) where
        Self: Sized,
    {
    }

    /// Restores or exports sculpted state on the worker model instance.
    fn apply_sculpt_payload(
        &mut self,
        _payload: &SculptStatePayload,
        _params: &HashMap<String, f64>,
    ) {
    }

    /// Exports current worker model sculpt state into `payload`.
    fn export_sculpt_payload(&self, _params: &HashMap<String, f64>) -> SculptStatePayload {
        SculptStatePayload::default()
    }
    /// Allow the model to draw custom visualization in the central panel.
    /// Return true if custom rendering was performed.
    fn custom_central_panel(_ui: &mut eframe::egui::Ui, _snapshot: Option<&StateSnapshot>) -> bool
    where
        Self: Sized,
    {
        false
    }

    /// Allow the model to draw custom visualization with [`SpatialBrushKit`] sampling,
    /// returning `(handled, emitted_strokes)`.
    fn custom_central_panel_with_brush(
        ui: &mut eframe::egui::Ui,
        snapshot: Option<&StateSnapshot>,
        _brush_kit: &mut SpatialBrushKit,
    ) -> (bool, Vec<BrushStroke>)
    where
        Self: Sized,
    {
        (Self::custom_central_panel(ui, snapshot), Vec::new())
    }

    /// Evaluate a fast scalar observable `(metric_label, value)` from `params` for sensitivity analysis.
    fn evaluate_observable(_params: &HashMap<String, f64>) -> Option<(&'static str, f64)>
    where
        Self: Sized,
    {
        None
    }

    /// Returns a stable tool identifier when this unified model emits live
    /// [`crate::audio::SonificationFrame`]s.
    fn sonification_tool_id() -> Option<&'static str>
    where
        Self: Sized,
    {
        None
    }

    /// Optional `(x_key, y_key)` parameter pairs exposed in a 2D [`crate::async_sim::ParameterAtlasController`].
    fn atlas_axis_pairs() -> &'static [(&'static str, &'static str)]
    where
        Self: Sized,
    {
        &[]
    }

    /// Constructs a zero-allocation [`RegimeCartographer`] for `axis_keys` when supported.
    fn build_atlas_cartographer(
        _params: &HashMap<String, f64>,
        _axis_keys: (&'static str, &'static str),
    ) -> Option<Arc<dyn RegimeCartographer>>
    where
        Self: Sized,
    {
        None
    }
}
