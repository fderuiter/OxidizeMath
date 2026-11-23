#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod atlas;
mod canvas;
mod controls;
mod persistence;
pub mod tool;
pub use tool::*;

use crate::async_sim::{
    CheckpointParams, RegimeCartographer, SimCommand, SimulationRunner, StateSnapshot,
    TimeTravelCheckpoint,
};
use crate::framework::{CalloutManager, ChallengeEvaluation, MAX_SYNTHESIZED_PARAMS};
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

    /// Allow the model to draw custom visualization in the central panel.
    /// Return true if custom rendering was performed.
    fn custom_central_panel(_ui: &mut eframe::egui::Ui, _snapshot: Option<&StateSnapshot>) -> bool
    where
        Self: Sized,
    {
        false
    }

    /// Evaluate a fast scalar observable `(metric_label, value)` from `params`
    /// for parameter sensitivity analysis. Returns `None` when no fast scalar
    /// observable is available on the UI thread.
    fn evaluate_observable(_params: &HashMap<String, f64>) -> Option<(&'static str, f64)>
    where
        Self: Sized,
    {
        None
    }

    /// Opt-in keyframe capture stride `K` (`Some(1)` for compact ODEs, `Some(16)` for 2D grids).
    fn checkpoint_stride(&self) -> Option<usize> {
        None
    }

    /// Current deterministic step index for opted-in time-travel models.
    fn current_step(&self) -> usize {
        0
    }

    /// Captures a self-contained [`TimeTravelCheckpoint`] at `current_step()`.
    fn capture_checkpoint(&self, _params: &HashMap<String, f64>) -> Option<TimeTravelCheckpoint> {
        None
    }

    /// Captures active model parameters for mid-run parameter epoch tracking.
    fn capture_active_params(&self, params: &HashMap<String, f64>) -> Option<CheckpointParams> {
        self.checkpoint_stride()
            .map(|_| CheckpointParams::NamedMap(params.clone()))
    }

    /// Restores dynamical fields, active parameters, step index, and cloned `OxidizeRng`.
    fn restore_checkpoint(&mut self, _checkpoint: &TimeTravelCheckpoint) {}

    /// Reconstructs bounded trajectory histories from retained keyframes `..=kf_idx`.
    fn rebuild_history(&mut self, _front: &[TimeTravelCheckpoint], _back: &[TimeTravelCheckpoint]) {
    }

    /// Advances one zero-allocation sub-step during deterministic checkpoint replay.
    fn replay_step(&mut self, params: &CheckpointParams) {
        if let Some(map) = params.as_named_map() {
            self.step(map);
        }
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

#[allow(missing_docs)]
pub struct UnifiedSimRunner<M: UnifiedModel> {
    model: M,
    params: Arc<RwLock<HashMap<String, f64>>>,
    formula: Arc<RwLock<Option<CompiledFormulaPayload>>>,
    applied_formula_rev: Option<u64>,
    steps_per_frame: usize,
}

impl<M: UnifiedModel> UnifiedSimRunner<M> {
    #[allow(missing_docs)]
    pub fn new(params: Arc<RwLock<HashMap<String, f64>>>) -> Self {
        Self::with_formula(params, Arc::new(RwLock::new(None)))
    }

    /// Creates a runner sharing both `params` and an optional [`CompiledFormulaPayload`] slot.
    pub fn with_formula(
        params: Arc<RwLock<HashMap<String, f64>>>,
        formula: Arc<RwLock<Option<CompiledFormulaPayload>>>,
    ) -> Self {
        let initial_params = params.read().unwrap_or_else(|e| e.into_inner()).clone();
        let mut runner = Self {
            model: M::new(&initial_params),
            params,
            formula,
            applied_formula_rev: None,
            steps_per_frame: 5,
        };
        runner.sync_formula(&initial_params);
        runner
    }

    fn sync_formula_fields(
        model: &mut M,
        formula: &RwLock<Option<CompiledFormulaPayload>>,
        applied_formula_rev: &mut Option<u64>,
        current_params: &HashMap<String, f64>,
    ) {
        let guard = formula.read().unwrap_or_else(|e| e.into_inner());
        let next_rev = guard.as_ref().map(|p| p.revision);
        if next_rev != *applied_formula_rev {
            model.apply_compiled_formula(guard.as_ref(), current_params);
            *applied_formula_rev = next_rev;
        }
    }

    fn sync_formula(&mut self, current_params: &HashMap<String, f64>) {
        Self::sync_formula_fields(
            &mut self.model,
            &self.formula,
            &mut self.applied_formula_rev,
            current_params,
        );
    }
}

impl<M: UnifiedModel> SimulationRunner for UnifiedSimRunner<M> {
    fn process_command(&mut self, cmd: SimCommand) {
        let guard = self.params.read().unwrap_or_else(|e| e.into_inner());
        match cmd {
            SimCommand::SetSpeed(speed) => self.steps_per_frame = speed,
            SimCommand::Reset => {
                self.model.reset_model(&guard);
                self.applied_formula_rev = None;
                Self::sync_formula_fields(
                    &mut self.model,
                    &self.formula,
                    &mut self.applied_formula_rev,
                    &guard,
                );
            }
            _ => self.model.process_command(cmd, &guard),
        }
    }

    fn step(&mut self) {
        let guard = self.params.read().unwrap_or_else(|e| e.into_inner());
        Self::sync_formula_fields(
            &mut self.model,
            &self.formula,
            &mut self.applied_formula_rev,
            &guard,
        );
        self.model.step(&guard);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        self.model.get_snapshot()
    }

    fn get_steps_per_frame(&self) -> usize {
        self.steps_per_frame
    }

    fn is_finite_state(&self) -> bool {
        self.model.is_finite_state()
    }

    fn diagnostic_report(&self) -> Option<crate::framework::recovery::DiagnosticReport> {
        self.model.diagnostic_report().cloned()
    }

    fn checkpoint_stride(&self) -> Option<usize> {
        self.model.checkpoint_stride()
    }

    fn current_step(&self) -> usize {
        self.model.current_step()
    }

    fn capture_checkpoint(&self) -> Option<TimeTravelCheckpoint> {
        let guard = self.params.read().unwrap_or_else(|e| e.into_inner());
        self.model.capture_checkpoint(&guard)
    }

    fn capture_active_params(&self) -> Option<CheckpointParams> {
        let guard = self.params.read().unwrap_or_else(|e| e.into_inner());
        self.model.capture_active_params(&guard)
    }

    fn restore_checkpoint(&mut self, checkpoint: &TimeTravelCheckpoint) {
        self.model.restore_checkpoint(checkpoint);
    }

    fn rebuild_history(&mut self, front: &[TimeTravelCheckpoint], back: &[TimeTravelCheckpoint]) {
        self.model.rebuild_history(front, back);
    }

    fn replay_step(&mut self, params: &CheckpointParams) {
        self.model.replay_step(params);
    }
}
