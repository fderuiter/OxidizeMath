#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod atlas;
mod canvas;
mod controls;
mod persistence;
pub mod tool;
pub use tool::*;

use crate::async_sim::{RegimeCartographer, SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::MAX_SYNTHESIZED_PARAMS;
use math_commons::ast::CompiledExpr;
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;
use std::ops::RangeInclusive;
use std::sync::{Arc, RwLock};

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

    /// Whether `StateSnapshot::structured_data` carries a `ScalarField2D` for contours and 16-bit textures.
    fn has_scalar_field() -> bool
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

    fn sync_formula(&mut self, current_params: &HashMap<String, f64>) {
        let payload_opt = self
            .formula
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let next_rev = payload_opt.as_ref().map(|p| p.revision);
        if next_rev != self.applied_formula_rev {
            self.model
                .apply_compiled_formula(payload_opt.as_ref(), current_params);
            self.applied_formula_rev = next_rev;
        }
    }
}

impl<M: UnifiedModel> SimulationRunner for UnifiedSimRunner<M> {
    fn process_command(&mut self, cmd: SimCommand) {
        let current_params = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        match cmd {
            SimCommand::SetSpeed(speed) => self.steps_per_frame = speed,
            SimCommand::Reset => {
                self.model = M::new(&current_params);
                self.applied_formula_rev = None;
                self.sync_formula(&current_params);
            }
            _ => self.model.process_command(cmd, &current_params),
        }
    }

    fn step(&mut self) {
        let current_params = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        self.sync_formula(&current_params);
        self.model.step(&current_params);
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
}
