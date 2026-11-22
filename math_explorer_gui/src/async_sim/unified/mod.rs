#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod canvas;
mod controls;
mod state_io;
pub mod tool;
pub use tool::*;

use crate::async_sim::{SimCommand, SimulationController, SimulationRunner, StateSnapshot};
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Interactive central-panel context granting `UnifiedModel` implementations access to the
/// simulation worker controller, active snapshot, shared parameters, and custom series state.
#[allow(missing_docs)]
pub struct UnifiedCanvasContext<'a> {
    pub controller: &'a mut SimulationController,
    pub stepped_snapshot: Option<&'a StateSnapshot>,
    pub params: &'a Arc<RwLock<HashMap<String, f64>>>,
    pub custom_series: &'a Arc<RwLock<HashMap<String, Vec<f64>>>>,
}

impl UnifiedCanvasContext<'_> {
    /// Returns the active frame snapshot, preferring any stepped export snapshot.
    #[must_use]
    pub fn snapshot(&self) -> Option<&StateSnapshot> {
        self.stepped_snapshot
            .or_else(|| self.controller.latest_snapshot())
    }
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

    /// Return the default starting value for `name`.
    fn default_parameter(name: &str, constraint: &ParameterConstraint) -> f64
    where
        Self: Sized,
    {
        Self::parameter_choices(name)
            .first()
            .map_or((constraint.min + constraint.max) / 2.0, |&(_, v)| v)
    }

    /// Return true when slider edits to `name` should dispatch `SimCommand::SetWavepacket`
    /// without resetting sculpted custom series state.
    fn is_wavepacket_parameter(_name: &str) -> bool
    where
        Self: Sized,
    {
        false
    }

    /// Return true when clicking `action` should clear persisted `custom_series` entries.
    fn clears_custom_series_on_action(_action: &str) -> bool
    where
        Self: Sized,
    {
        false
    }

    /// Optional `(series_key, expected_len, min_val, max_val)` schemas for custom series state.
    fn custom_series_schema() -> &'static [(&'static str, usize, f64, f64)]
    where
        Self: Sized,
    {
        &[]
    }

    /// Apply persisted custom series vectors (such as `"custom_potential"`) after `new` or `Reset`.
    fn apply_custom_series(
        &mut self,
        _series: &HashMap<String, Vec<f64>>,
        _params: &HashMap<String, f64>,
    ) {
    }

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

    /// Interactive central-panel hook with controller, parameter, and custom series access.
    fn custom_central_panel_interactive(
        ui: &mut eframe::egui::Ui,
        ctx: &mut UnifiedCanvasContext<'_>,
    ) -> bool
    where
        Self: Sized,
    {
        Self::custom_central_panel(ui, ctx.snapshot())
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
}

#[allow(missing_docs)]
pub struct UnifiedSimRunner<M: UnifiedModel> {
    model: M,
    params: Arc<RwLock<HashMap<String, f64>>>,
    custom_series: Arc<RwLock<HashMap<String, Vec<f64>>>>,
    steps_per_frame: usize,
}

impl<M: UnifiedModel> UnifiedSimRunner<M> {
    #[allow(missing_docs)]
    pub fn new(params: Arc<RwLock<HashMap<String, f64>>>) -> Self {
        Self::new_with_series(params, Arc::new(RwLock::new(HashMap::new())))
    }

    #[allow(missing_docs)]
    pub fn new_with_series(
        params: Arc<RwLock<HashMap<String, f64>>>,
        custom_series: Arc<RwLock<HashMap<String, Vec<f64>>>>,
    ) -> Self {
        let initial_params = params.read().unwrap_or_else(|e| e.into_inner()).clone();
        let initial_series = custom_series
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let mut model = M::new(&initial_params);
        if !initial_series.is_empty() {
            model.apply_custom_series(&initial_series, &initial_params);
        }
        Self {
            model,
            params,
            custom_series,
            steps_per_frame: 5,
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
                let mut model = M::new(&current_params);
                let current_series = self
                    .custom_series
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                if !current_series.is_empty() {
                    model.apply_custom_series(&current_series, &current_params);
                }
                self.model = model;
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
