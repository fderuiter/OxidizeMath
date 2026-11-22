#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod canvas;
mod controls;
pub mod tool;
pub use tool::*;

use crate::async_sim::{
    CheckpointParams, SimCommand, SimulationRunner, StateSnapshot, TimeTravelCheckpoint,
};
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

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
}

#[allow(missing_docs)]
pub struct UnifiedSimRunner<M: UnifiedModel> {
    model: M,
    params: Arc<RwLock<HashMap<String, f64>>>,
    steps_per_frame: usize,
}

impl<M: UnifiedModel> UnifiedSimRunner<M> {
    #[allow(missing_docs)]
    pub fn new(params: Arc<RwLock<HashMap<String, f64>>>) -> Self {
        let guard = params.read().unwrap_or_else(|e| e.into_inner());
        let model = M::new(&guard);
        drop(guard);
        Self {
            model,
            params,
            steps_per_frame: 5,
        }
    }
}

impl<M: UnifiedModel> SimulationRunner for UnifiedSimRunner<M> {
    fn process_command(&mut self, cmd: SimCommand) {
        let guard = self.params.read().unwrap_or_else(|e| e.into_inner());
        match cmd {
            SimCommand::SetSpeed(speed) => self.steps_per_frame = speed,
            SimCommand::Reset => {
                self.model = M::new(&guard);
            }
            _ => self.model.process_command(cmd, &guard),
        }
    }

    fn step(&mut self) {
        let guard = self.params.read().unwrap_or_else(|e| e.into_inner());
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
