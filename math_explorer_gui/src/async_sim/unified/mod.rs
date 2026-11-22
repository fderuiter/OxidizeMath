#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod canvas;
mod controls;
mod persistence;
pub mod tool;
pub use tool::*;

use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
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

    /// Returns `(grid_w, grid_h, flip_y)` when interactive `RasterBrushCanvas` sculpting is supported.
    fn raster_brush_grid() -> Option<(usize, usize, bool)>
    where
        Self: Sized,
    {
        None
    }

    /// Returns the default boolean obstacle mask for `raster_brush_grid` when initialized.
    fn default_obstacle_mask(_params: &HashMap<String, f64>) -> Option<Vec<bool>>
    where
        Self: Sized,
    {
        None
    }

    /// Validates an encoded RLE obstacle mask prior to applying state changes.
    fn validate_obstacle_rle(_encoded: &str) -> Result<Vec<bool>, String>
    where
        Self: Sized,
    {
        Err("Obstacle RLE masks are not supported for this model".to_string())
    }

    /// Applies a validated RLE obstacle mask onto this model instance.
    fn apply_obstacle_rle(&mut self, _encoded: &str) {}

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
}

#[allow(missing_docs)]
pub struct UnifiedSimRunner<M: UnifiedModel> {
    model: M,
    params: Arc<RwLock<HashMap<String, f64>>>,
    obstacle_rle: Arc<RwLock<Option<String>>>,
    steps_per_frame: usize,
}

impl<M: UnifiedModel> UnifiedSimRunner<M> {
    #[allow(missing_docs)]
    pub fn new(params: Arc<RwLock<HashMap<String, f64>>>) -> Self {
        Self::with_obstacle_mask(params, Arc::new(RwLock::new(None)))
    }

    #[allow(missing_docs)]
    pub fn with_obstacle_mask(
        params: Arc<RwLock<HashMap<String, f64>>>,
        obstacle_rle: Arc<RwLock<Option<String>>>,
    ) -> Self {
        let initial_params = params.read().unwrap_or_else(|e| e.into_inner()).clone();
        let mut model = M::new(&initial_params);
        if let Some(ref rle) = *obstacle_rle.read().unwrap_or_else(|e| e.into_inner()) {
            model.apply_obstacle_rle(rle);
        }
        Self {
            model,
            params,
            obstacle_rle,
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
                let mut rebuilt = M::new(&current_params);
                if let Some(ref rle) = *self.obstacle_rle.read().unwrap_or_else(|e| e.into_inner())
                {
                    rebuilt.apply_obstacle_rle(rle);
                }
                self.model = rebuilt;
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
