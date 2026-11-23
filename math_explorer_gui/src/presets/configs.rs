#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Domain-specific simulation preset enums and configuration schemas.

use serde::{Deserialize, Serialize};

/// ODE System preset type enum.
#[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Copy, Debug)]
pub enum OdePreset {
    /// Exponential growth/decay model.
    Exponential,
    /// Harmonic oscillator spring model.
    HarmonicOscillator,
    /// Logistic population growth model.
    LogisticGrowth,
    /// Coupled 2x2 linear system x' = A * x with Poincaré trace-determinant classification.
    LinearSystem2D,
    /// Coupled 2D planar custom dynamic equations x' = f(t, x, y), y' = g(t, x, y).
    PlanarCustom,
    /// Custom dynamic scalar user equation y' = f(t, y).
    Custom,
}

impl OdePreset {
    /// All selectable ODE presets in sidebar order.
    pub const ALL: [Self; 6] = [
        Self::Exponential,
        Self::HarmonicOscillator,
        Self::LogisticGrowth,
        Self::LinearSystem2D,
        Self::PlanarCustom,
        Self::Custom,
    ];

    /// User-friendly display name of the ODE preset.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            OdePreset::Exponential => "y' = k*y (Exponential)",
            OdePreset::HarmonicOscillator => "y'' = -k*y (Harmonic Oscillator)",
            OdePreset::LogisticGrowth => "y' = r*y*(1 - y/K) (Logistic Growth)",
            OdePreset::LinearSystem2D => "x' = A*x (2x2 Linear / Poincaré)",
            OdePreset::PlanarCustom => "x' = f(t,x,y), y' = g(t,x,y) (2D Planar)",
            OdePreset::Custom => "y' = f(t, y) (Custom Equation)",
        }
    }

    /// Returns `true` when this preset evolves a 2D state vector and supports phase-plane portraits.
    #[must_use]
    pub fn is_2d(self) -> bool {
        matches!(
            self,
            Self::HarmonicOscillator | Self::LinearSystem2D | Self::PlanarCustom
        )
    }

    /// Returns `(first_time_label, second_time_label, phase_x_label, phase_y_label)`.
    #[must_use]
    pub fn state_labels(self) -> (&'static str, &'static str, &'static str, &'static str) {
        match self {
            Self::HarmonicOscillator => ("y(t)", "y'(t)", "y", "y'"),
            Self::LinearSystem2D | Self::PlanarCustom => ("x(t)", "y(t)", "x", "y"),
            Self::Exponential | Self::LogisticGrowth | Self::Custom => ("y(t)", "y'(t)", "y", "y'"),
        }
    }
}

/// Maximum allowed integration step count for ODE solvers.
pub const MAX_ODE_STEPS: usize = 200_000;

/// Maximum interactive parametric UV grid resolution for 3D surface tools (`120`).
pub const MAX_INTERACTIVE_SURFACE_RESOLUTION: usize = 120;

/// Maximum interactive 2D vector and conformal-contour grid density (`100`).
pub const MAX_INTERACTIVE_GRID_SIZE: usize = 100;

/// Computes a bounded ODE integration step count when `dt` and `total_time` are finite,
/// strictly positive, and require at most [`MAX_ODE_STEPS`] steps.
#[must_use]
pub fn checked_ode_step_count(dt: f64, total_time: f64) -> Option<usize> {
    if !dt.is_finite() || dt <= 0.0 || !total_time.is_finite() || total_time <= 0.0 {
        return None;
    }
    let steps = (total_time / dt).ceil();
    if !steps.is_finite() || !(0.0..=MAX_ODE_STEPS as f64).contains(&steps) {
        return None;
    }
    Some(steps as usize)
}

/// Reaction kinetics model selector for 2D morphogenesis presets.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MorphogenesisKineticsMode {
    /// Classical activator-inhibitor Turing spots and stripes (`SchnakenbergKinetics`).
    #[default]
    Schnakenberg,
    /// Autocatalytic substrate-depletion mitosis and coral waves (`GrayScottKinetics`).
    GrayScott,
}

/// Configuration struct for Replicator Dynamics evolutionary game parameters.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReplicatorPresetConfig {
    /// Payoff matrix (rows x cols).
    pub payoff_matrix: Vec<Vec<f64>>,
    /// Initial population proportions per strategy.
    pub initial_population: Vec<f64>,
    /// Total simulation duration.
    pub time_horizon: f64,
    /// Integration time step size.
    pub dt: f64,
    /// Descriptive strategy names.
    pub strategy_names: Vec<String>,
    /// Optional symbolic payoff cell formulas `(rows x cols)` evaluated via `CompiledExpr`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symbolic_payoffs: Option<Vec<Vec<String>>>,
    /// Shared parameter values for symbolic payoff cells (e.g. `v`, `c` in Hawk-Dove).
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub symbolic_params: std::collections::BTreeMap<String, f64>,
}

const MAX_REPLICATOR_STRATEGIES: usize = 16;
const MAX_STRATEGY_NAME_BYTES: usize = 128;
const MIN_REPLICATOR_DT: f64 = 1e-6;
const MAX_REPLICATOR_DT: f64 = 5.0;
const MAX_REPLICATOR_TIME_HORIZON: f64 = 10_000.0;
const MAX_REPLICATOR_PAYOFF_ABS: f64 = 1e4;

impl ReplicatorPresetConfig {
    /// Validates matrix dimensions, numeric domain and step bounds, and optional symbolic formulas.
    pub fn validate(&self) -> Result<(), String> {
        let rows = self.payoff_matrix.len();
        if rows == 0 || rows > MAX_REPLICATOR_STRATEGIES {
            return Err(format!(
                "Payoff matrix strategy count ({rows}) must be in 1..={MAX_REPLICATOR_STRATEGIES}"
            ));
        }
        for (i, row) in self.payoff_matrix.iter().enumerate() {
            if row.len() != rows {
                return Err(format!(
                    "Payoff matrix must be square (row {i} has length {}, expected {rows})",
                    row.len(),
                ));
            }
            for (j, &val) in row.iter().enumerate() {
                if !val.is_finite()
                    || !(-MAX_REPLICATOR_PAYOFF_ABS..=MAX_REPLICATOR_PAYOFF_ABS).contains(&val)
                {
                    return Err(format!(
                        "Payoff matrix entry ({i}, {j}) must be finite and within [-10000, 10000]"
                    ));
                }
            }
        }
        if self.strategy_names.len() != rows {
            return Err(format!(
                "Strategy names count ({}) does not match payoff matrix dimensions ({rows}x{rows})",
                self.strategy_names.len(),
            ));
        }
        if self
            .strategy_names
            .iter()
            .any(|name| name.len() > MAX_STRATEGY_NAME_BYTES)
        {
            return Err(format!(
                "Strategy names must be at most {MAX_STRATEGY_NAME_BYTES} bytes"
            ));
        }
        self.validate_population_and_time_bounds(rows)?;
        self.validate_symbolic(rows)
    }

    fn validate_population_and_time_bounds(&self, rows: usize) -> Result<(), String> {
        if self.initial_population.len() != rows {
            return Err(format!(
                "Initial population count ({}) does not match strategy count ({rows})",
                self.initial_population.len(),
            ));
        }
        if !self
            .initial_population
            .iter()
            .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
        {
            return Err(
                "Initial population proportions must be finite and within [0, 1]".to_string(),
            );
        }
        let pop_sum: f64 = self.initial_population.iter().copied().sum();
        if !pop_sum.is_finite() || pop_sum <= 0.0 {
            return Err("Initial population sum must be strictly positive".to_string());
        }
        if !self.dt.is_finite() || !(MIN_REPLICATOR_DT..=MAX_REPLICATOR_DT).contains(&self.dt) {
            return Err(format!(
                "Replicator dt ({}) must be finite and within [{MIN_REPLICATOR_DT}, {MAX_REPLICATOR_DT}]",
                self.dt
            ));
        }
        if !self.time_horizon.is_finite()
            || self.time_horizon <= 0.0
            || self.time_horizon > MAX_REPLICATOR_TIME_HORIZON
        {
            return Err(format!(
                "Replicator time_horizon ({}) must be finite and within (0, {MAX_REPLICATOR_TIME_HORIZON}]",
                self.time_horizon
            ));
        }
        if checked_ode_step_count(self.dt, self.time_horizon).is_none() {
            return Err(format!(
                "Replicator ODE step count (time_horizon / dt) exceeds maximum allowed limit ({MAX_ODE_STEPS})"
            ));
        }
        Ok(())
    }

    fn validate_symbolic(&self, rows: usize) -> Result<(), String> {
        match &self.symbolic_payoffs {
            Some(sym_rows) => {
                if sym_rows.len() != rows {
                    return Err(format!(
                        "Symbolic payoff matrix row count ({}) does not match dimensions ({rows}x{rows})",
                        sym_rows.len()
                    ));
                }
                for (i, row) in sym_rows.iter().enumerate() {
                    if row.len() != rows {
                        return Err(format!(
                            "Symbolic payoff row {i} has length {}, expected {rows}",
                            row.len()
                        ));
                    }
                    for (j, expr_str) in row.iter().enumerate() {
                        let trimmed = expr_str.trim();
                        if trimmed.is_empty() || trimmed.len() > 256 {
                            return Err(format!(
                                "Symbolic payoff cell ({i}, {j}) must be non-empty and <= 256 chars"
                            ));
                        }
                    }
                }
                if self.symbolic_params.len() > crate::framework::math_input::MAX_SYNTHESIZED_PARAMS
                {
                    return Err(format!(
                        "Symbolic parameter count ({}) exceeds maximum ({})",
                        self.symbolic_params.len(),
                        crate::framework::math_input::MAX_SYNTHESIZED_PARAMS
                    ));
                }
                for (name, &val) in &self.symbolic_params {
                    if !val.is_finite() || !(-1e4..=1e4).contains(&val) {
                        return Err(format!(
                            "Symbolic parameter '{name}' must be finite and within [-10000, 10000]"
                        ));
                    }
                }
                Ok(())
            }
            None => {
                if !self.symbolic_params.is_empty() {
                    return Err(
                        "symbolic_params cannot be specified without symbolic_payoffs".to_string(),
                    );
                }
                Ok(())
            }
        }
    }
}
