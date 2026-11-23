//! Domain-specific simulation preset configuration models and validation helpers.

use serde::{Deserialize, Serialize};

/// ODE System preset type enum.
#[derive(Serialize, Deserialize, PartialEq, Clone, Copy, Debug)]
pub enum OdePreset {
    /// Exponential growth/decay model.
    Exponential,
    /// Harmonic oscillator spring model.
    HarmonicOscillator,
    /// Logistic population growth model.
    LogisticGrowth,
    /// Custom dynamic user equation.
    Custom,
}

impl OdePreset {
    /// User-friendly display name of the ODE preset.
    pub fn name(&self) -> &'static str {
        match self {
            OdePreset::Exponential => "y' = k*y (Exponential)",
            OdePreset::HarmonicOscillator => "y'' = -k*y (Harmonic Oscillator)",
            OdePreset::LogisticGrowth => "y' = r*y*(1 - y/K) (Logistic Growth)",
            OdePreset::Custom => "y' = f(t, y) (Custom Equation)",
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

/// Reaction kinetics family selectable in [`MorphogenesisPresetConfig`].
#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Copy, Debug, Default)]
pub enum MorphogenesisKinetics {
    /// Schnakenberg activator-inhibitor Turing kinetics (`a`, `b`).
    #[default]
    Schnakenberg,
    /// Gray-Scott cubic autocatalysis kinetics (`f`, `k`).
    GrayScott,
}

impl MorphogenesisKinetics {
    /// User-facing display label for the kinetics family.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Schnakenberg => "Schnakenberg",
            Self::GrayScott => "Gray-Scott",
        }
    }
}

/// Compatibility alias for [`MorphogenesisKinetics`] used by Lenia and preset callers.
pub type MorphogenesisKineticsMode = MorphogenesisKinetics;

/// Default Gray-Scott feed rate `F` (Pearson mitosis regime).
pub const DEFAULT_GRAY_SCOTT_F: f64 = 0.0367;
/// Default Gray-Scott kill rate `k` (Pearson mitosis regime).
pub const DEFAULT_GRAY_SCOTT_K: f64 = 0.0649;

/// Serde default helper returning [`DEFAULT_GRAY_SCOTT_F`].
#[must_use]
pub const fn default_gray_scott_f() -> f64 {
    DEFAULT_GRAY_SCOTT_F
}

/// Serde default helper returning [`DEFAULT_GRAY_SCOTT_K`].
#[must_use]
pub const fn default_gray_scott_k() -> f64 {
    DEFAULT_GRAY_SCOTT_K
}

/// Serde default helper enabling automatic 2D CFL time-step clamping.
#[must_use]
pub const fn default_auto_dt() -> bool {
    true
}

/// Configuration struct for Turing and Gray-Scott Morphogenesis simulation parameters.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MorphogenesisPresetConfig {
    /// Active reaction kinetics family (`Schnakenberg` when omitted).
    #[serde(default, alias = "kinetics_mode")]
    pub kinetics: MorphogenesisKinetics,
    /// Reaction kinetic parameter `a` (Schnakenberg).
    pub a: f64,
    /// Reaction kinetic parameter `b` (Schnakenberg).
    pub b: f64,
    /// Dimensionless substrate feed rate `F` (Gray-Scott).
    #[serde(default = "default_gray_scott_f")]
    pub f: f64,
    /// Dimensionless autocatalyst kill rate `k` (Gray-Scott).
    #[serde(default = "default_gray_scott_k")]
    pub k: f64,
    /// Diffusion coefficient for species `u`.
    pub d_u: f64,
    /// Diffusion coefficient for species `v`.
    pub d_v: f64,
    /// Simulation time step size.
    pub dt: f64,
    /// Whether time stepping automatically substeps within the 2D CFL bound.
    #[serde(default = "default_auto_dt")]
    pub auto_dt: bool,
    /// Grid width.
    pub width: usize,
    /// Grid height.
    pub height: usize,
}

impl Default for MorphogenesisPresetConfig {
    fn default() -> Self {
        Self {
            kinetics: MorphogenesisKinetics::Schnakenberg,
            a: 0.1,
            b: 0.9,
            f: DEFAULT_GRAY_SCOTT_F,
            k: DEFAULT_GRAY_SCOTT_K,
            d_u: 1.0,
            d_v: 100.0,
            dt: 0.05,
            auto_dt: true,
            width: 100,
            height: 100,
        }
    }
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

impl ReplicatorPresetConfig {
    /// Validates matrix dimensions, strategy counts, and optional symbolic payoff formulas.
    pub fn validate(&self) -> Result<(), String> {
        let rows = self.payoff_matrix.len();
        if rows == 0 {
            return Err("Payoff matrix must not be empty".to_string());
        }
        for (i, row) in self.payoff_matrix.iter().enumerate() {
            if row.len() != rows {
                return Err(format!(
                    "Payoff matrix must be square (row {} has length {}, expected {})",
                    i,
                    row.len(),
                    rows
                ));
            }
        }
        if self.strategy_names.len() != rows {
            return Err(format!(
                "Strategy names count ({}) does not match payoff matrix dimensions ({}x{})",
                self.strategy_names.len(),
                rows,
                rows
            ));
        }
        if self.initial_population.len() != rows {
            return Err(format!(
                "Initial population count ({}) does not match strategy count ({})",
                self.initial_population.len(),
                rows
            ));
        }
        self.validate_symbolic(rows)
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
