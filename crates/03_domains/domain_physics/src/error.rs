use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};
use thiserror::Error;

use crate::physics::solid_state::types::ElectronVolts;

/// Error variants raised by chaotic dynamical system simulations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ChaosError {
    /// The simulation parameters are invalid (e.g., negative time step).
    #[error("Invalid chaos simulation parameter: {0}")]
    InvalidParameter(String),
    /// Wrapped centralized mathematical error.
    #[error("{0}")]
    Math(#[from] math_commons::error::MathError),
}

impl Diagnostic for ChaosError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        match self {
            Self::Math(err) => err.severity(),
            _ => Severity::Error,
        }
    }
    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        if let Self::Math(math_err) = self {
            math_err.visit_fields(visitor);
        }
        visitor.record_str("error_type", "ChaosError");
        visitor.record_display("description", self);
    }
}

/// Error variants for Standard Model renormalization-group and Higgs-sector calculations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum StandardModelError {
    /// Energy scale must be positive.
    #[error("Invalid energy scale {scale} in {context}: must be positive")]
    InvalidEnergyScale {
        /// Rejected energy scale $\mu$.
        scale: f64,
        /// Caller context where the scale was evaluated.
        context: String,
    },
    /// Coupling constant must be positive.
    #[error("Invalid coupling constant alpha = {alpha}: must be positive")]
    InvalidCouplingConstant {
        /// Rejected coupling constant $\alpha$.
        alpha: f64,
    },
    /// The coupling constant diverges at this scale (Landau Pole).
    #[error("Coupling constant diverges at Landau pole scale {scale}")]
    LandauPole {
        /// Energy scale at which the running coupling hits a Landau pole.
        scale: f64,
    },
    /// Invalid number of quark flavors (e.g., negative).
    #[error("Invalid quark flavor count nf = {nf}")]
    InvalidFlavors {
        /// Rejected active quark flavor count $n_f$.
        nf: f64,
    },
    /// Spontaneous symmetry breaking requires a negative mass-squared parameter (`mu2 < 0`).
    #[error("Spontaneous symmetry breaking requires mu^2 < 0, got {mu2}")]
    UnbrokenSymmetry {
        /// Non-negative mass-squared parameter `mu^2`.
        mu2: f64,
    },
    /// Vacuum stability requires a positive quartic self-coupling (`lambda > 0`).
    #[error("Higgs potential stability requires lambda > 0, got {lambda}")]
    UnstableHiggsPotential {
        /// Non-positive quartic self-coupling `lambda`.
        lambda: f64,
    },
    /// Wrapped centralized mathematical error.
    #[error("{0}")]
    Math(#[from] math_commons::error::MathError),
}

/// Error variants for solid-state BCS gap solvers, Fock states, and lattice models.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum SolidStateError {
    /// The iterative solver failed to converge within the maximum number of iterations.
    #[error("Convergence failed after {0} iterations. Last value: {1}")]
    ConvergenceFailure(usize, ElectronVolts),
    /// A parameter provided to the model or solver was invalid.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    /// A general computation error occurred.
    #[error("Computation error: {0}")]
    ComputationError(String),
    /// Fock state mode index exceeded the number of allocated modes.
    #[error("Fock state index {index} is out of bounds for {len} modes")]
    FockIndexOutOfBounds {
        /// Out-of-bounds mode index.
        index: usize,
        /// Number of allocated Fock modes.
        len: usize,
    },
    /// Attempted to place more than one fermion into a single quantum state.
    #[error("Pauli exclusion principle violated: fermion occupation count {requested} exceeds 1")]
    PauliExclusionViolation {
        /// Invalid requested fermion occupation number.
        requested: u8,
    },
    /// Bosonic mode occupation reached the maximum representable count (`u8::MAX`).
    #[error("Boson occupation saturated at maximum representable count ({max})")]
    BosonOccupationOverflow {
        /// Maximum representable occupation count.
        max: u8,
    },
    /// Wrapped centralized mathematical error.
    #[error("{0}")]
    Math(#[from] math_commons::error::MathError),
}

/// Error variants for radiotherapy dose-kernel evaluation and ion-chamber fluence calibration.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum DoseFluenceError {
    /// Point-spread kernel evaluated at singular origin $r = 0$.
    #[error("Radius cannot be zero (singularity at r=0)")]
    Singularity,
    /// Radial distance $r$ was negative.
    #[error("Radius must be non-negative")]
    NegativeRadius,
    /// Non-negative dosimetric quantity received an invalid value.
    #[error("Physical quantity must be non-negative: {0}")]
    InvalidPhysicalQuantity(String),
    /// Ordinary least-squares linear calibration requires at least two paired samples.
    #[error("Linear calibration requires at least {minimum} data points, got {actual}")]
    InsufficientCalibrationPoints {
        /// Minimum required sample count.
        minimum: usize,
        /// Supplied sample count.
        actual: usize,
    },
    /// Input calibration voltages had zero variance, making the slope undefined.
    #[error("Zero predictor variance in linear calibration; cannot fit line")]
    ZeroCalibrationVariance,
    /// Wrapped centralized mathematical error.
    #[error("{0}")]
    Math(#[from] math_commons::error::MathError),
}

/// Error variants for mmWave FMCW radar signal processing, MUSIC DOA, and respiratory tracking.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum RadarError {
    /// Insufficient snapshots for MUSIC algorithm.
    #[error("Not enough snapshots to compute stable Covariance Matrix: {actual} < {required}")]
    InsufficientSnapshots {
        /// Minimum snapshot count required.
        required: usize,
        /// Provided snapshot count.
        actual: usize,
    },
    /// Signal subspace dimension error (e.g. >= samples).
    #[error("Signal subspace dimension {subspace} equals or exceeds sample size {samples}")]
    InvalidSignalSubspace {
        /// Total antenna / sample dimension.
        samples: usize,
        /// Requested signal-subspace rank.
        subspace: usize,
    },
    /// Invalid configuration parameters.
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
    /// Antenna weight slice length did not match the configured antenna array size.
    #[error("Antenna weight count ({actual}) must match antenna count ({expected})")]
    AntennaWeightCountMismatch {
        /// Configured antenna element count.
        expected: usize,
        /// Supplied weight vector length.
        actual: usize,
    },
    /// Invalid scalar parameter passed to a radar tracking filter.
    #[error("Invalid tracking filter parameter {parameter}: {value}")]
    InvalidTrackingParameter {
        /// Name of the invalid filter parameter.
        parameter: &'static str,
        /// Rejected parameter value.
        value: f64,
    },
    /// Wrapped centralized mathematical error.
    #[error("Mathematical error: {0}")]
    Math(#[from] math_commons::error::MathError),
}

/// Error variants for relativistic astrophysics, black-hole metrics, and radiation processes.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum HighEnergyError {
    /// Mass must be positive.
    #[error("Mass must be positive, got {mass}")]
    InvalidMass {
        /// Rejected non-positive mass $M$.
        mass: f64,
    },
    /// Radius must be greater than Schwarzschild radius.
    #[error("Radius {radius} must exceed Schwarzschild limit {limit}")]
    InvalidRadius {
        /// Requested radial coordinate $r$.
        radius: f64,
        /// Event-horizon / Schwarzschild lower bound $r_s$.
        limit: f64,
    },
    /// Energy density cannot be negative.
    #[error("Magnetic energy density cannot be negative, got {u_b}")]
    InvalidEnergyDensity {
        /// Rejected magnetic energy density $U_B$.
        u_b: f64,
    },
    /// Lorentz factor must be >= 1.
    #[error("Lorentz factor gamma must be at least 1, got {gamma}")]
    InvalidLorentzFactor {
        /// Rejected Lorentz factor $\gamma$.
        gamma: f64,
    },
    /// Power law index p must be > 1.
    #[error("Power-law spectral index p must exceed 1, got {p}")]
    InvalidPowerLawIndex {
        /// Rejected spectral index $p$.
        p: f64,
    },
    /// Adiabatic index must be > 1.
    #[error("Adiabatic index gamma must exceed 1, got {gamma}")]
    InvalidAdiabaticIndex {
        /// Rejected adiabatic index $\gamma$.
        gamma: f64,
    },
    /// Density must be positive.
    #[error("Density rho must be positive, got {rho}")]
    InvalidDensity {
        /// Rejected rest-mass density $\rho$.
        rho: f64,
    },
    /// Pressure cannot be negative.
    #[error("Pressure p cannot be negative, got {p}")]
    InvalidPressure {
        /// Rejected pressure $p$.
        p: f64,
    },
    /// Velocity is invalid (e.g. >= c or < 0 if speed).
    #[error("Relativistic velocity v is out of valid subluminal bounds: {v}")]
    InvalidVelocity {
        /// Rejected coordinate speed $v$.
        v: f64,
    },
    /// Momentum is invalid (e.g. m^2 + p^2 < 0 somehow? or just checking validity).
    #[error("Invalid relativistic four-momentum")]
    InvalidMomentum,
    /// Invalid statistics parameters (e.g. negative counts).
    #[error("Invalid counting statistics parameters: {reason}")]
    InvalidStatisticsParams {
        /// Human-readable validation failure reason.
        reason: String,
    },
    /// Wrapped centralized mathematical error.
    #[error("{0}")]
    Math(#[from] math_commons::error::MathError),
    /// Wrapped differential-geometry tensor error.
    #[error("{0}")]
    Tensor(#[from] pure_math::error::TensorError),
}

/// Error variants for continuum and Lattice Boltzmann fluid-dynamics simulations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum FluidError {
    /// Density must be strictly positive.
    #[error("Fluid density must be strictly positive, got {value}")]
    InvalidDensity {
        /// Rejected fluid density $\rho$.
        value: f64,
    },
    /// Dynamic viscosity must be non-negative.
    #[error("Dynamic viscosity must be non-negative, got {value}")]
    InvalidViscosity {
        /// Rejected viscosity $\mu$.
        value: f64,
    },
    /// Lattice Boltzmann grid dimensions overflowed addressable capacity or were empty.
    #[error("Invalid or overflowing Lattice Boltzmann grid dimensions ({width}x{height})")]
    GridDimensionsOverflow {
        /// Requested grid width.
        width: usize,
        /// Requested grid height.
        height: usize,
    },
    /// Wrapped centralized mathematical error.
    #[error("{0}")]
    Math(#[from] math_commons::error::MathError),
}

/// Error variants for statistical-mechanics ensembles and Bose/Fermi distributions.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum StatMechError {
    /// Temperature must be non-negative (absolute zero is allowed as a limit).
    #[error("Invalid thermodynamic temperature: {0} K")]
    InvalidTemperature(f64),
    /// Chemical potential is invalid for the particle type and state (e.g., > Energy for Bosons).
    #[error(
        "Invalid chemical potential ({chemical_potential}) for state energy ({energy}): {reason}"
    )]
    InvalidChemicalPotential {
        /// Rejected chemical potential $\mu$.
        chemical_potential: f64,
        /// Single-particle state energy $\varepsilon$.
        energy: f64,
        /// Physical constraint violated (e.g. Bose-Einstein $\mu > \varepsilon$).
        reason: String,
    },
    /// Wrapped differential-equation solver error.
    #[error("{0}")]
    Diffeq(#[from] engine_diffeq::DiffeqError),
    /// Wrapped partial-differential-equation solver error.
    #[error("{0}")]
    Pde(#[from] engine_pde::PdeError),
    /// Wrapped centralized mathematical error.
    #[error("{0}")]
    Math(#[from] math_commons::error::MathError),
}

impl Diagnostic for StandardModelError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        match self {
            Self::Math(err) => err.severity(),
            _ => Severity::Error,
        }
    }
    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        if let Self::Math(math_err) = self {
            math_err.visit_fields(visitor);
        }
        visitor.record_str("error_type", "StandardModelError");
        visitor.record_display("description", self);
    }
}

impl Diagnostic for SolidStateError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        match self {
            Self::Math(err) => err.severity(),
            _ => Severity::Error,
        }
    }
    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        if let Self::Math(math_err) = self {
            math_err.visit_fields(visitor);
        }
        visitor.record_str("error_type", "SolidStateError");
        visitor.record_display("description", self);
    }
}

impl Diagnostic for DoseFluenceError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        match self {
            Self::Math(err) => err.severity(),
            _ => Severity::Error,
        }
    }
    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        if let Self::Math(math_err) = self {
            math_err.visit_fields(visitor);
        }
        visitor.record_str("error_type", "DoseFluenceError");
        visitor.record_display("description", self);
    }
}

impl Diagnostic for RadarError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        match self {
            Self::Math(err) => err.severity(),
            _ => Severity::Error,
        }
    }
    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        if let Self::Math(math_err) = self {
            math_err.visit_fields(visitor);
        }
        visitor.record_str("error_type", "RadarError");
        visitor.record_display("description", self);
    }
}

impl Diagnostic for HighEnergyError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        match self {
            Self::Math(err) => err.severity(),
            Self::Tensor(err) => err.severity(),
            _ => Severity::Error,
        }
    }
    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        match self {
            Self::Math(math_err) => math_err.visit_fields(visitor),
            Self::Tensor(tensor_err) => tensor_err.visit_fields(visitor),
            _ => {}
        }
        visitor.record_str("error_type", "HighEnergyError");
        visitor.record_display("description", self);
    }
}

impl Diagnostic for FluidError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        match self {
            Self::Math(err) => err.severity(),
            _ => Severity::Error,
        }
    }
    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        if let Self::Math(math_err) = self {
            math_err.visit_fields(visitor);
        }
        visitor.record_str("error_type", "FluidError");
        visitor.record_display("description", self);
    }
}

impl Diagnostic for StatMechError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        match self {
            Self::Math(err) => err.severity(),
            Self::Diffeq(err) => err.severity(),
            Self::Pde(err) => err.severity(),
            _ => Severity::Error,
        }
    }
    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        match self {
            Self::Math(math_err) => math_err.visit_fields(visitor),
            Self::Diffeq(err) => err.visit_fields(visitor),
            Self::Pde(err) => err.visit_fields(visitor),
            _ => {}
        }
        visitor.record_str("error_type", "StatMechError");
        visitor.record_display("description", self);
    }
}
mod umbrella;

pub use crate::physics::{
    medical::radar_gating::GatingError, nuclear::types::NuclearError,
    quantum::hydrogenic::QuantumError, stat_mech::sandpile::SandpileError,
};
pub use umbrella::PhysicsError;

#[cfg(test)]
mod tests;
