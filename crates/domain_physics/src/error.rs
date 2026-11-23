use diagnostics::{Diagnostic, Severity};
use std::collections::HashMap;
use thiserror::Error;

use crate::physics::solid_state::types::ElectronVolts;

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub enum ChaosError {
    /// The simulation parameters are invalid (e.g., negative time step).
    InvalidParameter(String),
    /// Wrapped centralized mathematical error.
    Math(math_commons::error::MathError),
}

impl Diagnostic for ChaosError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "ChaosError".to_string());
        if let Self::Math(math_err) = self {
            map.extend(math_err.metadata());
        }
        map.insert("description".to_string(), self.to_string());
        map
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub enum StandardModelError {
    /// Energy scale must be positive.
    #[allow(missing_docs)]
    InvalidEnergyScale { scale: f64, context: String },
    /// Coupling constant must be positive.
    #[allow(missing_docs)]
    InvalidCouplingConstant { alpha: f64 },
    /// The coupling constant diverges at this scale (Landau Pole).
    #[allow(missing_docs)]
    LandauPole { scale: f64 },
    /// Invalid number of quark flavors (e.g., negative).
    #[allow(missing_docs)]
    InvalidFlavors { nf: f64 },
}

#[derive(Error, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
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
}

#[derive(Error, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub enum DoseFluenceError {
    #[error("Radius cannot be zero (singularity at r=0)")]
    #[allow(missing_docs)]
    Singularity,
    #[error("Radius must be non-negative")]
    #[allow(missing_docs)]
    NegativeRadius,
    #[error("Physical quantity must be non-negative: {0}")]
    #[allow(missing_docs)]
    InvalidPhysicalQuantity(String),
}

#[derive(Error, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub enum RadarError {
    /// Insufficient snapshots for MUSIC algorithm.
    #[error("Not enough snapshots to compute stable Covariance Matrix: {actual} < {required}")]
    #[allow(missing_docs)]
    InsufficientSnapshots { required: usize, actual: usize },

    /// Signal subspace dimension error (e.g. >= samples).
    #[error("Signal subspace dimension {subspace} equals or exceeds sample size {samples}")]
    #[allow(missing_docs)]
    InvalidSignalSubspace { samples: usize, subspace: usize },

    /// Invalid configuration parameters.
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Wrapped centralized mathematical error.
    #[error("Mathematical error: {0}")]
    Math(#[from] math_commons::error::MathError),
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub enum HighEnergyError {
    /// Mass must be positive.
    #[allow(missing_docs)]
    InvalidMass { mass: f64 },
    /// Radius must be greater than Schwarzschild radius.
    #[allow(missing_docs)]
    InvalidRadius { radius: f64, limit: f64 },
    /// Energy density cannot be negative.
    #[allow(missing_docs)]
    InvalidEnergyDensity { u_b: f64 },
    /// Lorentz factor must be >= 1.
    #[allow(missing_docs)]
    InvalidLorentzFactor { gamma: f64 },
    /// Power law index p must be > 1.
    #[allow(missing_docs)]
    InvalidPowerLawIndex { p: f64 },
    /// Adiabatic index must be > 1.
    #[allow(missing_docs)]
    InvalidAdiabaticIndex { gamma: f64 },
    /// Density must be positive.
    #[allow(missing_docs)]
    InvalidDensity { rho: f64 },
    /// Pressure cannot be negative.
    #[allow(missing_docs)]
    InvalidPressure { p: f64 },
    /// Velocity is invalid (e.g. >= c or < 0 if speed).
    #[allow(missing_docs)]
    InvalidVelocity { v: f64 },
    /// Momentum is invalid (e.g. m^2 + p^2 < 0 somehow? or just checking validity).
    InvalidMomentum,
    /// Invalid statistics parameters (e.g. negative counts).
    #[allow(missing_docs)]
    InvalidStatisticsParams { reason: String },
    /// Wrapped centralized mathematical error.
    Math(math_commons::error::MathError),
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub enum FluidError {
    /// Density must be strictly positive.
    #[allow(missing_docs)]
    InvalidDensity { value: f64 },
    /// Dynamic viscosity must be non-negative.
    #[allow(missing_docs)]
    InvalidViscosity { value: f64 },
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub enum StatMechError {
    /// Temperature must be non-negative (absolute zero is allowed as a limit).
    InvalidTemperature(f64),
    /// Chemical potential is invalid for the particle type and state (e.g., > Energy for Bosons).
    InvalidChemicalPotential {
        #[allow(missing_docs)]
        chemical_potential: f64,
        #[allow(missing_docs)]
        energy: f64,
        #[allow(missing_docs)]
        reason: String,
    },
    /// Wrapped centralized mathematical error.
    Math(math_commons::error::MathError),
}

impl Diagnostic for StandardModelError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "StandardModelError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl Diagnostic for SolidStateError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "SolidStateError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl Diagnostic for DoseFluenceError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "DoseFluenceError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl Diagnostic for RadarError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "RadarError".to_string());
        if let Self::Math(math_err) = self {
            map.extend(math_err.metadata());
        }
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl Diagnostic for HighEnergyError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "HighEnergyError".to_string());
        if let Self::Math(math_err) = self {
            map.extend(math_err.metadata());
        }
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl Diagnostic for FluidError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "FluidError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl Diagnostic for StatMechError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "StatMechError".to_string());
        if let Self::Math(math_err) = self {
            map.extend(math_err.metadata());
        }
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl std::fmt::Display for ChaosError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for ChaosError {
    #[verified_engine::verified]
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Math(err) => Some(err),
            Self::InvalidParameter(_) => None,
        }
    }
}

impl std::fmt::Display for StandardModelError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for StandardModelError {}

impl std::fmt::Display for HighEnergyError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl From<math_commons::error::MathError> for HighEnergyError {
    fn from(err: math_commons::error::MathError) -> Self {
        HighEnergyError::Math(err)
    }
}

impl std::error::Error for HighEnergyError {
    #[verified_engine::verified]
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Math(err) => Some(err),
            _ => None,
        }
    }
}

impl std::fmt::Display for FluidError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for FluidError {}

impl std::fmt::Display for StatMechError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl From<math_commons::error::MathError> for ChaosError {
    fn from(err: math_commons::error::MathError) -> Self {
        ChaosError::Math(err)
    }
}

impl std::error::Error for StatMechError {
    #[verified_engine::verified]
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Math(err) => Some(err),
            _ => None,
        }
    }
}

impl From<math_commons::error::MathError> for StatMechError {
    fn from(err: math_commons::error::MathError) -> Self {
        StatMechError::Math(err)
    }
}

mod umbrella;

pub use crate::physics::{medical::radar_gating::GatingError, nuclear::types::NuclearError};
pub use umbrella::PhysicsError;

#[cfg(test)]
mod tests;
