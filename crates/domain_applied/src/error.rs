use diagnostics::{Diagnostic, Severity};
use pure_math::pure_math::analysis::roots::AnalysisError;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum PharmacokineticsError {
    /// An invalid parameter was provided (e.g., negative volume or rate constant).
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// A numerical analysis method failed (e.g., root finding did not converge).
    #[error("Analysis error: {0}")]
    AnalysisError(#[from] AnalysisError),
}

#[derive(Debug, Clone, PartialEq, Error)]
#[allow(missing_docs)]
pub enum BatteryError {
    /// Depth of Discharge must be between 0.0 and 100.0.
    #[error("DepthOfDischarge must be between 0.0 and 100.0, got {0}")]
    InvalidDepthOfDischarge(f64),
    /// Capacity must be between 0.0 and 1.0.
    #[error("Capacity must be between 0.0 and 1.0, got {0}")]
    InvalidCapacity(f64),
    /// Cycles cannot be negative.
    #[error("Cycles cannot be negative, got {0}")]
    NegativeCycles(f64),
}

#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum GameTheoryError {
    /// Payoff matrix is not square.
    #[allow(missing_docs)]
    NonSquarePayoffMatrix { rows: usize, cols: usize },
    /// Invalid parameter value (e.g. frequency outside [0, 1]).
    #[allow(missing_docs)]
    InvalidParameter { name: String, value: f64 },
}

#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum EngineeringError {
    /// Invalid parameter value (e.g. TotalBits = 0).
    #[allow(missing_docs)]
    InvalidParameter { name: String, value: f64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[allow(missing_docs)]
pub enum LoraError {
    #[error("Ensemble is empty; cannot combine.")]
    #[allow(missing_docs)]
    EmptyEnsemble,
    #[error("Weights cannot be empty.")]
    #[allow(missing_docs)]
    EmptyWeights,
    #[error("The number of weights must match the number of modules in the ensemble.")]
    #[allow(missing_docs)]
    WeightModuleMismatch,
    #[error("Mismatched tensor shapes for the same key.")]
    #[allow(missing_docs)]
    TensorShapeMismatch,
    #[error("Mismatched keys between LoRA modules.")]
    #[allow(missing_docs)]
    KeyMismatch,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum IsosurfaceError {
    /// The grid dimensions are too small (must be at least 2x2x2).
    InvalidGrid(String),
    /// Wrapped centralized mathematical error.
    Math(math_commons::error::MathError),
}

impl Diagnostic for PharmacokineticsError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "error_type".to_string(),
            "PharmacokineticsError".to_string(),
        );
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl Diagnostic for BatteryError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "BatteryError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl Diagnostic for GameTheoryError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "GameTheoryError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl Diagnostic for EngineeringError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "EngineeringError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl Diagnostic for LoraError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "LoraError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl Diagnostic for IsosurfaceError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "IsosurfaceError".to_string());
        if let Self::Math(math_err) = self {
            map.extend(math_err.metadata());
        }
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl From<math_commons::error::MathError> for IsosurfaceError {
    fn from(err: math_commons::error::MathError) -> Self {
        IsosurfaceError::Math(err)
    }
}

impl std::fmt::Display for GameTheoryError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for GameTheoryError {}

impl std::fmt::Display for EngineeringError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for EngineeringError {}

impl std::fmt::Display for IsosurfaceError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for IsosurfaceError {
    #[verified_engine::verified]
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Math(err) => Some(err),
            Self::InvalidGrid(_) => None,
        }
    }
}

mod submodule_diagnostics;
mod umbrella;

pub use crate::applied::{
    algorithms::kalman::KalmanError, clinical_trials::types::ClinicalTrialError,
    isosurface::types::VoxelGridError,
};
pub use umbrella::AppliedError;

#[cfg(test)]
mod tests;
