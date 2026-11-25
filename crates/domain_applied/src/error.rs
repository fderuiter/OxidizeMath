use diagnostics::{Diagnostic, DiagnosticVisitor, Severity};
use pure_math::pure_math::analysis::roots::AnalysisError;
use thiserror::Error;

/// Errors returned by pharmacokinetic compartmental and root-finding solvers.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum PharmacokineticsError {
    /// An invalid parameter was provided (e.g., negative volume or rate constant).
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// A numerical analysis method failed (e.g., root finding did not converge).
    #[error("Analysis error: {0}")]
    AnalysisError(#[from] AnalysisError),
}

/// Errors returned when constructing bounded battery state-of-health quantities.
#[derive(Debug, Clone, PartialEq, Error)]
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
    /// Wrapped curve-fitting solver error.
    #[error("{0}")]
    Fit(#[from] math_commons::error::FitError),
}

/// Errors returned by game-theoretic equilibrium and mean-field solvers.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum GameTheoryError {
    /// Payoff matrix is not square.
    #[error("Payoff matrix must be square, got {rows}x{cols}")]
    NonSquarePayoffMatrix {
        /// Actual row count.
        rows: usize,
        /// Actual column count.
        cols: usize,
    },
    /// Invalid parameter value (e.g. frequency outside [0, 1]).
    #[error("Invalid game theory parameter {name}: {value}")]
    InvalidParameter {
        /// Parameter identifier.
        name: String,
        /// Rejected numeric value.
        value: f64,
    },
    /// Lower and upper bound vectors had different lengths.
    #[error(
        "Box constraint bound lengths must match: min has {min_len} elements, max has {max_len}"
    )]
    BoundsDimensionMismatch {
        /// Length of the lower-bound vector.
        min_len: usize,
        /// Length of the upper-bound vector.
        max_len: usize,
    },
    /// Lower bound exceeded upper bound or was non-finite along a dimension.
    #[error(
        "Invalid box bounds at dimension {dimension}: min ({min}) must be finite and <= max ({max})"
    )]
    InvalidBoxBounds {
        /// Zero-based coordinate index with invalid bounds.
        dimension: usize,
        /// Lower bound at `dimension`.
        min: f64,
        /// Upper bound at `dimension`.
        max: f64,
    },
    /// Wrapped centralized mathematical error.
    #[error("{0}")]
    Math(#[from] math_commons::error::MathError),
    /// Wrapped Monte Carlo estimation error.
    #[error("{0}")]
    MonteCarlo(#[from] pure_math::error::MonteCarloError),
}

/// Errors returned by engineering, CNC, UART, and FreeSurfer GLM solvers.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum EngineeringError {
    /// Invalid parameter value (e.g. TotalBits = 0).
    #[error("Invalid engineering parameter {name}: {value}")]
    InvalidParameter {
        /// Parameter identifier.
        name: String,
        /// Rejected numeric value.
        value: f64,
    },
    /// Design matrix normal equations `X^T * X` are singular and cannot be inverted.
    #[error("Design matrix normal equations X^T * X are singular and not invertible")]
    SingularDesignMatrix,
    /// Contrast variance denominator is zero or negative, leaving the t-statistic undefined.
    #[error("Contrast standard error denominator is zero; t-statistic is undefined")]
    ZeroContrastVariance,
    /// Matrix or vector dimensions are non-conformable.
    #[error("{context}: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Invariant context describing the mismatched operand.
        context: &'static str,
        /// Expected dimension length.
        expected: usize,
        /// Actual dimension length.
        actual: usize,
    },
    /// At least one 3D MRI volume axis had zero voxels.
    #[error("MRI volume dimensions ({}, {}, {}) must all be non-zero", dims[0], dims[1], dims[2])]
    ZeroVolumeDimension {
        /// Supplied `[nx, ny, nz]` voxel dimensions.
        dims: [usize; 3],
    },
    /// Isotropic MRI voxel spacing was non-finite or non-positive.
    #[error("MRI voxel_size ({voxel_size}) must be finite and strictly positive")]
    InvalidVoxelSize {
        /// Rejected voxel size in millimeters.
        voxel_size: f64,
    },
    /// One or more world-space origin coordinates of an MRI volume were non-finite.
    #[error(
        "MRI origin coordinates ({}, {}, {}) must all be finite",
        origin[0],
        origin[1],
        origin[2]
    )]
    NonFiniteOrigin {
        /// Rejected world-space origin `[x0, y0, z0]`.
        origin: [f64; 3],
    },
    /// Product `nx * ny * nz` overflowed `usize`.
    #[error(
        "MRI volume dimensions ({}, {}, {}) overflow usize capacity",
        dims[0],
        dims[1],
        dims[2]
    )]
    VolumeDimensionsOverflow {
        /// Supplied `[nx, ny, nz]` voxel dimensions.
        dims: [usize; 3],
    },
    /// Wrapped centralized mathematical error.
    #[error("{0}")]
    Math(#[from] math_commons::error::MathError),
}

/// Errors returned by LoRA adapter ensemble combination routines.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LoraError {
    /// No LoRA adapter modules were supplied to combine.
    #[error("Ensemble is empty; cannot combine.")]
    EmptyEnsemble,
    /// Empty weight slice supplied for LoRA adapter combination.
    #[error("Weights cannot be empty.")]
    EmptyWeights,
    /// Weight count differed from the number of LoRA modules in the ensemble.
    #[error("The number of weights must match the number of modules in the ensemble.")]
    WeightModuleMismatch,
    /// Corresponding parameter matrices across LoRA modules had incompatible dimensions.
    #[error("Mismatched tensor shapes for the same key.")]
    TensorShapeMismatch,
    /// LoRA modules did not expose identical parameter key sets.
    #[error("Mismatched keys between LoRA modules.")]
    KeyMismatch,
}

/// Errors returned by Marching Cubes isosurface mesh extraction.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum IsosurfaceError {
    /// The grid dimensions are too small (must be at least 2x2x2).
    #[error("Invalid isosurface grid: {0}")]
    InvalidGrid(String),
    /// Wrapped centralized mathematical error.
    #[error("{0}")]
    Math(#[from] math_commons::error::MathError),
}

impl Diagnostic for PharmacokineticsError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        if let Self::AnalysisError(err) = self {
            err.visit_fields(visitor);
        }
        visitor.record_str("error_type", "PharmacokineticsError");
        visitor.record_display("description", self);
    }
}

impl Diagnostic for BatteryError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        match self {
            Self::Fit(err) => err.severity(),
            _ => Severity::Error,
        }
    }
    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        if let Self::Fit(fit_err) = self {
            fit_err.visit_fields(visitor);
        }
        visitor.record_str("error_type", "BatteryError");
        visitor.record_display("description", self);
    }
}

impl Diagnostic for GameTheoryError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        match self {
            Self::Math(err) => err.severity(),
            Self::MonteCarlo(err) => err.severity(),
            _ => Severity::Error,
        }
    }
    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        match self {
            Self::Math(math_err) => math_err.visit_fields(visitor),
            Self::MonteCarlo(mc_err) => mc_err.visit_fields(visitor),
            _ => {}
        }
        visitor.record_str("error_type", "GameTheoryError");
        visitor.record_display("description", self);
    }
}

impl Diagnostic for EngineeringError {
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
        visitor.record_str("error_type", "EngineeringError");
        visitor.record_display("description", self);
    }
}

impl Diagnostic for LoraError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visitor.record_str("error_type", "LoraError");
        visitor.record_display("description", self);
    }
}

impl Diagnostic for IsosurfaceError {
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
        visitor.record_str("error_type", "IsosurfaceError");
        visitor.record_display("description", self);
    }
}
mod submodule_diagnostics;
mod umbrella;

pub use crate::applied::{
    algorithms::kalman::KalmanError, clinical_trials::types::ClinicalTrialError,
    isosurface::types::VoxelGridError, portfolio::PortfolioError,
};
pub use umbrella::AppliedError;

#[cfg(test)]
mod tests;
