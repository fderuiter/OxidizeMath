use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};
use thiserror::Error;

/// Errors returned when constructing bounded battery state-of-health quantities or calibrating aging curves.
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
    Fit(#[from] engine_opt::error::FitError),
}

/// Errors returned by CNC machining and asynchronous UART hardware calculators.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum HardwareError {
    /// Invalid hardware parameter value (e.g., non-positive cutting speed or zero frame bits).
    #[error("Invalid hardware parameter {name}: {value}")]
    InvalidParameter {
        /// Parameter identifier.
        name: String,
        /// Rejected numeric value.
        value: f64,
    },
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

impl Diagnostic for HardwareError {
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
        visitor.record_str("error_type", "HardwareError");
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

mod umbrella;
pub use umbrella::EngineeringDomainError;

#[cfg(test)]
mod tests;
