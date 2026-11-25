//! Error definitions for `#![no_std]` Quasi-Monte Carlo views and sequences.
// [cite:clinical_trials_statistics]
#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]

use core::fmt;

/// Typed error returned by [`super::SobolView`], [`super::HaltonView`], and [`super::NiederreiterView`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum QmcError {
    /// Requested dimension is zero or exceeds `max_supported` (`64`).
    InvalidDimension {
        /// Requested dimension.
        dimension: usize,
        /// Maximum supported dimension (`64`).
        max_supported: usize,
    },
    /// Radical-inverse base is less than `2`.
    InvalidBase {
        /// Invalid base (`< 2`).
        base: u32,
    },
    /// Caller output or shift slice length does not match the configured QMC dimension.
    BufferLengthMismatch {
        /// Expected slice length (`dimension`).
        expected: usize,
        /// Provided slice length.
        actual: usize,
    },
    /// Caller state slice length does not match the configured QMC dimension.
    StateLengthMismatch {
        /// Expected state slice length (`dimension`).
        expected: usize,
        /// Provided state slice length.
        actual: usize,
    },
}

impl fmt::Display for QmcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InvalidDimension {
                dimension,
                max_supported,
            } => write!(
                f,
                "QMC dimension must be in 1..={max_supported}, got {dimension}"
            ),
            Self::InvalidBase { base } => {
                write!(f, "Radical inverse base must be >= 2, got {base}")
            }
            Self::BufferLengthMismatch { expected, actual } => write!(
                f,
                "QMC buffer length mismatch: expected {expected}, got {actual}"
            ),
            Self::StateLengthMismatch { expected, actual } => write!(
                f,
                "QMC state slice length mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl core::error::Error for QmcError {}

impl math_diagnostics::Diagnostic for QmcError {
    fn severity(&self) -> math_diagnostics::Severity {
        math_diagnostics::Severity::Error
    }

    fn visit_fields(&self, visitor: &mut dyn math_diagnostics::DiagnosticVisitor) {
        visitor.record_str("error_type", "QmcError");
        visitor.record_display("description", self);
        match *self {
            Self::InvalidDimension {
                dimension,
                max_supported,
            } => {
                visitor.record_str("code", "RNG-QMC-001");
                visitor.record_usize("dimension", dimension);
                visitor.record_usize("max_supported", max_supported);
            }
            Self::InvalidBase { base } => {
                visitor.record_str("code", "RNG-QMC-002");
                visitor.record_u64("base", u64::from(base));
            }
            Self::BufferLengthMismatch { expected, actual } => {
                visitor.record_str("code", "RNG-QMC-003");
                visitor.record_usize("expected", expected);
                visitor.record_usize("actual", actual);
            }
            Self::StateLengthMismatch { expected, actual } => {
                visitor.record_str("code", "RNG-QMC-004");
                visitor.record_usize("expected", expected);
                visitor.record_usize("actual", actual);
            }
        }
    }
}
