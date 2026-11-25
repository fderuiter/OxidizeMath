//! Error definitions for `#![no_std]` Quasi-Monte Carlo views and sequences.
// [cite:clinical_trials_statistics]
#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]

use core::fmt;

/// Typed error returned by [`super::SobolView`], [`super::HaltonView`], and [`super::NiederreiterView`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
