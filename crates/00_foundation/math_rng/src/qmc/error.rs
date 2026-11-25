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

#[cfg(test)]
mod tests {
    use super::*;
    use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};

    struct CodeCapture {
        code: Option<&'static str>,
        fields: usize,
    }

    impl DiagnosticVisitor for CodeCapture {
        fn record_str(&mut self, key: &str, value: &str) {
            self.fields += 1;
            if key == "code" {
                self.code = match value {
                    "RNG-QMC-001" => Some("RNG-QMC-001"),
                    "RNG-QMC-002" => Some("RNG-QMC-002"),
                    "RNG-QMC-003" => Some("RNG-QMC-003"),
                    "RNG-QMC-004" => Some("RNG-QMC-004"),
                    _ => None,
                };
            }
        }

        fn record_display(&mut self, _key: &str, _value: &dyn fmt::Display) {
            self.fields += 1;
        }
    }

    #[test]
    fn qmc_error_visit_fields_is_zero_alloc_and_records_codes() {
        let cases = [
            (
                QmcError::InvalidDimension {
                    dimension: 0,
                    max_supported: 64,
                },
                "RNG-QMC-001",
            ),
            (QmcError::InvalidBase { base: 1 }, "RNG-QMC-002"),
            (
                QmcError::BufferLengthMismatch {
                    expected: 4,
                    actual: 2,
                },
                "RNG-QMC-003",
            ),
            (
                QmcError::StateLengthMismatch {
                    expected: 8,
                    actual: 3,
                },
                "RNG-QMC-004",
            ),
        ];

        let (total_fields, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            let mut sum = 0usize;
            for (err, expected_code) in cases {
                assert_eq!(err.severity(), Severity::Error);
                let mut capture = CodeCapture {
                    code: None,
                    fields: 0,
                };
                err.visit_fields(&mut capture);
                assert_eq!(capture.code, Some(expected_code));
                sum += capture.fields;
            }
            sum
        });
        assert!(total_fields >= 16);
    }
}
