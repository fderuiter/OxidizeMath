use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};
use thiserror::Error;

/// Biophysical parameter validation errors for Hodgkin-Huxley membrane conductance models.
#[derive(Error, Debug, PartialEq)]
pub enum HodgkinHuxleyError {
    /// Ion-channel gating variable $x \in \{n, m, h\}$ fell outside the unit interval $[0, 1]$.
    #[error("Invalid gating variable value: {0} (must be between 0 and 1)")]
    InvalidGatingVariable(f64),
    /// Maximal ionic or leak conductance $g$ was negative.
    #[error("Invalid conductance value: {0} (must be non-negative)")]
    InvalidConductance(f64),
}

impl Diagnostic for HodgkinHuxleyError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visitor.record_str("error_type", "HodgkinHuxleyError");
        visitor.record_display("description", self);
    }
}

pub use crate::biology::{kinetics::KineticsError, reaction_diffusion::ReactionDiffusionError};

/// Errors emitted by botanical morphogenesis kernels (`lsystem` and `phyllotaxis`).
#[derive(Error, Debug, Clone, PartialEq)]
pub enum BotanicalError {
    /// Requested primordia count is zero, non-finite, or exceeds the pre-allocated capacity.
    #[error("Primordia count {count} must be between 1 and {max}")]
    InvalidPrimordiaCount {
        /// Requested primordia count.
        count: usize,
        /// Maximum supported primordia count.
        max: usize,
    },
    /// Divergence angle is non-finite.
    #[error("Divergence angle {angle} rad must be finite")]
    InvalidDivergenceAngle {
        /// Supplied divergence angle in radians.
        angle: f64,
    },
    /// Radial scale factor is non-positive or non-finite.
    #[error("Radial scale factor {scale} must be finite and strictly positive")]
    InvalidScaleFactor {
        /// Supplied radial scale factor.
        scale: f64,
    },
    /// L-system string derivation exceeded the pre-allocated symbol tape bound.
    #[error("L-system derivation requested {requested} symbols, exceeding maximum tape size {max}")]
    SymbolTapeOverflow {
        /// Requested total symbol count.
        requested: usize,
        /// Maximum configured symbol tape capacity.
        max: usize,
    },
    /// L-system turtle bracket push exceeded the pre-allocated branch stack depth.
    #[error("L-system turtle bracket stack exceeded maximum depth {max_depth}")]
    TurtleStackOverflow {
        /// Maximum configured stack depth.
        max_depth: usize,
    },
    /// L-system string contained a closing bracket `]` with an empty branch stack.
    #[error("L-system string contains an unmatched closing bracket")]
    UnmatchedPopBracket,
    /// Grammar or turtle parameter is invalid or non-finite.
    #[error("Invalid L-system parameter {name}: {value}")]
    InvalidGrammarParameter {
        /// Parameter identifier.
        name: &'static str,
        /// Supplied invalid numeric value.
        value: f64,
    },
    /// L-system axiom cannot be empty.
    #[error("L-system axiom cannot be empty")]
    EmptyAxiom,
}

impl Diagnostic for BotanicalError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visitor.record_str("error_type", "BotanicalError");
        visitor.record_display("description", self);
    }
}

impl Diagnostic for KineticsError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visitor.record_str("error_type", "KineticsError");
        visitor.record_display("description", self);
    }
}

impl Diagnostic for ReactionDiffusionError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        match self {
            Self::Math(math_err) => math_err.visit_fields(visitor),
            Self::CellMaskLengthMismatch { expected, actual } => {
                visitor.record_usize("expected", *expected);
                visitor.record_usize("actual", *actual);
            }
            _ => {}
        }
        visitor.record_str("error_type", "ReactionDiffusionError");
        visitor.record_display("description", self);
    }
}

mod umbrella;
pub use umbrella::BiologyError;

#[cfg(test)]
mod tests {
    use super::*;
    use diagnostics::DiagnosticBus;
    use math_commons::error::MathError;
    use math_commons::math_kernel::types::Dimension;
    use std::error::Error as StdError;

    #[test]
    #[verified_engine::verified]
    fn test_biology_umbrella_and_wrapper_sources() {
        let sample_math = MathError::DimensionMismatch {
            expected: Dimension(2),
            actual: Dimension(3),
        };
        let rd = ReactionDiffusionError::from(sample_math);
        assert!(rd.source().is_some());

        let cases: Vec<BiologyError> = vec![
            HodgkinHuxleyError::InvalidConductance(-1.0).into(),
            KineticsError::NegativeSubstrateConcentration.into(),
            rd.into(),
            sample_math.into(),
            domain_applied::error::GameTheoryError::NonSquarePayoffMatrix { rows: 1, cols: 2 }
                .into(),
        ];

        let bus = DiagnosticBus::new();
        for err in &cases {
            assert!(err.source().is_some());
            assert_eq!(err.severity(), Severity::Error);
            assert_eq!(
                err.metadata().get("umbrella_type").map(String::as_str),
                Some("BiologyError")
            );
            bus.emit_diagnostic(err);
        }
    }
}
