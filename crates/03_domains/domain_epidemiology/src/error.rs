use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};
use thiserror::Error;

/// Domain error variants for compartmental ODE, next-generation matrix, and final-size solvers.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum EpidemiologyError {
    /// Matrix V (Transition Matrix) is singular and cannot be inverted.
    #[error("Transition matrix V is singular and cannot be inverted")]
    SingularTransitionMatrix,
    /// Invalid Parameter (e.g., negative rate).
    #[error("Invalid epidemiological parameter {name}: {value}")]
    InvalidParameter {
        /// Identifier of the rejected epidemiological parameter.
        name: String,
        /// Rejected numerical parameter value.
        value: f64,
    },
    /// Missing Parameter (e.g., required field not set in builder).
    #[error("Missing required epidemiological parameter: {name}")]
    MissingParameter {
        /// Name of the omitted required builder parameter.
        name: String,
    },
    /// Basic reproduction number `R0` was non-positive or non-finite.
    #[error("Basic reproduction number R0 must be finite and positive, got {r0}")]
    InvalidReproductionNumber {
        /// Rejected `R0` parameter value.
        r0: f64,
    },
    /// Newton-Raphson derivative approached zero during final-size iteration.
    #[error("Final-size solver derivative ({derivative}) approached zero at S_inf = {x}")]
    NearZeroDerivative {
        /// Current `S_inf` iterate where the derivative vanished.
        x: f64,
        /// Evaluated derivative `f'(x)`.
        derivative: f64,
    },
    /// Newton-Raphson final-size solver failed to converge within maximum iterations.
    #[error("Final-size Newton-Raphson solver failed to converge within {iterations} iterations")]
    ConvergenceFailure {
        /// Maximum iteration count exhausted by the solver.
        iterations: usize,
    },
    /// Wrapped centralized mathematical error.
    #[error("{0}")]
    Math(#[from] math_commons::error::MathError),
}

impl Diagnostic for EpidemiologyError {
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
        visitor.record_str("error_type", "EpidemiologyError");
        visitor.record_display("description", self);
    }
}
pub use crate::epidemiology::networks::NetworkModelError;

impl Diagnostic for NetworkModelError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visitor.record_str("error_type", "NetworkModelError");
        visitor.record_display("description", self);
    }
}

mod umbrella;
pub use umbrella::EpidemiologyDomainError;

#[cfg(test)]
mod tests {
    use super::*;
    use diagnostics::DiagnosticBus;
    use math_commons::error::MathError;
    use math_commons::math_kernel::types::Dimension;
    use std::error::Error as StdError;

    #[test]
    #[verified_engine::verified]
    fn test_epidemiology_sources_and_umbrella_diagnostics() {
        let sample_math = MathError::DimensionMismatch {
            expected: Dimension(1),
            actual: Dimension(2),
        };
        let epi = EpidemiologyError::from(sample_math);
        assert!(epi.source().is_some());

        let cases: Vec<EpidemiologyDomainError> = vec![
            epi.into(),
            NetworkModelError::InvalidNodeCount.into(),
            NetworkModelError::InvalidIntervention.into(),
            sample_math.into(),
            pure_math::error::StochasticError::InvalidReactionIndex(5).into(),
        ];

        let bus = DiagnosticBus::new();
        for err in &cases {
            assert!(err.source().is_some());
            assert_eq!(err.severity(), Severity::Error);
            assert_eq!(
                err.metadata().get("umbrella_type").map(String::as_str),
                Some("EpidemiologyDomainError")
            );
            bus.emit_diagnostic(err);
        }
    }
}
