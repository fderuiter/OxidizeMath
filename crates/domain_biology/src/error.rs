use diagnostics::{Diagnostic, Severity};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
#[allow(missing_docs)]
pub enum HodgkinHuxleyError {
    #[error("Invalid gating variable value: {0} (must be between 0 and 1)")]
    #[allow(missing_docs)]
    InvalidGatingVariable(f64),
    #[error("Invalid conductance value: {0} (must be non-negative)")]
    #[allow(missing_docs)]
    InvalidConductance(f64),
}

impl Diagnostic for HodgkinHuxleyError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "HodgkinHuxleyError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

pub use crate::biology::{kinetics::KineticsError, reaction_diffusion::ReactionDiffusionError};

impl Diagnostic for KineticsError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "KineticsError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl Diagnostic for ReactionDiffusionError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "error_type".to_string(),
            "ReactionDiffusionError".to_string(),
        );
        if let Self::Math(math_err) = self {
            for (k, v) in math_err.metadata() {
                map.entry(k).or_insert(v);
            }
        }
        map.insert("description".to_string(), self.to_string());
        map
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
        let rd = ReactionDiffusionError::from(sample_math.clone());
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
