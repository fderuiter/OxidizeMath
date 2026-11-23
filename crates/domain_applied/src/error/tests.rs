use super::*;
use diagnostics::{Diagnostic, DiagnosticBus, Severity};
use math_commons::error::MathError;
use math_commons::math_kernel::types::Dimension;
use pure_math::error::AnalysisError;
use std::error::Error as StdError;

fn sample_math_err() -> MathError {
    MathError::DimensionMismatch {
        expected: Dimension(2),
        actual: Dimension(3),
    }
}

#[test]
#[verified_engine::verified]
fn test_applied_leaf_wrappers_preserve_source_chain() {
    let pk = PharmacokineticsError::from(AnalysisError::ConvergenceError(0.5));
    assert!(pk.source().is_some());

    let kalman = KalmanError::from(sample_math_err());
    assert!(kalman.source().is_some());

    let iso = IsosurfaceError::from(sample_math_err());
    assert!(iso.source().is_some());
}

#[test]
#[verified_engine::verified]
fn test_applied_umbrella_variants_preserve_source_and_diagnostics() {
    let cases: Vec<AppliedError> = vec![
        BatteryError::NegativeCycles(-1.0).into(),
        ClinicalTrialError::ZeroTotalCount.into(),
        EngineeringError::InvalidParameter {
            name: "bits".to_string(),
            value: 0.0,
        }
        .into(),
        GameTheoryError::NonSquarePayoffMatrix { rows: 2, cols: 3 }.into(),
        IsosurfaceError::from(sample_math_err()).into(),
        KalmanError::MatrixInversionError.into(),
        LoraError::EmptyEnsemble.into(),
        PharmacokineticsError::from(AnalysisError::ConvergenceError(1.0)).into(),
        VoxelGridError::InvalidDimensions.into(),
        WatertightGridError::InvalidIsoLevel(f64::NAN).into(),
        sample_math_err().into(),
        AnalysisError::InvalidParameters("unbracketed".to_string()).into(),
    ];

    let bus = DiagnosticBus::new();
    for err in &cases {
        assert!(err.source().is_some());
        assert_eq!(err.severity(), Severity::Error);
        assert_eq!(
            err.metadata().get("umbrella_type").map(String::as_str),
            Some("AppliedError")
        );
        bus.emit_diagnostic(err);
    }
}
