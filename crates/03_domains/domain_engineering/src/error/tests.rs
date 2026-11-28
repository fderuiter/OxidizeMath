use super::*;
use diagnostics::{Diagnostic, DiagnosticBus, Severity};
use engine_opt::error::FitError;
use math_commons::error::MathError;
use math_commons::math_kernel::types::Dimension;
use pure_math::error::KalmanError;
use std::error::Error as StdError;

fn sample_math_err() -> MathError {
    MathError::DimensionMismatch {
        expected: Dimension(2),
        actual: Dimension(3),
    }
}

#[test]
#[verified_engine::verified]
fn test_engineering_leaf_wrappers_preserve_source_chain() {
    let battery = BatteryError::from(FitError::SingularMatrix);
    assert!(battery.source().is_some());

    let hw = HardwareError::from(sample_math_err());
    assert!(hw.source().is_some());

    let radar = RadarError::from(sample_math_err());
    assert!(radar.source().is_some());
}

#[test]
#[verified_engine::verified]
fn test_engineering_umbrella_variants_preserve_source_and_diagnostics() {
    let cases: Vec<EngineeringDomainError> = vec![
        BatteryError::NegativeCycles(-1.0).into(),
        HardwareError::InvalidParameter {
            name: "CuttingSpeed".to_string(),
            value: 0.0,
        }
        .into(),
        RadarError::from(sample_math_err()).into(),
        FitError::SingularMatrix.into(),
        sample_math_err().into(),
        KalmanError::MatrixInversionError.into(),
    ];

    let bus = DiagnosticBus::new();
    for err in &cases {
        assert!(err.source().is_some());
        assert_eq!(err.severity(), Severity::Error);
        assert_eq!(
            err.metadata().get("umbrella_type").map(String::as_str),
            Some("EngineeringDomainError")
        );
        bus.emit_diagnostic(err);
    }
}
