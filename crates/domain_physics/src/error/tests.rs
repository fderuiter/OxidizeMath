use super::*;
use diagnostics::{Diagnostic, DiagnosticBus, Severity};
use domain_applied::error::KalmanError;
use math_commons::error::MathError;
use math_commons::math_kernel::types::Dimension;
use std::error::Error as StdError;

fn sample_math_err() -> MathError {
    MathError::DimensionMismatch {
        expected: Dimension(2),
        actual: Dimension(3),
    }
}

#[test]
#[verified_engine::verified]
fn test_physics_leaf_wrappers_preserve_source_chain() {
    let chaos = ChaosError::from(sample_math_err());
    assert!(chaos.source().is_some());

    let radar = RadarError::from(sample_math_err());
    assert!(radar.source().is_some());

    let he = HighEnergyError::from(sample_math_err());
    assert!(he.source().is_some());

    let sm = StatMechError::from(sample_math_err());
    assert!(sm.source().is_some());
}

#[test]
#[verified_engine::verified]
fn test_physics_umbrella_variants_preserve_source_and_diagnostics() {
    let cases: Vec<PhysicsError> = vec![
        ChaosError::from(sample_math_err()).into(),
        DoseFluenceError::Singularity.into(),
        FluidError::InvalidDensity { value: -1.0 }.into(),
        GatingError::InterlockLatched.into(),
        HighEnergyError::InvalidMomentum.into(),
        NuclearError::InvalidHalfLife.into(),
        RadarError::from(sample_math_err()).into(),
        SolidStateError::InvalidParameter("band".to_string()).into(),
        StandardModelError::LandauPole { scale: 100.0 }.into(),
        StatMechError::InvalidTemperature(-5.0).into(),
        sample_math_err().into(),
        KalmanError::MatrixInversionError.into(),
    ];

    let bus = DiagnosticBus::new();
    for err in &cases {
        assert!(err.source().is_some());
        assert_eq!(err.severity(), Severity::Error);
        assert_eq!(
            err.metadata().get("umbrella_type").map(String::as_str),
            Some("PhysicsError")
        );
        bus.emit_diagnostic(err);
    }
}
