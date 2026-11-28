//! Integration tests for clinical respiratory beam gating state machine and optical WAH.
use approx::assert_relative_eq;
use domain_physics::physics::medical::{
    optical_motion::processing::weighted_average_height as optical_weighted_average_height,
    radar_gating::gating::{BeamState, GatingError, GatingLogic},
};
use nalgebra::Point3;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_optical_weighted_average_height_zero_allocation() {
    let vertices = [
        Point3::new(-1.0, -1.0, 1.2),
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(1.0, 1.0, 1.4),
        Point3::new(2.0, -1.0, 1.6),
    ];

    let ((wah_optical, wah_empty), metrics) = VerifiedEngine::run_verified(|| {
        (
            optical_weighted_average_height(&vertices),
            optical_weighted_average_height(&[]),
        )
    });

    assert!(metrics.function_calls >= 2);
    assert_eq!(wah_empty, 0.0);
    assert_relative_eq!(
        wah_optical,
        1.3,
        epsilon = math_commons::registry::TOLERANCE_FAST
    );
}

#[test]
#[verified_engine::verified]
fn test_gating_logic() {
    let mut gating = GatingLogic::new(10.0, 1.0, 0.0);
    gating.arm().expect("Failed to arm gating logic");

    assert_eq!(gating.evaluate(8.0, 0.0), BeamState::BeamActive);
    assert!(gating.is_beam_active());

    assert_eq!(gating.evaluate(10.5, 0.0), BeamState::BeamActive);
    assert!(gating.is_beam_active());

    assert_eq!(gating.evaluate(11.5, 0.0), BeamState::BeamHold);
    assert!(!gating.is_beam_active());

    assert_eq!(gating.evaluate(9.5, 0.0), BeamState::BeamHold);
    assert!(!gating.is_beam_active());

    assert_eq!(gating.evaluate(8.5, 0.0), BeamState::BeamActive);
    assert!(gating.is_beam_active());
}

#[test]
#[verified_engine::verified]
fn test_gating_lifecycle_and_unarmed_evaluation() {
    let mut gating = GatingLogic::new(10.0, 1.0, 0.0);

    assert_eq!(gating.state(), BeamState::Standby);
    assert!(!gating.is_beam_active());

    let state = gating.evaluate(8.0, 0.0);
    assert_eq!(state, BeamState::Standby);
    assert!(!gating.is_beam_active());

    let armed_state = gating.arm().expect("Arming should succeed from Standby");
    assert_eq!(armed_state, BeamState::Armed);
    assert_eq!(gating.state(), BeamState::Armed);
    assert!(!gating.is_beam_active());

    assert_eq!(gating.arm().unwrap(), BeamState::Armed);

    assert_eq!(gating.evaluate(8.0, 0.0), BeamState::BeamActive);
    assert!(gating.is_beam_active());

    let disarmed = gating.disarm().expect("Disarm should succeed");
    assert_eq!(disarmed, BeamState::Standby);
    assert_eq!(gating.state(), BeamState::Standby);
    assert!(!gating.is_beam_active());

    assert_eq!(gating.disarm().unwrap(), BeamState::Standby);
}

#[test]
#[verified_engine::verified]
fn test_gating_ieee754_nan_inf_guards() {
    let mut gating = GatingLogic::new(10.0, 1.0, 0.0);
    gating.arm().expect("Arming should succeed");

    assert_eq!(gating.evaluate(8.0, 0.0), BeamState::BeamActive);
    assert!(gating.is_beam_active());

    let tripped = gating.evaluate(f64::NAN, 0.0);
    assert_eq!(tripped, BeamState::SafetyInterlock);
    assert_eq!(gating.state(), BeamState::SafetyInterlock);
    assert!(!gating.is_beam_active());

    assert_eq!(gating.evaluate(8.0, 0.0), BeamState::SafetyInterlock);
    assert!(!gating.is_beam_active());

    assert_eq!(gating.arm(), Err(GatingError::InterlockLatched));
    assert_eq!(gating.disarm(), Err(GatingError::InterlockLatched));

    assert_eq!(gating.reset_interlock().unwrap(), BeamState::Standby);
    assert_eq!(gating.state(), BeamState::Standby);

    gating.arm().unwrap();
    assert_eq!(
        gating.evaluate(8.0, f64::INFINITY),
        BeamState::SafetyInterlock
    );
    assert!(!gating.is_beam_active());

    gating.reset_interlock().unwrap();
    gating.arm().unwrap();
    assert_eq!(
        gating.evaluate(8.0, f64::NEG_INFINITY),
        BeamState::SafetyInterlock
    );
    assert!(!gating.is_beam_active());
}

#[test]
#[verified_engine::verified]
fn test_gating_invalid_transitions() {
    let mut gating = GatingLogic::new(10.0, 1.0, 0.0);

    assert_eq!(gating.reset_interlock(), Err(GatingError::NotInterlocked));

    gating.arm().unwrap();
    gating.evaluate(8.0, 0.0);

    assert_eq!(
        gating.arm(),
        Err(GatingError::InvalidTransition {
            from: BeamState::BeamActive,
            to: BeamState::Armed,
        })
    );

    assert_eq!(gating.trigger_interlock(), BeamState::SafetyInterlock);
    assert_eq!(gating.state(), BeamState::SafetyInterlock);
    assert!(!gating.is_beam_active());

    assert_eq!(gating.reset_interlock().unwrap(), BeamState::Standby);
}
