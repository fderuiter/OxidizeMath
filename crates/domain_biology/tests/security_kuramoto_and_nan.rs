//! Tier 5 security tests for `domain_biology`: `KuramotoNetwork::try_new`
//! `checked_mul` overflow, zero-oscillator guard, `NaN`/infinity coupling rejection,
//! subnormal coupling stability, and zero-allocation RK4 integration.

use domain_biology::biology::neuroscience::KuramotoNetwork;
use math_commons::error::MathError;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static TEST_ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn kuramoto_try_new_rejects_zero_nodes_overflow_and_non_finite_coupling() {
    assert!(matches!(
        KuramotoNetwork::try_new(0, 1.0),
        Err(MathError::StrictlyOutOfBounds { .. })
    ));
    assert!(matches!(
        KuramotoNetwork::try_new((usize::MAX / 2) + 1, 1.0),
        Err(MathError::CapacityOverflow { .. })
    ));
    assert!(matches!(
        KuramotoNetwork::try_new(8, -0.5),
        Err(MathError::OutOfBounds { .. })
    ));
    assert!(matches!(
        KuramotoNetwork::try_new(8, f64::NAN),
        Err(MathError::OutOfBounds { .. })
    ));
    assert!(matches!(
        KuramotoNetwork::try_new(8, f64::INFINITY),
        Err(MathError::OutOfBounds { .. })
    ));
    assert!(matches!(
        KuramotoNetwork::try_new(8, f64::NEG_INFINITY),
        Err(MathError::OutOfBounds { .. })
    ));
}

#[test]
fn kuramoto_subnormal_coupling_and_step_execute_with_zero_allocations() {
    let subnormal = f64::MIN_POSITIVE * 0.5;
    let mut net = KuramotoNetwork::try_new(8, subnormal).expect("valid subnormal coupling");

    let ((r, psi), metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..16 {
            net.step(0.02);
        }
        // Non-finite or non-positive dt must safely no-op without poisoning phases.
        net.step(f64::NAN);
        net.step(f64::INFINITY);
        net.step(-0.1);
        net.order_parameter()
    });

    assert_eq!(metrics.allocations, 0);
    assert_eq!(metrics.allocated_bytes, 0);
    assert!(r.is_finite() && (0.0..=1.0 + 1e-12).contains(&r));
    assert!(psi.is_finite());
    assert!(net.phases().iter().all(|theta| theta.is_finite()));
}
