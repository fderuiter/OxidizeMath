//! Tier 5 security tests for `domain_epidemiology`: zero-node/population guards,
//! `NaN`/infinity rejection on compartmental & network models, subnormal rate stability,
//! and zero-allocation graph simulation inside `VerifiedEngine::run_verified`.

use domain_epidemiology::epidemiology::networks::{
    NetworkEpidemicModel, NetworkModelError, NodeState,
};
use domain_epidemiology::epidemiology::{SEIRModel, SIRModel};
use oxidize_core::rng::OxidizeRng;
use pure_math::pure_math::analysis::ode::TimeStepper;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static TEST_ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn sir_seir_and_network_builders_reject_zero_negative_nan_and_infinite_inputs() {
    assert!(SIRModel::new(0.0, 0.0, 0.3, 0.1).is_err());
    assert!(SIRModel::new(-10.0, 1.0, 0.3, 0.1).is_err());
    assert!(SIRModel::new(f64::NAN, 1.0, 0.3, 0.1).is_err());
    assert!(SIRModel::new(f64::INFINITY, 1.0, 0.3, 0.1).is_err());
    assert!(SIRModel::new(100.0, f64::NAN, 0.3, 0.1).is_err());
    assert!(SIRModel::new(100.0, 10.0, f64::NAN, 0.1).is_err());
    assert!(SIRModel::new(100.0, 10.0, 0.3, f64::INFINITY).is_err());

    assert!(SEIRModel::new(f64::NAN, 5.0, 0.4, 0.2, 0.1).is_err());
    assert!(SEIRModel::new(100.0, 5.0, f64::NEG_INFINITY, 0.2, 0.1).is_err());

    assert_eq!(
        NetworkEpidemicModel::builder().num_nodes(0).build().err(),
        Some(NetworkModelError::InvalidNodeCount)
    );
    assert_eq!(
        NetworkEpidemicModel::builder()
            .num_nodes(3)
            .parameters(f64::NAN, 0.1)
            .build()
            .err(),
        Some(NetworkModelError::InvalidParameters)
    );
    assert_eq!(
        NetworkEpidemicModel::builder()
            .num_nodes(3)
            .parameters(0.2, f64::INFINITY)
            .build()
            .err(),
        Some(NetworkModelError::InvalidParameters)
    );
    assert_eq!(
        NetworkEpidemicModel::builder()
            .num_nodes(2)
            .adjacency(vec![vec![2], vec![]])
            .build()
            .err(),
        Some(NetworkModelError::AdjacencyMismatch)
    );
}

#[test]
fn subnormal_epidemic_rates_and_network_steps_run_with_zero_allocations() {
    let subnormal = f64::MIN_POSITIVE * 0.5;
    let mut sir = SIRModel::new(1000.0, 10.0, subnormal, subnormal).expect("valid subnormal SIR");
    let mut net = NetworkEpidemicModel::builder()
        .num_nodes(4)
        .parameters(subnormal, subnormal)
        .states(vec![
            NodeState::Infected,
            NodeState::Susceptible,
            NodeState::Susceptible,
            NodeState::Vaccinated,
        ])
        .adjacency(vec![vec![1], vec![0, 2], vec![1, 3], vec![2]])
        .build()
        .expect("valid 4-node network");
    let mut rng = OxidizeRng::from_seed(99);

    assert_eq!(
        net.set_beta(f64::NAN),
        Err(NetworkModelError::InvalidParameters)
    );
    assert_eq!(
        net.set_gamma(f64::INFINITY),
        Err(NetworkModelError::InvalidParameters)
    );

    let ((s_state, r0), metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..8 {
            sir.step(0.1);
            net.step_with_rng(&mut rng);
        }
        (*sir.get_state(), net.effective_r0())
    });

    assert_eq!(metrics.allocations, 0);
    assert_eq!(metrics.allocated_bytes, 0);
    assert!(s_state.s.is_finite() && s_state.i.is_finite() && s_state.r.is_finite());
    assert!(r0.is_finite() && r0 >= 0.0);
}
