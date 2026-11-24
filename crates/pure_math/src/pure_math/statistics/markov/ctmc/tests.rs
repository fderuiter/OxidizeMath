use super::*;
use approx::assert_relative_eq;
use nalgebra::DMatrix;

#[test]
#[verified_engine::verified]
fn test_two_state_birth_death() {
    // State 0 → State 1 at rate 2.0
    // State 1 → State 0 at rate 3.0
    let generator = DMatrix::from_row_slice(2, 2, &[-2.0, 2.0, 3.0, -3.0]);

    let chain = ContinuousMarkovChain::new(generator).unwrap();

    assert_eq!(chain.num_states(), 2);

    // Steady-state should be π = [3/5, 2/5]
    let pi = chain.steady_state().unwrap();
    assert_relative_eq!(pi[0], 0.6, epsilon = 1e-4);
    assert_relative_eq!(pi[1], 0.4, epsilon = 1e-4);
}

#[test]
#[verified_engine::verified]
fn test_transition_probabilities() {
    let generator = DMatrix::from_row_slice(2, 2, &[-1.0, 1.0, 1.0, -1.0]);

    let chain = ContinuousMarkovChain::new(generator).unwrap();

    // At t=0, P(0) should be identity
    let p_0 = chain.transition_probabilities(0.0).unwrap();
    assert_relative_eq!(
        p_0[(0, 0)],
        1.0,
        epsilon = math_commons::registry::TOLERANCE_HIGH
    );
    assert_relative_eq!(
        p_0[(0, 1)],
        0.0,
        epsilon = math_commons::registry::TOLERANCE_HIGH
    );
    assert_relative_eq!(
        p_0[(1, 0)],
        0.0,
        epsilon = math_commons::registry::TOLERANCE_HIGH
    );
    assert_relative_eq!(
        p_0[(1, 1)],
        1.0,
        epsilon = math_commons::registry::TOLERANCE_HIGH
    );

    // At t > 0, check stochasticity
    let p_t = chain.transition_probabilities(1.0).unwrap();
    for i in 0..2 {
        let row_sum: f64 = p_t.row(i).iter().sum();
        assert_relative_eq!(
            row_sum,
            1.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }

    // For long time, should approach steady state
    let p_large = chain.transition_probabilities(10.0).unwrap();
    let pi = chain.steady_state().unwrap();

    // Each row should be approximately π
    for i in 0..2 {
        assert_relative_eq!(p_large[(i, 0)], pi[0], epsilon = 1e-3);
        assert_relative_eq!(p_large[(i, 1)], pi[1], epsilon = 1e-3);
    }
}

#[test]
#[verified_engine::verified]
fn test_simulation() {
    let generator = DMatrix::from_row_slice(2, 2, &[-1.0, 1.0, 2.0, -2.0]);

    let chain = ContinuousMarkovChain::new(generator).unwrap();
    let mut rng = oxidize_core::rng::OxidizeRng::default();

    let trajectory = chain.simulate_trajectory(0, 10.0, &mut rng).unwrap();

    // Check trajectory properties
    assert!(!trajectory.is_empty());
    assert_eq!(trajectory[0], (0.0, 0)); // Starts at state 0

    // Times should be increasing
    for i in 1..trajectory.len() {
        assert!(trajectory[i].0 > trajectory[i - 1].0);
    }

    // All states should be valid
    for (_, state) in &trajectory {
        assert!(*state < 2);
    }
}

#[test]
#[verified_engine::verified]
fn test_validation_errors() {
    // Non-square matrix
    let g = DMatrix::from_row_slice(2, 3, &[-1.0, 1.0, 0.0, 1.0, -1.0, 0.0]);
    assert!(ContinuousMarkovChain::new(g).is_err());

    // Rows don't sum to 0
    let g = DMatrix::from_row_slice(2, 2, &[-1.0, 1.5, 1.0, -1.0]);
    assert!(ContinuousMarkovChain::new(g).is_err());

    // Positive diagonal
    let g = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, -2.0]);
    assert!(ContinuousMarkovChain::new(g).is_err());

    // Negative off-diagonal
    let g = DMatrix::from_row_slice(2, 2, &[-1.0, -1.0, 2.0, -2.0]);
    assert!(ContinuousMarkovChain::new(g).is_err());
}

#[test]
#[verified_engine::verified]
fn test_absorption_times() {
    // States 0, 1 are transient; state 2 is absorbing
    let generator = DMatrix::from_row_slice(
        3,
        3,
        &[
            -2.0, 1.0, 1.0, // State 0 → 1 or 2
            1.0, -2.0, 1.0, // State 1 → 0 or 2
            0.0, 0.0, 0.0, // State 2 (absorbing)
        ],
    );

    let chain = ContinuousMarkovChain::new(generator).unwrap();
    let transient = vec![0, 1];

    let times = chain.expected_absorption_times(&transient).unwrap();

    // Both transient states should have positive expected absorption times
    assert!(times[0] > 0.0);
    assert!(times[1] > 0.0);

    // By symmetry, they should be equal
    assert_relative_eq!(
        times[0],
        times[1],
        epsilon = math_commons::registry::TOLERANCE_HIGH
    );
}

#[test]
#[verified_engine::verified]
fn test_deterministic_simulation() {
    let generator = DMatrix::from_row_slice(2, 2, &[-1.0, 1.0, 1.0, -1.0]);
    let chain = ContinuousMarkovChain::new(generator).unwrap();

    let mut rng1 = oxidize_core::rng::OxidizeRng::new(12345);
    let traj1 = chain.simulate_trajectory(0, 5.0, &mut rng1).unwrap();

    let mut rng2 = oxidize_core::rng::OxidizeRng::new(12345);
    let traj2 = chain.simulate_trajectory(0, 5.0, &mut rng2).unwrap();

    // Same seed should produce same trajectory
    assert_eq!(traj1.len(), traj2.len());
    for i in 0..traj1.len() {
        assert_eq!(traj1[i], traj2[i]);
    }
}

#[test]
#[verified_engine::verified]
fn test_slow_rate_asymmetric_generators_steady_state() {
    // 1e-4 asymmetric generator: rate 0->1 is 1e-4, rate 1->0 is 2e-4 => pi = [2/3, 1/3]
    let g_slow_4 = DMatrix::from_row_slice(2, 2, &[-1e-4, 1e-4, 2e-4, -2e-4]);
    let chain_4 = ContinuousMarkovChain::new(g_slow_4).unwrap();
    let pi_4 = chain_4.steady_state().unwrap();
    assert_relative_eq!(pi_4[0], 2.0 / 3.0, epsilon = 1e-12);
    assert_relative_eq!(pi_4[1], 1.0 / 3.0, epsilon = 1e-12);

    // 1e-7 asymmetric generator: rate 0->1 is 1e-7, rate 1->0 is 3e-7 => pi = [0.75, 0.25]
    let g_slow_7 = DMatrix::from_row_slice(2, 2, &[-1e-7, 1e-7, 3e-7, -3e-7]);
    let chain_7 = ContinuousMarkovChain::new(g_slow_7).unwrap();
    let pi_7 = chain_7.steady_state().unwrap();
    assert_relative_eq!(pi_7[0], 0.75, epsilon = 1e-12);
    assert_relative_eq!(pi_7[1], 0.25, epsilon = 1e-12);
}

#[test]
#[verified_engine::verified]
fn test_multi_closed_class_steady_state_returns_none() {
    // State 0 transient draining to absorbing states 1 and 2 -> 2 closed classes => None
    let g = DMatrix::from_row_slice(3, 3, &[-3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let chain = ContinuousMarkovChain::new(g).unwrap();
    assert_eq!(chain.closed_classes().len(), 2);
    assert!(chain.steady_state().is_none());
}

#[test]
fn test_ctmc_zero_allocation_gillespie_verified() {
    let g = DMatrix::from_row_slice(2, 2, &[-1.0, 1.0, 2.0, -2.0]);
    let chain = ContinuousMarkovChain::new(g).unwrap();
    let mut times = [0.0_f64; 256];
    let mut states = [0_usize; 256];
    let mut sojourns = [0.0_f64; 2];
    let mut rng = oxidize_core::rng::OxidizeRng::new(7);

    let (events, metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        chain
            .simulate_gillespie_into(0, 200.0, &mut times, &mut states, &mut sojourns, &mut rng)
            .unwrap()
    });
    assert!(metrics.function_calls > 0);
    assert!(events >= 1);
    assert_relative_eq!(sojourns[0] + sojourns[1], 200.0, epsilon = 1e-10);
}

#[cfg(feature = "serde")]
#[test]
fn test_ctmc_serde_roundtrip_and_invalid_generator_rejection() {
    use serde::Deserialize;
    use serde::de::value::{Error as ValErr, MapDeserializer, SeqDeserializer};

    let generator = DMatrix::from_row_slice(2, 2, &[-2.0, 2.0, 3.0, -3.0]);
    let chain = ContinuousMarkovChain::new(generator).unwrap();
    let json = serde_json::to_string(&chain).unwrap();
    let decoded: ContinuousMarkovChain<f64> = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.num_states(), 2);
    assert_relative_eq!(decoded.generator()[(0, 1)], 2.0, epsilon = 1e-12);

    // Reject row not summing to 0 and negative off-diagonal rates via MarkovError::InvalidGenerator
    let bad_sum = r#"{"generator":[[-1.0,2.0],[3.0,-3.0]]}"#;
    let err = serde_json::from_str::<ContinuousMarkovChain<f64>>(bad_sum).unwrap_err();
    assert!(err.to_string().contains("Invalid generator matrix"));

    let bad_off_diag = r#"{"generator":[[1.0,-1.0],[3.0,-3.0]]}"#;
    let err2 = serde_json::from_str::<ContinuousMarkovChain<f64>>(bad_off_diag).unwrap_err();
    assert!(err2.to_string().contains("Invalid generator matrix"));

    // Reject NaN generator entries via MarkovError::InvalidGenerator
    let rows = vec![vec![f64::NAN, 0.0], vec![0.0, 0.0]];
    let de: MapDeserializer<'_, _, ValErr> = MapDeserializer::new(std::iter::once((
        "generator",
        SeqDeserializer::new(rows.into_iter()),
    )));
    let nan_err = ContinuousMarkovChain::<f64>::deserialize(de).unwrap_err();
    assert!(nan_err.to_string().contains("Invalid generator matrix"));
}

#[test]
fn test_slow_ctmc_steady_state_multi_sink_rejection_and_zero_alloc() {
    use math_commons::stochastic::SolverWorkspace;
    use verified_engine::engine::VerifiedEngine;

    let slow_gen = DMatrix::from_row_slice(2, 2, &[-1e-7, 1e-7, 3e-7, -3e-7]);
    let slow_chain = ContinuousMarkovChain::new(slow_gen.clone()).unwrap();
    let pi = slow_chain
        .steady_state()
        .expect("slow irreducible 2-state CTMC must have unique steady state");
    assert_relative_eq!(pi[0], 0.75, epsilon = 1e-12);
    assert_relative_eq!(pi[1], 0.25, epsilon = 1e-12);

    let scale_rel_res = (slow_gen.transpose() * &pi).norm() / 3e-7;
    assert!(
        scale_rel_res < 1e-12,
        "expected scale-relative residual < 1e-12, got {scale_rel_res:e}"
    );

    let mut ws = SolverWorkspace::new(2, 0);
    let mut out = DVector::zeros(2);
    VerifiedEngine::run_verified(|| {
        assert!(slow_chain.steady_state_with_workspace(&mut ws, &mut out));
    });
    assert_relative_eq!(out[0], 0.75, epsilon = 1e-12);
    assert_relative_eq!(out[1], 0.25, epsilon = 1e-12);

    // Multi-absorbing CTMC: states 0 and 2 absorbing, state 1 splits between them -> None.
    let multi_abs = DMatrix::from_row_slice(
        3,
        3,
        &[
            0.0, 0.0, 0.0, //
            1.5, -3.0, 1.5, //
            0.0, 0.0, 0.0, //
        ],
    );
    let multi_chain = ContinuousMarkovChain::new(multi_abs).unwrap();
    assert!(multi_chain.steady_state().is_none());
}
