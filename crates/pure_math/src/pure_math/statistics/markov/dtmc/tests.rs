use super::*;
use approx::assert_relative_eq;
use nalgebra::{DMatrix, DVector};

#[test]
#[verified_engine::verified]
fn test_simple_absorbing_chain() {
    // Two transient states, one absorbing
    let p = DMatrix::from_row_slice(
        3,
        3,
        &[
            0.5, 0.3, 0.2, // State 0 (transient)
            0.2, 0.6, 0.2, // State 1 (transient)
            0.0, 0.0, 1.0, // State 2 (absorbing)
        ],
    );

    let states = vec![
        StateType::Transient,
        StateType::Transient,
        StateType::Absorbing,
    ];

    let chain = MarkovChain::new(p, states).unwrap();

    assert_eq!(chain.num_transient(), 2);
    assert_eq!(chain.num_absorbing(), 1);

    // Test Q matrix
    let q = chain.q_matrix();
    assert_eq!(q.nrows(), 2);
    assert_eq!(q.ncols(), 2);
    assert_relative_eq!(q[(0, 0)], 0.5);
    assert_relative_eq!(q[(0, 1)], 0.3);
    assert_relative_eq!(q[(1, 0)], 0.2);
    assert_relative_eq!(q[(1, 1)], 0.6);

    // Test R matrix
    let r = chain.r_matrix();
    assert_eq!(r.nrows(), 2);
    assert_eq!(r.ncols(), 1);
    assert_relative_eq!(r[(0, 0)], 0.2);
    assert_relative_eq!(r[(1, 0)], 0.2);

    // Test fundamental matrix
    let n = chain.fundamental_matrix().unwrap();
    assert_eq!(n.nrows(), 2);
    assert_eq!(n.ncols(), 2);

    // Verify (I - Q) * N = I
    let q = chain.q_matrix();
    let i_minus_q = DMatrix::identity(2, 2) - q;
    let product = i_minus_q * n;
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(
                product[(i, j)],
                expected,
                epsilon = math_commons::registry::TOLERANCE_HIGH
            );
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_absorption_probabilities() {
    // Classic gambler's ruin: states 0 and 4 are absorbing
    let p = DMatrix::from_row_slice(
        5,
        5,
        &[
            1.0, 0.0, 0.0, 0.0, 0.0, // State 0 (absorbing - ruin)
            0.5, 0.0, 0.5, 0.0, 0.0, // State 1
            0.0, 0.5, 0.0, 0.5, 0.0, // State 2
            0.0, 0.0, 0.5, 0.0, 0.5, // State 3
            0.0, 0.0, 0.0, 0.0, 1.0, // State 4 (absorbing - win)
        ],
    );

    let states = vec![
        StateType::Absorbing, // 0
        StateType::Transient, // 1
        StateType::Transient, // 2
        StateType::Transient, // 3
        StateType::Absorbing, // 4
    ];

    let chain = MarkovChain::new(p, states).unwrap();

    let absorption = chain.absorption_probabilities().unwrap();

    // absorption[i,0] = probability of reaching state 0 from transient state i
    // absorption[i,1] = probability of reaching state 4 from transient state i

    // For symmetric random walk, probability of reaching 0 from state i is (4-i)/4
    // and probability of reaching 4 is i/4

    // Transient state 1 (index 0 in absorption matrix)
    assert_relative_eq!(
        absorption[(0, 0)],
        0.75,
        epsilon = math_commons::registry::TOLERANCE_HIGH
    ); // Reach 0
    assert_relative_eq!(
        absorption[(0, 1)],
        0.25,
        epsilon = math_commons::registry::TOLERANCE_HIGH
    ); // Reach 4

    // Transient state 2 (index 1 in absorption matrix)
    assert_relative_eq!(
        absorption[(1, 0)],
        0.5,
        epsilon = math_commons::registry::TOLERANCE_HIGH
    );
    assert_relative_eq!(
        absorption[(1, 1)],
        0.5,
        epsilon = math_commons::registry::TOLERANCE_HIGH
    );

    // Transient state 3 (index 2 in absorption matrix)
    assert_relative_eq!(
        absorption[(2, 0)],
        0.25,
        epsilon = math_commons::registry::TOLERANCE_HIGH
    );
    assert_relative_eq!(
        absorption[(2, 1)],
        0.75,
        epsilon = math_commons::registry::TOLERANCE_HIGH
    );
}

#[test]
#[verified_engine::verified]
fn test_expected_possession_value() {
    // Simple basketball example
    let p = DMatrix::from_row_slice(
        4,
        4,
        &[
            0.5, 0.3, 0.1, 0.1, // State 0: offense
            0.2, 0.4, 0.2, 0.2, // State 1: advantage
            0.0, 0.0, 1.0, 0.0, // State 2: score (absorbing, +2)
            0.0, 0.0, 0.0, 1.0, // State 3: turnover (absorbing, 0)
        ],
    );

    let states = vec![
        StateType::Transient,
        StateType::Transient,
        StateType::Absorbing, // Score
        StateType::Absorbing, // Turnover
    ];

    let chain = MarkovChain::new(p, states).unwrap();

    // Rewards: scoring gives 2 points, turnover gives 0
    let rewards = DVector::from_vec(vec![2.0, 0.0]);

    let epv = chain.expected_possession_value(&rewards).unwrap();

    // EPV should be positive for both transient states
    assert!(epv[0] > 0.0);
    assert!(epv[1] > 0.0);

    // State 1 (advantage) should have higher EPV than state 0
    assert!(epv[1] > epv[0]);
}

#[test]
#[verified_engine::verified]
fn test_n_step_transition() {
    let p = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);

    let states = vec![StateType::Transient, StateType::Transient];
    let chain = MarkovChain::new(p.clone(), states).unwrap();

    // P^0 should be identity
    let p0 = chain.n_step_transition(0);
    assert_relative_eq!(p0[(0, 0)], 1.0);
    assert_relative_eq!(p0[(0, 1)], 0.0);
    assert_relative_eq!(p0[(1, 0)], 0.0);
    assert_relative_eq!(p0[(1, 1)], 1.0);

    // P^1 should be P
    let p1 = chain.n_step_transition(1);
    assert_relative_eq!(p1[(0, 0)], 0.7);
    assert_relative_eq!(p1[(0, 1)], 0.3);

    // P^2 = P * P
    let p2 = chain.n_step_transition(2);
    let p2_direct = &p * &p;
    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(
                p2[(i, j)],
                p2_direct[(i, j)],
                epsilon = math_commons::registry::TOLERANCE_HIGH
            );
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_stationary_distribution_ergodic() {
    // Simple ergodic chain
    let p = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);

    let states = vec![StateType::Transient, StateType::Transient];
    let chain = MarkovChain::new(p, states).unwrap();

    let pi = chain.stationary_distribution().unwrap();

    // Should sum to 1
    assert_relative_eq!(
        pi.sum(),
        1.0,
        epsilon = math_commons::registry::TOLERANCE_HIGH
    );

    // Should satisfy π·P = π
    let pi_p = chain.transition_matrix().transpose() * &pi;
    for i in 0..2 {
        assert_relative_eq!(
            pi[i],
            pi_p[i],
            epsilon = math_commons::registry::TOLERANCE_HIGH
        );
    }

    // For this specific chain, the stationary distribution is [4/7, 3/7]
    assert_relative_eq!(
        pi[0],
        4.0 / 7.0,
        epsilon = math_commons::registry::TOLERANCE_HIGH
    );
    assert_relative_eq!(
        pi[1],
        3.0 / 7.0,
        epsilon = math_commons::registry::TOLERANCE_HIGH
    );
}

#[test]
#[verified_engine::verified]
fn test_validation_errors() {
    // Test non-square matrix
    let p = DMatrix::from_row_slice(2, 3, &[0.5, 0.3, 0.2, 0.3, 0.4, 0.3]);
    let states = vec![StateType::Transient, StateType::Transient];
    assert!(MarkovChain::new(p, states).is_err());

    // Test row that doesn't sum to 1
    let p = DMatrix::from_row_slice(2, 2, &[0.5, 0.3, 0.4, 0.6]);
    let states = vec![StateType::Transient, StateType::Transient];
    assert!(MarkovChain::new(p, states).is_err());

    // Test invalid absorbing state
    let p = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
    let states = vec![StateType::Absorbing, StateType::Transient];
    assert!(MarkovChain::new(p, states).is_err());
}

#[test]
#[verified_engine::verified]
fn test_period_two_walk_stationary_qr_nullspace() {
    let p = DMatrix::from_row_slice(
        3,
        3,
        &[
            0.0, 1.0, 0.0, // 0 -> 1
            0.5, 0.0, 0.5, // 1 -> 0 or 2
            0.0, 1.0, 0.0, // 2 -> 1
        ],
    );
    let states = vec![StateType::Transient; 3];
    let chain = MarkovChain::new(p, states).unwrap();

    assert!(chain.is_irreducible());
    assert!(!chain.is_aperiodic());
    assert_eq!(chain.period(), Some(2));
    assert_eq!(chain.state_period(0).unwrap(), 2);

    let pi = chain.stationary_distribution().unwrap();
    assert_relative_eq!(pi[0], 0.25, epsilon = 1e-12);
    assert_relative_eq!(pi[1], 0.5, epsilon = 1e-12);
    assert_relative_eq!(pi[2], 0.25, epsilon = 1e-12);

    // Periodic chain does not converge from a point mass initial state
    assert!(chain.limiting_distribution_from_state(0).is_none());
}

#[test]
#[verified_engine::verified]
fn test_multi_absorbing_stationary_none_and_asymmetric_limiting() {
    // State 0 transient -> absorbs into state 1 (0.95) or state 2 (0.05)
    // E.g. P[0,0]=0.5, P[0,1]=0.475, P[0,2]=0.025 => N = 2, B = [0.95, 0.05]
    let p = DMatrix::from_row_slice(
        3,
        3,
        &[
            0.5, 0.475, 0.025, // State 0 (transient)
            0.0, 1.0, 0.0, // State 1 (absorbing)
            0.0, 0.0, 1.0, // State 2 (absorbing)
        ],
    );
    let states = vec![
        StateType::Transient,
        StateType::Absorbing,
        StateType::Absorbing,
    ];
    let chain = MarkovChain::new(p, states).unwrap();

    assert_eq!(chain.closed_classes().len(), 2);
    assert!(chain.stationary_distribution().is_none());

    let e0 = DVector::from_vec(vec![1.0, 0.0, 0.0]);
    let lim = chain.limiting_distribution(&e0).unwrap();
    assert_relative_eq!(lim[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(lim[1], 0.95, epsilon = 1e-12);
    assert_relative_eq!(lim[2], 0.05, epsilon = 1e-12);

    let lim_state = chain.limiting_distribution_from_state(0).unwrap();
    assert_relative_eq!(lim_state[1], 0.95, epsilon = 1e-12);
    assert_relative_eq!(lim_state[2], 0.05, epsilon = 1e-12);
}

#[test]
#[verified_engine::verified]
fn test_tarjan_scc_and_mean_hitting_times() {
    // 3-state symmetric cycle with self-loops: P = [[0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]]
    let p = DMatrix::from_row_slice(3, 3, &[0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5]);
    let chain = MarkovChain::new(p, vec![StateType::Transient; 3]).unwrap();

    let classes = chain.communicating_classes();
    assert_eq!(classes.len(), 1);
    assert!(classes[0].is_closed);
    assert_eq!(chain.period(), Some(1));

    // Expected steps to state 2: from 2 is 0; from 1 is 2; from 0 is 4.
    let h = chain.mean_hitting_times(2).unwrap();
    assert_relative_eq!(h[0], 4.0, epsilon = 1e-12);
    assert_relative_eq!(h[1], 2.0, epsilon = 1e-12);
    assert_relative_eq!(h[2], 0.0, epsilon = 1e-12);
}

#[test]
fn test_dtmc_zero_allocation_simulation_verified() {
    let p = DMatrix::from_row_slice(2, 2, &[0.75, 0.25, 0.5, 0.5]);
    let chain = MarkovChain::new(p, vec![StateType::Transient; 2]).unwrap();
    let mut visits = [0_usize; 2];
    let mut rng = oxidize_core::rng::OxidizeRng::new(42);

    let (final_state, metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        chain
            .simulate_visits_into(0, 2_000, &mut visits, &mut rng)
            .unwrap()
    });
    assert!(final_state < 2);
    assert!(metrics.function_calls > 0);
    assert_eq!(visits[0] + visits[1], 2_000);
}
