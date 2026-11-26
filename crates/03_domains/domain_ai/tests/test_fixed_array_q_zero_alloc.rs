//! Zero-allocation integration tests for `FixedArrayQFunction` and `QLearningAgent::select_action_into`.

use domain_ai::ai::reinforcement_learning::{
    DiscreteAction, DiscreteState, EpsilonGreedy, FixedArrayQFunction, QFunction, QLearningAgent,
};
use math_commons::primitives::UnitInterval;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_fixed_array_q_function_and_select_action_into_zero_alloc() {
    let q_func = FixedArrayQFunction::<16, 5>::new();
    let alpha = UnitInterval::new(0.25).expect("valid alpha");
    let gamma = UnitInterval::new(0.90).expect("valid gamma");
    let epsilon = UnitInterval::new(0.15).expect("valid epsilon");

    let mut agent = QLearningAgent::new_generic(
        q_func,
        alpha,
        gamma,
        Box::new(EpsilonGreedy::new(epsilon)),
        Some(2026),
    );

    let actions = [
        DiscreteAction(0),
        DiscreteAction(1),
        DiscreteAction(2),
        DiscreteAction(3),
        DiscreteAction(4),
    ];
    let mut q_scratch = [0.0_f64; 5];
    let mut chosen_count = 0_usize;

    VerifiedEngine::run_verified(|| {
        let mut s = DiscreteState(0);
        for step in 0..64_usize {
            if let Some(act) = agent.select_action_into(&s, &actions, &mut q_scratch) {
                chosen_count += 1;
                let target_action = 3_usize;
                let dist = act.0.abs_diff(target_action) as f64;
                let reward = 2.0 - dist;
                let next_s = DiscreteState((s.0 + act.0 + step) % 16);
                agent.update(&s, &act, reward, &next_s, &actions);
                s = next_s;
            }
        }
    });

    assert_eq!(chosen_count, 64);
    assert!(
        agent
            .q_func
            .value(&DiscreteState(0), &DiscreteAction(3))
            .is_finite(),
        "Updated Q-value must be finite"
    );
    assert_eq!(
        agent.q_func.value(&DiscreteState(99), &DiscreteAction(99)),
        0.0,
        "Out-of-bounds FixedArrayQFunction lookup must safely return 0.0"
    );
}

#[test]
fn test_select_action_into_deterministic_seed_parity() {
    let alpha = UnitInterval::new(0.2).expect("valid alpha");
    let gamma = UnitInterval::new(0.95).expect("valid gamma");
    let eps = UnitInterval::new(0.35).expect("valid epsilon");

    let mut agent_a = QLearningAgent::new_generic(
        FixedArrayQFunction::<8, 4>::new(),
        alpha,
        gamma,
        Box::new(EpsilonGreedy::new(eps)),
        Some(77),
    );
    let mut agent_b = QLearningAgent::new_generic(
        FixedArrayQFunction::<8, 4>::new(),
        alpha,
        gamma,
        Box::new(EpsilonGreedy::new(eps)),
        Some(77),
    );

    let actions = [
        DiscreteAction(0),
        DiscreteAction(1),
        DiscreteAction(2),
        DiscreteAction(3),
    ];
    let mut scratch = [0.0_f64; 4];
    let state = DiscreteState(2);

    for idx in 0..4 {
        let val = idx as f64 * 0.5;
        agent_a.q_func.update(&state, &DiscreteAction(idx), val);
        agent_b.q_func.update(&state, &DiscreteAction(idx), val);
    }

    for _ in 0..32 {
        let pick_into = agent_a.select_action_into(&state, &actions, &mut scratch);
        let pick_vec = agent_b.select_action(&state, &actions);
        assert_eq!(pick_into, pick_vec);
    }
}
