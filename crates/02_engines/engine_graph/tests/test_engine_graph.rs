//! Exact tree/chain marginals, MAP decoding, chain forward-backward, Potts MRF, and
//! zero-allocation `VerifiedAllocator` verification tests for `engine_graph`.

use approx::assert_relative_eq;
use engine_graph::{
    BpConfig, ChainFbWorkspace, ChainModelView, FactorGraphView, FactorSpec,
    chain_forward_backward_into, evaluate_assignment_log_score, max_product_bp_into,
    potts_neighborhood_posteriors_into, sum_product_bp_into,
};
use std::sync::Mutex;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

static TEST_LOCK: Mutex<()> = Mutex::new(());

struct TreeOracle {
    log_z: f64,
    m0: [f64; 2],
    m1: [f64; 3],
    m2: [f64; 2],
    m3: [f64; 2],
    best_log: f64,
    best_states: [usize; 4],
}

fn compute_tree_oracle(fg: &FactorGraphView<'_>) -> TreeOracle {
    let mut exact_z = 0.0;
    let mut m0 = [0.0; 2];
    let mut m1 = [0.0; 3];
    let mut m2 = [0.0; 2];
    let mut m3 = [0.0; 2];
    let mut best_log = f64::NEG_INFINITY;
    let mut best_states = [0usize; 4];

    for flat in 0..24usize {
        let st = [flat / 12, (flat / 4) % 3, (flat / 2) % 2, flat % 2];
        let ls = evaluate_assignment_log_score(fg, &st);
        let w = ls.exp();
        exact_z += w;
        m0[st[0]] += w;
        m1[st[1]] += w;
        m2[st[2]] += w;
        m3[st[3]] += w;
        if ls > best_log {
            best_log = ls;
            best_states = st;
        }
    }
    for slot in &mut m0 {
        *slot /= exact_z;
    }
    for slot in &mut m1 {
        *slot /= exact_z;
    }
    for slot in &mut m2 {
        *slot /= exact_z;
    }
    for slot in &mut m3 {
        *slot /= exact_z;
    }
    TreeOracle {
        log_z: exact_z.ln(),
        m0,
        m1,
        m2,
        m3,
        best_log,
        best_states,
    }
}

#[test]
fn test_sum_product_and_max_product_match_exact_brute_force_on_tree_zero_alloc() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let cards = [2usize, 3, 2, 2];
    let factors = [
        FactorSpec::unary(0, 0, 2),
        FactorSpec::unary(1, 2, 3),
        FactorSpec::pairwise(0, 1, 5, 2, 3),
        FactorSpec::pairwise(1, 2, 11, 3, 2),
        FactorSpec::pairwise(1, 3, 17, 3, 2),
    ];
    let potentials = [
        0.65, 0.35, 0.2, 0.5, 0.3, 0.9, 0.4, 0.1, 0.2, 0.7, 0.8, 0.8, 0.2, 0.3, 0.7, 0.6, 0.4, 0.5,
        0.5, 0.9, 0.1, 0.25, 0.75,
    ];
    let fg = FactorGraphView::new(&cards, &factors, &potentials).unwrap();
    let oracle = compute_tree_oracle(&fg);

    let mut beliefs = [0.0_f64; 9];
    let mut max_marginals = [0.0_f64; 9];
    let mut map_states = [0usize; 4];
    let mut scratch = [0.0_f64; 128];

    let ((sp, mp), _) = VerifiedEngine::run_verified(|| {
        let sp =
            sum_product_bp_into(&fg, &BpConfig::default(), &mut beliefs, &mut scratch).unwrap();
        let mp = max_product_bp_into(
            &fg,
            &BpConfig::default(),
            &mut max_marginals,
            &mut map_states,
            &mut scratch,
        )
        .unwrap();
        (sp, mp)
    });

    assert!(sp.converged);
    assert_relative_eq!(sp.log_partition, oracle.log_z, max_relative = 1e-12);
    for (b, e) in fg.var_belief(&beliefs, 0).iter().zip(oracle.m0.iter()) {
        assert_relative_eq!(b, e, epsilon = 1e-12);
    }
    for (b, e) in fg.var_belief(&beliefs, 1).iter().zip(oracle.m1.iter()) {
        assert_relative_eq!(b, e, epsilon = 1e-12);
    }
    for (b, e) in fg.var_belief(&beliefs, 2).iter().zip(oracle.m2.iter()) {
        assert_relative_eq!(b, e, epsilon = 1e-12);
    }
    for (b, e) in fg.var_belief(&beliefs, 3).iter().zip(oracle.m3.iter()) {
        assert_relative_eq!(b, e, epsilon = 1e-12);
    }
    assert!(mp.converged);
    assert_eq!(map_states, oracle.best_states);
    assert_relative_eq!(mp.log_map_score, oracle.best_log, max_relative = 1e-12);
}

#[test]
fn test_chain_forward_backward_and_potts_zero_alloc() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let initial = [0.6, 0.4];
    let transitions = [0.75, 0.25, 0.2, 0.8];
    let emissions = [0.8, 0.2, 0.1, 0.9];
    let obs = [0usize, 1, 1, 0];

    let mut alpha = [0.0_f64; 8];
    let mut beta = [0.0_f64; 8];
    let mut scales = [0.0_f64; 4];
    let mut gamma = [0.0_f64; 8];
    let mut potts_scratch = [0.0_f64; 3];
    let mut potts_post = [0.0_f64; 3];

    let model = ChainModelView {
        num_states: 2,
        num_obs: 2,
        initial: &initial,
        transitions: &transitions,
        emissions: &emissions,
    };

    let (ll, _) = VerifiedEngine::run_verified(|| {
        let mut ws = ChainFbWorkspace {
            alpha: &mut alpha,
            beta: &mut beta,
            scales: &mut scales,
            gamma: &mut gamma,
        };
        let ll = chain_forward_backward_into(&model, &obs, &mut ws).unwrap();
        potts_neighborhood_posteriors_into(
            &[0.3, 0.5, 0.2],
            &[1.0 / 3.0; 3],
            &[1, 5, 0],
            0.8,
            &mut potts_scratch,
            &mut potts_post,
        )
        .unwrap();
        ll
    });

    assert!(ll.is_finite() && ll < 0.0);
    for t in 0..4 {
        assert_relative_eq!(gamma[2 * t] + gamma[2 * t + 1], 1.0, epsilon = 1e-14);
    }
    assert_relative_eq!(potts_post.iter().sum::<f64>(), 1.0, epsilon = 1e-14);
    assert!(potts_post[1] > potts_post[0] && potts_post[1] > potts_post[2]);
}
