//! Integration tests verifying zero-allocation hot paths under [`VerifiedAllocator`]
//! and exact 10,000-step log-domain Viterbi decoding equivalence.
#![cfg_attr(any(), verified(opt_out = "tests"))]

use nalgebra::{DMatrix, DVector};
use oxidize_core::rng::OxidizeRng;
use pure_math::pure_math::statistics::markov::hmm::{
    BaumWelchConfig, BaumWelchWorkspace, HiddenMarkovModel, ViterbiWorkspace,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

fn market_regime_hmm() -> HiddenMarkovModel<f64> {
    HiddenMarkovModel::new(
        DVector::from_vec(vec![0.33, 0.33, 0.34]),
        DMatrix::from_row_slice(
            3,
            3,
            &[0.70, 0.20, 0.10, 0.20, 0.70, 0.10, 0.25, 0.25, 0.50],
        ),
        DMatrix::from_row_slice(
            3,
            5,
            &[
                0.30, 0.30, 0.20, 0.10, 0.10, // Bull
                0.10, 0.10, 0.20, 0.30, 0.30, // Bear
                0.10, 0.20, 0.40, 0.20, 0.10, // Sideways
            ],
        ),
    )
    .expect("valid market HMM")
}

fn exact_reference_log_viterbi(hmm: &HiddenMarkovModel<f64>, obs: &[usize]) -> Vec<usize> {
    let n = hmm.num_states();
    let t_max = obs.len();
    let mut log_dp = vec![vec![f64::NEG_INFINITY; n]; t_max];
    let mut ptr = vec![vec![0_usize; n]; t_max];
    for (s, slot) in log_dp[0].iter_mut().enumerate() {
        *slot = hmm.initial()[s].ln() + hmm.emissions()[(s, obs[0])].ln();
    }
    for t in 1..t_max {
        for j in 0..n {
            let mut best = f64::NEG_INFINITY;
            let mut arg = 0;
            for (i, &prev) in log_dp[t - 1].iter().enumerate() {
                let cand = prev + hmm.transitions()[(i, j)].ln();
                if cand > best {
                    best = cand;
                    arg = i;
                }
            }
            log_dp[t][j] = best + hmm.emissions()[(j, obs[t])].ln();
            ptr[t][j] = arg;
        }
    }
    let mut path = vec![0; t_max];
    let mut best_last = f64::NEG_INFINITY;
    for (i, &v) in log_dp[t_max - 1].iter().enumerate() {
        if v > best_last {
            best_last = v;
            path[t_max - 1] = i;
        }
    }
    for t in (0..t_max - 1).rev() {
        path[t] = ptr[t + 1][path[t + 1]];
    }
    path
}

#[test]
fn test_viterbi_into_10000_steps_zero_alloc_and_exact_dp_match() {
    let hmm = market_regime_hmm();
    let mut rng = OxidizeRng::new(4242);
    let (_, obs_10k) = hmm.generate(10_000, &mut rng).expect("10k sequence");
    let expected = exact_reference_log_viterbi(&hmm, &obs_10k);

    let mut vit_ws = ViterbiWorkspace::new(3, 5, obs_10k.len());
    let mut decoded = vec![0_usize; obs_10k.len()];

    let (res, _metrics) =
        VerifiedEngine::run_verified(|| hmm.viterbi_into(&obs_10k, &mut vit_ws, &mut decoded));
    res.expect("zero-allocation 10,000-step viterbi_into");

    assert_eq!(decoded, expected);
    assert!(decoded[400..].iter().any(|&s| s != 0));
}

#[test]
fn test_fit_baum_welch_into_zero_alloc_and_monotonic_likelihood() {
    let generator_hmm = market_regime_hmm();
    let mut rng = OxidizeRng::new(9001);
    let (_, obs) = generator_hmm.generate(300, &mut rng).expect("300 sequence");

    let mut model = HiddenMarkovModel::new(
        DVector::from_vec(vec![0.45, 0.30, 0.25]),
        DMatrix::from_row_slice(
            3,
            3,
            &[0.50, 0.30, 0.20, 0.30, 0.50, 0.20, 0.33, 0.33, 0.34],
        ),
        DMatrix::from_row_slice(
            3,
            5,
            &[
                0.28, 0.22, 0.20, 0.16, 0.14, // State 0 init
                0.14, 0.16, 0.20, 0.24, 0.26, // State 1 init
                0.18, 0.21, 0.24, 0.19, 0.18, // State 2 init
            ],
        ),
    )
    .expect("initial guess");

    let config = BaumWelchConfig::new(25, 1e-8, 1e-6).expect("valid config");
    let mut bw_ws = BaumWelchWorkspace::new(3, 5, obs.len());
    let mut history = [0.0_f64; 25];

    let (res, _metrics) = VerifiedEngine::run_verified(|| {
        model.fit_baum_welch_into(&obs, &config, &mut bw_ws, &mut history)
    });
    let report = res.expect("zero-allocation fit_baum_welch_into");

    assert!(report.iterations >= 1);
    assert!(report.final_log_likelihood >= report.initial_log_likelihood - 1e-12);
    let mut prev = report.initial_log_likelihood;
    for &ll in &history[..report.iterations] {
        assert!(ll >= prev - 1e-12, "LL decreased: {ll} < {prev}");
        prev = ll;
    }
}
