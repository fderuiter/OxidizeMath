//! Integration tests verifying zero-allocation hot paths under `VerifiedAllocator`
//! for `HiddenMarkovModel` (`HmmScratch`) and `BivariateCopulaScratch`.
#![cfg_attr(any(), verified(opt_out = "tests"))]

use approx::assert_relative_eq;
use nalgebra::{DMatrix, DVector};
use pure_math::pure_math::statistics::copula::BivariateCopulaScratch;
use pure_math::pure_math::statistics::markov::hmm::{HiddenMarkovModel, HmmScratch};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_hmm_and_copula_paradox_kernels_zero_allocation_hot_path() {
    let initial = DVector::from_vec(vec![0.25, 0.05, 0.70]);
    let transitions = DMatrix::from_row_slice(
        3,
        3,
        &[0.20, 0.40, 0.40, 0.60, 0.00, 0.40, 0.05, 0.95, 0.00],
    );
    let emissions = DMatrix::from_row_slice(
        3,
        3,
        &[0.50, 0.15, 0.35, 0.10, 0.85, 0.05, 0.10, 0.70, 0.20],
    );
    let hmm = HiddenMarkovModel::new(initial, transitions, emissions).expect("valid 3-state hmm");
    let mut hmm_scratch = HmmScratch::new(4, 12);
    let mut copula_scratch = BivariateCopulaScratch::new(16, 16);

    let obs = [0usize, 1, 1];
    let rhos = [-0.50, -0.20, 0.0, 0.30, 0.50, 0.70];
    let quantiles = [0.80, 0.90, 0.95, 0.99, 0.999];

    let ((hmm_eval, p_neg, p_pos, tail_cmp), _metrics) = VerifiedEngine::run_verified(|| {
        let eval = hmm
            .evaluate_with_scratch(&obs, &mut hmm_scratch)
            .expect("zero-alloc hmm evaluation");
        copula_scratch.populate_tail_regime_grid(&rhos, &quantiles, 3.0);
        let neg = copula_scratch.evaluate_gaussian_joint(0.20, 0.20, -0.50);
        let pos = copula_scratch.evaluate_gaussian_joint(0.20, 0.20, 0.70);
        let tail = copula_scratch.evaluate_tail_comparison(0.20, 0.20, 0.999, 0.50, 3.0);
        (eval, neg, pos, tail)
    });

    assert_relative_eq!(hmm_eval.likelihood, 0.069_376_5, epsilon = 1e-12);
    assert_relative_eq!(hmm_eval.viterbi_joint_prob, 0.028_262_5, epsilon = 1e-12);
    assert_relative_eq!(hmm_eval.marginal_joint_prob, 0.0, epsilon = 1e-15);
    assert_eq!(hmm_scratch.viterbi_slice(3), &[0, 2, 1]);
    assert_eq!(hmm_scratch.marginal_slice(3), &[0, 1, 1]);

    assert_relative_eq!(p_neg, 0.008_438, epsilon = 1e-5);
    assert_relative_eq!(p_pos, 0.112_902, epsilon = 1e-5);
    assert_relative_eq!(tail_cmp.tail_multiplier, 5.8254, epsilon = 1e-2);
}
