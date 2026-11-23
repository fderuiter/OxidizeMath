//! VerifiedAllocator lock tests confirming zero dynamic heap allocations across
//! multivariate Cholesky copula evaluation & sampling (`2..=8` legs), Ornstein-Uhlenbeck
//! `simulate_into`, and Hidden Markov Model `baum_welch_with_workspace` EM iterations.

use nalgebra::{DMatrix, DVector};
use oxidize_core::rng::OxidizeRng;
use pure_math::pure_math::statistics::copula::{
    Correlation, CorrelationMatrix, GaussianCopula, MAX_COPULA_DIMENSION, Probability,
    sgp_joint_probability,
};
use pure_math::pure_math::statistics::markov::hmm::{BaumWelchWorkspace, HiddenMarkovModel};
use pure_math::pure_math::statistics::ou_process::{EulerMaruyama, OuParams, TimeStep};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_copula_ou_and_hmm_baum_welch_zero_heap_allocations() {
    let rho = Correlation::new(0.35).unwrap();
    let corr6 = CorrelationMatrix::equicorrelated(6, rho).unwrap();
    let copula6 = GaussianCopula::new(corr6);
    let marginals6 = [
        Probability::new(0.65).unwrap(),
        Probability::new(0.70).unwrap(),
        Probability::new(0.55).unwrap(),
        Probability::new(0.80).unwrap(),
        Probability::new(0.60).unwrap(),
        Probability::new(0.75).unwrap(),
    ];
    let corr8 = CorrelationMatrix::equicorrelated(MAX_COPULA_DIMENSION, rho).unwrap();
    let marginals8 = [Probability::new(0.72).unwrap(); MAX_COPULA_DIMENSION];

    let ou_params = OuParams::from_values(0.5, 1.4, 0.2).unwrap();
    let ou_solver = EulerMaruyama::new(ou_params, TimeStep::new(0.01).unwrap());
    let mut ou_buf = [0.0_f64; 128];

    let observations = [0_usize, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1];
    let mut hmm = HiddenMarkovModel::new(
        DVector::from_vec(vec![0.5, 0.5]),
        DMatrix::from_row_slice(2, 2, &[0.6, 0.4, 0.4, 0.6]),
        DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.3, 0.7]),
    )
    .unwrap();
    let mut hmm_ws = BaumWelchWorkspace::new(2, 2, observations.len());
    let mut rng = OxidizeRng::new(42);
    let mut sample_u = [0.0_f64; 6];

    let ((p6, p8, report), metrics) = VerifiedEngine::run_verified(|| {
        let p6 = copula6.multivariate_cdf(&marginals6).unwrap();
        let p8 = sgp_joint_probability(&marginals8, &corr8).unwrap();
        copula6
            .sample_uniform_into(&mut rng, &mut sample_u)
            .unwrap();
        ou_solver.simulate_into(0.65, &mut ou_buf, &mut rng);
        let report = hmm
            .baum_welch_with_workspace(&observations, 12, 1e-6, &mut hmm_ws)
            .unwrap();
        (p6, p8, report)
    });

    assert!((0.0..=1.0).contains(&p6.value()));
    assert!((0.0..=1.0).contains(&p8.value()));
    assert!(sample_u.iter().all(|u| (0.0..=1.0).contains(u)));
    assert!(ou_buf.iter().all(|v| v.is_finite()));
    assert!(report.final_log_likelihood >= report.initial_log_likelihood - 1e-9);
    assert!(metrics.function_calls >= 4);
}
