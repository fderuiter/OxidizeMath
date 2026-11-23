#![cfg_attr(any(), verified_engine::verified(opt_out = "tests"))]

use super::*;
use crate::pure_math::statistics::distributions::{ContinuousDistribution, DiscreteDistribution};
use oxidize_core::rng::OxidizeRng;
use verified_engine::engine::VerifiedEngine;

#[test]
fn test_all_five_conjugate_models_intervals_and_bayes_factors() {
    // 1. Beta-Binomial
    let bb = BetaBinomial::new(2.0, 2.0).unwrap().update(18, 20).unwrap();
    assert!((bb.posterior_mean() - 20.0 / 24.0).abs() < 1e-12);
    let et = bb.credible_interval(0.95).unwrap();
    let hpd = bb.hpd_interval(0.95).unwrap();
    assert!(hpd.width() <= et.width() + 1e-6);
    let bf = BetaBinomial::uniform()
        .bayes_factor_vs_point(0.5, 19, 20)
        .unwrap();
    assert!(bf.bf10 > 10.0);
    assert!(matches!(
        bf.evidence,
        JeffreysEvidence::Strong | JeffreysEvidence::VeryStrong | JeffreysEvidence::Decisive
    ));

    // 2. Gamma-Poisson
    let counts = [4_u64, 5, 6, 3, 7, 5];
    let gp_prior = GammaPoisson::new(2.0, 1.0).unwrap();
    let gp_post = gp_prior.update_counts(&counts).unwrap();
    assert!((gp_post.posterior_mean() - 32.0 / 7.0).abs() < 1e-12);
    let pred_nb = gp_post.predictive_dist().unwrap();
    assert!((pred_nb.mean() - gp_post.posterior_mean()).abs() < 1e-10);
    let bf_gp = gp_prior.bayes_factor_vs_rate(0.5, &counts).unwrap();
    assert!(bf_gp.bf10 > 100.0);

    // 3. Normal-Normal
    let data = [2.1, 1.9, 2.3, 2.0, 1.8, 2.2];
    let nn_prior = NormalNormal::new(0.0, 4.0, 0.25).unwrap();
    let nn_post = nn_prior.update_slice(&data).unwrap();
    assert!((nn_post.posterior_mean() - 2.03).abs() < 0.05);
    let nn_bf = nn_prior.bayes_factor_vs_point(0.0, &data).unwrap();
    assert!(nn_bf.bf10 > 100.0);

    // 4. Normal-Inverse-Gamma
    let nig_prior = NormalInverseGamma::new(0.0, 1.0, 2.0, 1.0).unwrap();
    let nig_post = nig_prior.update_slice(&data).unwrap();
    let t_post = nig_post.marginal_mean_dist().unwrap();
    assert!((t_post.mean().unwrap() - nig_post.mu()).abs() < 1e-12);
    assert!(nig_post.credible_interval_mean(0.95).unwrap().contains(2.0));
    assert!(nig_prior.ln_marginal_likelihood(&data).unwrap().is_finite());

    // 5. Dirichlet-Multinomial
    let dm_prior = DirichletMultinomial::<3>::symmetric(1.0).unwrap();
    let dm_post = dm_prior.update(&[25, 10, 5]).unwrap();
    let means = dm_post.posterior_mean();
    assert!((means[0] - 26.0 / 43.0).abs() < 1e-12);
    let ci0 = dm_post.marginal_credible_interval(0, 0.90).unwrap();
    assert!(ci0.contains(means[0]));
    let dm_bf = dm_prior
        .bayes_factor_vs_probs(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], &[25, 10, 5])
        .unwrap();
    assert!(dm_bf.bf10 > 3.0);
}

#[test]
fn test_metropolis_hastings_matches_conjugate_posterior_zero_alloc() {
    let data = [3.1, 2.8, 3.4, 2.9, 3.2, 3.0, 2.7, 3.3];
    let nn = NormalNormal::new(0.0, 4.0, 0.5)
        .unwrap()
        .update_slice(&data)
        .unwrap();
    let exact_dist = nn.posterior_dist().unwrap();

    let mut chain = [0.0_f64; 2048];
    let mut scratch = [0.0_f64; 2048];
    let mut nd_chain = [[0.0_f64; 2]; 2048];
    let mut rng = OxidizeRng::from_seed(2026);

    let ((summary, emp_ci, nd_mean), _) = VerifiedEngine::run_verified(|| {
        // Also verify conjugate updates, HPD, Bayes factor, and Dirichlet inside run_verified!
        let bb = BetaBinomial::uniform().update(12, 20).unwrap();
        let _hpd = bb.hpd_interval(0.90).unwrap();
        let _bf = BetaBinomial::uniform()
            .bayes_factor_vs_point(0.5, 12, 20)
            .unwrap();
        let dm = DirichletMultinomial::<3>::symmetric(1.0)
            .unwrap()
            .update(&[4, 5, 6])
            .unwrap();
        let _simp = dm.sample_simplex(&mut rng);

        let mut mh = MetropolisHastings1D::new(0.0, 0.35).unwrap();
        let sum = mh
            .sample_into(|mu| exact_dist.ln_pdf(mu), 400, &mut chain, &mut rng)
            .unwrap();
        let ci = empirical_credible_interval(&chain, &mut scratch, 0.95).unwrap();

        let mut mh_nd = MetropolisHastingsND::<2>::new([0.0, 0.0], [0.9, 0.9]).unwrap();
        let m2 = mh_nd
            .sample_into(
                |theta| -0.5 * ((theta[0] - 1.5).powi(2) + (theta[1] + 2.0).powi(2)),
                500,
                &mut nd_chain,
                &mut rng,
            )
            .unwrap();
        (sum, ci, m2)
    });

    assert!((summary.mean - nn.posterior_mean()).abs() < 0.05);
    assert!((summary.variance - nn.posterior_variance()).abs() < 0.03);
    assert!(summary.acceptance_rate > 0.15 && summary.acceptance_rate < 0.90);
    assert!(summary.effective_sample_size > 25.0);
    let exact_ci = nn.credible_interval(0.95).unwrap();
    assert!((emp_ci.lower - exact_ci.lower).abs() < 0.08);
    assert!((emp_ci.upper - exact_ci.upper).abs() < 0.08);
    assert!((nd_mean[0] - 1.5).abs() < 0.25);
    assert!((nd_mean[1] + 2.0).abs() < 0.25);
}
