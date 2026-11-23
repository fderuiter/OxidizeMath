//! Numerical precision, quantile inversion, conjugate Bayes, conformal coverage, optimal transport,
//! and zero-allocation `VerifiedAllocator` verification tests for `engine_prob`.

use approx::assert_relative_eq;
use engine_prob::{
    Bernoulli, Beta, BetaBinomial, Binomial, Categorical, ChiSquared, ContinuousDist,
    DirichletMultinomial, DiscreteDist, EValue, Exponential, Gamma, GammaPoisson, Normal,
    NormalInverseGamma, NormalNormal, Poisson, SinkhornConfig, StudentsT, Uniform, beta,
    conformal_p_value, conformal_prediction_mask_into, conformal_quantile_into,
    discrete_posterior_into, erf, erf_inv, erfc, erfc_inv, gamma, ln_gamma,
    normalize_log_weights_into, regularized_incomplete_beta, regularized_lower_gamma,
    regularized_upper_gamma, sinkhorn_log_domain_into, sinkhorn_scratch_len, wasserstein_1d_into,
    wasserstein_1d_weights,
};
use oxidize_core::rng::OxidizeRng;
use std::sync::Mutex;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

static TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn test_special_functions_accuracy() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    assert_relative_eq!(gamma(1.0), 1.0, max_relative = 1e-14);
    assert_relative_eq!(gamma(5.0), 24.0, max_relative = 1e-14);
    assert_relative_eq!(
        gamma(0.5),
        core::f64::consts::PI.sqrt(),
        max_relative = 1e-14
    );
    assert_relative_eq!(ln_gamma(10.0), 362_880.0_f64.ln(), max_relative = 1e-14);
    assert_relative_eq!(beta(2.0, 3.0), 1.0 / 12.0, max_relative = 1e-13);

    for x in [0.15, 0.8, 2.5, 9.0] {
        let p = regularized_lower_gamma(3.5, x).unwrap();
        let q = regularized_upper_gamma(3.5, x).unwrap();
        assert_relative_eq!(p + q, 1.0, max_relative = 1e-13);
    }

    assert_relative_eq!(
        regularized_incomplete_beta(2.0, 5.0, 0.3).unwrap(),
        1.0 - regularized_incomplete_beta(5.0, 2.0, 0.7).unwrap(),
        max_relative = 1e-13
    );

    for x in [-2.7, -0.9, 0.0, 0.45, 1.8, 3.2] {
        let y = erfc(x);
        assert_relative_eq!(erf(x) + y, 1.0, max_relative = 1e-14);
        assert_relative_eq!(erfc_inv(y), x, max_relative = 1e-13, epsilon = 1e-14);
        let e = erf(x);
        assert_relative_eq!(erf(erf_inv(e)), e, max_relative = 1e-14, epsilon = 1e-15);
        assert_relative_eq!(erf_inv(e), x, max_relative = 1e-11, epsilon = 1e-14);
    }
}

#[test]
fn test_continuous_distributions_quantiles_within_1e10_and_moments() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let probs = [1e-6, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0 - 1e-6];
    let normal = Normal::new(1.5, 2.2).unwrap();
    let uniform = Uniform::new(-3.0, 5.0).unwrap();
    let exp_d = Exponential::new(1.7).unwrap();
    let gamma_d = Gamma::new(3.4, 1.2).unwrap();
    let beta_d = Beta::new(2.5, 4.2).unwrap();
    let t_d = StudentsT::new(7.0).unwrap();
    let chi2_d = ChiSquared::new(6.0).unwrap();

    for &p in &probs {
        for q_round in [
            normal.cdf(normal.quantile(p).unwrap()),
            uniform.cdf(uniform.quantile(p).unwrap()),
            exp_d.cdf(exp_d.quantile(p).unwrap()),
            gamma_d.cdf(gamma_d.quantile(p).unwrap()),
            beta_d.cdf(beta_d.quantile(p).unwrap()),
            t_d.cdf(t_d.quantile(p).unwrap()),
            chi2_d.cdf(chi2_d.quantile(p).unwrap()),
        ] {
            assert_relative_eq!(q_round, p, max_relative = 1e-11, epsilon = 1e-12);
        }
    }

    assert_relative_eq!(
        Normal::standard().quantile(0.975).unwrap(),
        1.959_963_984_540_054,
        max_relative = 1e-12
    );
    assert_relative_eq!(
        StudentsT::new(10.0).unwrap().quantile(0.975).unwrap(),
        2.228_138_851_964_938_5,
        max_relative = 1e-11
    );
    assert_relative_eq!(
        ChiSquared::new(4.0).unwrap().quantile(0.95).unwrap(),
        9.487_729_036_781_154,
        max_relative = 1e-11
    );
}

#[test]
fn test_discrete_distributions_and_seeded_determinism() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let bern = Bernoulli::new(0.35).unwrap();
    let binom = Binomial::new(20, 0.4).unwrap();
    let pois = Poisson::new(4.5).unwrap();
    let weights = [0.1, 0.25, 0.45, 0.2];
    let cat = Categorical::new(&weights).unwrap();

    assert_relative_eq!(binom.mean(), 8.0, max_relative = 1e-14);
    assert_relative_eq!(binom.variance(), 4.8, max_relative = 1e-14);
    assert_eq!(binom.quantile(binom.cdf(8)).unwrap(), 8);
    assert_eq!(pois.quantile(pois.cdf(5)).unwrap(), 5);
    assert_eq!(bern.quantile(0.2).unwrap(), 0);
    assert_eq!(bern.quantile(0.8).unwrap(), 1);
    assert_eq!(cat.quantile(0.5).unwrap(), 2);

    let mut rng_a = OxidizeRng::new(12345);
    let mut rng_b = OxidizeRng::new(12345);
    let norm = Normal::new(2.0, 0.5).unwrap();
    let gam = Gamma::new(0.7, 1.5).unwrap();
    for _ in 0..128 {
        assert_eq!(norm.sample(&mut rng_a), norm.sample(&mut rng_b));
        assert_eq!(gam.sample(&mut rng_a), gam.sample(&mut rng_b));
        assert_eq!(pois.sample(&mut rng_a), pois.sample(&mut rng_b));
        assert_eq!(cat.sample_index(&mut rng_a), cat.sample_index(&mut rng_b));
    }
}

#[test]
fn test_conjugate_families_and_zero_allocation() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let obs_norm = [1.8, 2.1, 1.9, 2.3, 2.0, 2.2];
    let counts_dm = [12_u64, 5, 3];
    let counts_gp = [3_u64, 5, 4, 6, 2];
    let mut log_w = [-2.0, -0.5, -1.2];
    let mut post_buf = [0.0; 3];

    let ((lml_all, lml_split), _) = VerifiedEngine::run_verified(|| {
        let bb = BetaBinomial::new(2.0, 2.0).unwrap().update(14, 6);
        let (bb_lo, bb_hi) = bb.credible_interval(0.95).unwrap();
        assert!(bb_lo < bb.posterior_mean() && bb.posterior_mean() < bb_hi);
        let bf_bb = bb
            .bayes_factor(&BetaBinomial::uniform_prior(), 14, 20)
            .unwrap();
        assert!(bf_bb.is_finite() && bf_bb > 0.0);

        let dm = DirichletMultinomial::new([1.0, 1.0, 1.0])
            .unwrap()
            .update(&counts_dm);
        let means = dm.posterior_means();
        let (d0_lo, d0_hi) = dm.credible_interval(0, 0.90).unwrap();
        assert!(d0_lo < means[0] && means[0] < d0_hi);

        let nn_prior = NormalNormal::new(0.0, 2.0, 0.5).unwrap();
        let nn_post = nn_prior.update_slice(&obs_norm).unwrap();
        let (nn_lo, nn_hi) = nn_post.credible_interval(0.95).unwrap();
        assert!(nn_lo < nn_post.posterior_mean() && nn_post.posterior_mean() < nn_hi);
        let bf_nn = nn_prior
            .bayes_factor(&NormalNormal::new(5.0, 0.2, 0.5).unwrap(), &obs_norm)
            .unwrap();
        assert!(bf_nn > 10.0);

        let nig_prior = NormalInverseGamma::new(0.0, 1.0, 2.0, 1.0).unwrap();
        let nig_post = nig_prior.update_slice(&obs_norm).unwrap();
        let (m_lo, m_hi) = nig_post.mean_credible_interval(0.95).unwrap();
        let (v_lo, v_hi) = nig_post.variance_credible_interval(0.95).unwrap();
        assert!(m_lo < nig_post.posterior_mean() && nig_post.posterior_mean() < m_hi);
        assert!(0.0 < v_lo && v_lo < v_hi);
        let lml_all = nig_prior.log_marginal_likelihood(&obs_norm).unwrap();
        let lml_a = nig_prior.log_marginal_likelihood(&obs_norm[..3]).unwrap();
        let lml_b = nig_prior
            .update_slice(&obs_norm[..3])
            .unwrap()
            .log_marginal_likelihood(&obs_norm[3..])
            .unwrap();

        let gp = GammaPoisson::new(2.0, 1.0)
            .unwrap()
            .update_counts(&counts_gp);
        let (g_lo, g_hi) = gp.credible_interval(0.95).unwrap();
        assert!(g_lo < gp.posterior_mean() && gp.posterior_mean() < g_hi);

        normalize_log_weights_into(&log_w, &mut post_buf).unwrap();
        log_w[0] = -1.5;
        discrete_posterior_into(&[0.7, 0.2, 0.1], &[0.5, 0.3, 0.2], &mut post_buf).unwrap();
        (lml_all, lml_a + lml_b)
    });
    assert_relative_eq!(lml_all, lml_split, max_relative = 1e-12);
}

#[test]
fn test_conformal_empirical_coverage_and_e_values_zero_alloc() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut rng = OxidizeRng::new(2026);
    let noise = Exponential::new(1.0).unwrap();
    let alpha = 0.10;
    let n_cal = 199usize;
    let mut cal_scores = [0.0_f64; 199];
    let mut covered = 0usize;
    let trials = 600usize;

    let (p_val, _) = VerifiedEngine::run_verified(|| {
        for _ in 0..trials {
            for slot in &mut cal_scores {
                *slot = noise.sample(&mut rng);
            }
            let q = conformal_quantile_into(&mut cal_scores, alpha).unwrap();
            let y_test = noise.sample(&mut rng);
            if y_test <= q {
                covered += 1;
            }
        }
        let mut mask = [false; 4];
        let count = conformal_prediction_mask_into(&[0.1, 0.4, 0.8, 0.95], 0.5, &mut mask).unwrap();
        assert_eq!(count, 2);
        let p = conformal_p_value(&[0.1, 0.2, 0.5, 0.9], 0.3).unwrap();
        let ev = EValue::from_p_value_power(0.01, 0.5).unwrap();
        assert!(ev.multiply(ev).rejects_at(0.05).unwrap());
        p
    });

    assert_relative_eq!(p_val, 3.0 / 5.0, max_relative = 1e-14);
    let coverage = (covered as f64) / (trials as f64);
    let slack = 3.0 / ((n_cal as f64).sqrt());
    assert!(coverage >= (1.0 - alpha) - slack);
}

#[test]
fn test_wasserstein_and_log_domain_sinkhorn_small_epsilon_zero_alloc() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut u = [0.0, 3.0, 1.0, 2.0];
    let mut v = [1.0, 4.0, 2.0, 3.0];
    let support = [0.0, 1.0, 2.0];
    let wa = [1.0, 0.0, 0.0];
    let wb = [0.0, 0.0, 1.0];
    let cost_3x3 = [0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0];
    let mut scratch = [0.0_f64; sinkhorn_scratch_len(3, 3)];
    let mut plan = [0.0_f64; 9];

    let ((w1, w_hist, plan_sum), _) = VerifiedEngine::run_verified(|| {
        let w1 = wasserstein_1d_into(&mut u, &mut v).unwrap();
        let w_hist = wasserstein_1d_weights(&wa, &wb, &support).unwrap();
        for eps in [0.5, 0.05, 1.0e-3, 1.0e-4] {
            let cfg = SinkhornConfig {
                epsilon: eps,
                max_iters: 400,
                tolerance: 1.0e-9,
            };
            let rep = sinkhorn_log_domain_into(
                &[0.5, 0.3, 0.2],
                &[0.2, 0.3, 0.5],
                &cost_3x3,
                &cfg,
                &mut scratch,
                Some(&mut plan),
            )
            .unwrap();
            assert!(rep.transport_cost.is_finite());
            assert!(plan.iter().all(|p| p.is_finite() && *p >= 0.0));
        }
        (w1, w_hist, plan.iter().sum::<f64>())
    });
    assert_relative_eq!(w1, 1.0, max_relative = 1e-14);
    assert_relative_eq!(w_hist, 2.0, max_relative = 1e-14);
    assert_relative_eq!(plan_sum, 1.0, epsilon = 1e-7);
}
