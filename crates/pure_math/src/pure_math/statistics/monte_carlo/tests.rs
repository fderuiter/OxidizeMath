use super::*;
use oxidize_core::rng::OxidizeRng;
use verified_engine::engine::VerifiedEngine;

#[test]
#[verified_engine::verified]
fn test_variance_reduction_modes_beat_iid_rmse_at_equal_budget() {
    // Target: E[sum_{j=1}^3 exp(U_j)] = 3 * (e - 1).
    // Control variate g(U) = sum_{j=1}^3 U_j with E[g(U)] = 1.5.
    let exact = 3.0 * (std::f64::consts::E - 1.0);
    let budget = 256_usize;
    let trials = 40_usize;
    let mut ws = MonteCarloWorkspace::new(budget, 3).expect("valid workspace");

    let mut iid_sq = 0.0;
    let mut anti_sq = 0.0;
    let mut cv_sq = 0.0;
    let mut lhs_sq = 0.0;
    let mut halton_sq = 0.0;
    let mut sobol_sq = 0.0;

    for trial in 0..trials {
        let seed = 10_000 + trial as u64;
        let f = |u: &[f64]| u[0].exp() + u[1].exp() + u[2].exp();

        let iid = ws
            .estimate_iid(budget, &mut OxidizeRng::new(seed), f)
            .unwrap();
        let anti = ws
            .estimate_antithetic(budget, &mut OxidizeRng::new(seed), f)
            .unwrap();
        let cv = ws
            .estimate_control_variate(budget, 1.5, &mut OxidizeRng::new(seed), |u| {
                (f(u), u[0] + u[1] + u[2])
            })
            .unwrap();
        let lhs = ws
            .estimate_latin_hypercube(budget, &mut OxidizeRng::new(seed), f)
            .unwrap();
        let halton = ws
            .estimate_qmc_halton(budget, &mut OxidizeRng::new(seed), f)
            .unwrap();
        let sobol = ws
            .estimate_qmc_sobol(budget, &mut OxidizeRng::new(seed), f)
            .unwrap();

        iid_sq += (iid.mean - exact).powi(2);
        anti_sq += (anti.mean - exact).powi(2);
        cv_sq += (cv.mean - exact).powi(2);
        lhs_sq += (lhs.mean - exact).powi(2);
        halton_sq += (halton.mean - exact).powi(2);
        sobol_sq += (sobol.mean - exact).powi(2);
    }

    let iid_rmse = (iid_sq / trials as f64).sqrt();
    let anti_rmse = (anti_sq / trials as f64).sqrt();
    let cv_rmse = (cv_sq / trials as f64).sqrt();
    let lhs_rmse = (lhs_sq / trials as f64).sqrt();
    let halton_rmse = (halton_sq / trials as f64).sqrt();
    let sobol_rmse = (sobol_sq / trials as f64).sqrt();

    assert!(
        anti_rmse < iid_rmse * 0.5,
        "anti={anti_rmse} vs iid={iid_rmse}"
    );
    assert!(cv_rmse < iid_rmse * 0.5, "cv={cv_rmse} vs iid={iid_rmse}");
    assert!(
        lhs_rmse < iid_rmse * 0.5,
        "lhs={lhs_rmse} vs iid={iid_rmse}"
    );
    assert!(
        halton_rmse < iid_rmse * 0.5,
        "halton={halton_rmse} vs iid={iid_rmse}"
    );
    assert!(
        sobol_rmse < iid_rmse * 0.5,
        "sobol={sobol_rmse} vs iid={iid_rmse}"
    );
}

#[test]
#[verified_engine::verified]
fn test_importance_sampling_exact_ess_and_degenerate_weights() {
    // Uniform weights [2, 2, 2, 2] => ESS = (8)^2 / (4 * 4) = 4.0.
    let ess_uniform = effective_sample_size(&[2.0, 2.0, 2.0, 2.0]).unwrap();
    assert!((ess_uniform - 4.0).abs() < 1e-12);

    // Weights [1, 3] => ESS = 16 / 10 = 1.6.
    let ess_skew = effective_sample_size(&[1.0, 3.0]).unwrap();
    assert!((ess_skew - 1.6).abs() < 1e-12);

    // Degenerate single-particle dominance [1.0, 0.0, 0.0, 0.0] => ESS = 1.0 -> DegenerateWeights.
    assert!(matches!(
        effective_sample_size(&[1.0, 0.0, 0.0, 0.0]),
        Err(MonteCarloError::DegenerateWeights { .. })
    ));
    assert!(matches!(
        effective_sample_size(&[0.0, 0.0, 0.0]),
        Err(MonteCarloError::DegenerateWeights { .. })
    ));

    // Unnormalized target p~(x) = exp(-2x) (true normalized distribution Exp(2) with mean 0.5)
    // sampled from heavier-tailed proposal q(x) = exp(-x) (X = -ln(u), w(x) = exp(-x) in (0, 1]).
    // Exact ESS/N -> (1/2)^2 / (1/3) = 0.75 (ESS ~= 768 for N = 1024).
    let exact_mean = 0.5;
    let mut ws = MonteCarloWorkspace::new(1024, 1).unwrap();
    let mut rng = OxidizeRng::new(2026);
    let est = ws
        .estimate_importance_sampling(1024, &mut rng, |u| {
            let x = -u[0].ln();
            let weight = (-x).exp();
            (x, weight)
        })
        .expect("importance sampling succeeds");

    assert!(
        (est.mean - exact_mean).abs() < 0.05,
        "SNIS mean {} vs exact {}",
        est.mean,
        exact_mean
    );
    assert!(est.ci_lower <= exact_mean && exact_mean <= est.ci_upper);
    assert!(est.effective_sample_size > 650.0 && est.effective_sample_size < 900.0);

    // Degenerate importance sampling rejected safely.
    let mut counter = 0_usize;
    let deg = ws.estimate_importance_sampling(64, &mut rng, |_| {
        counter += 1;
        let w = if counter == 1 { 1.0 } else { 0.0 };
        (1.0, w)
    });
    assert!(matches!(
        deg,
        Err(MonteCarloError::DegenerateWeights { .. })
    ));
}

#[test]
#[verified_engine::verified]
fn test_monte_carlo_workspace_zero_allocations_under_verified_allocator() {
    let mut ws = MonteCarloWorkspace::new(256, 4).expect("workspace");
    let mut rng = OxidizeRng::new(77);

    let (results, _) = VerifiedEngine::run_verified(|| {
        let e1 = ws
            .estimate_iid(128, &mut rng, |u| u[0] + u[1] + u[2] + u[3])
            .unwrap();
        let e2 = ws
            .estimate_antithetic(128, &mut rng, |u| u[0].exp() + u[1].exp())
            .unwrap();
        let e3 = ws
            .estimate_control_variate(128, 0.5, &mut rng, |u| (u[0].exp(), u[0]))
            .unwrap();
        let e4 = ws
            .estimate_importance_sampling(128, &mut rng, |u| (u[0], 0.5 + u[0]))
            .unwrap();
        let e5 = ws
            .estimate_latin_hypercube(128, &mut rng, |u| u[0] * u[1] + u[2])
            .unwrap();
        let e6 = ws
            .estimate_qmc_sobol(128, &mut rng, |u| u[0].sin() + u[3].cos())
            .unwrap();
        let e7 = ws
            .estimate_qmc_halton(128, &mut rng, |u| u[0].sin() + u[3].cos())
            .unwrap();
        e1.mean + e2.mean + e3.mean + e4.mean + e5.mean + e6.mean + e7.mean
    });
    assert!(results.is_finite());
}

// [cite:clinical_trials_statistics]
