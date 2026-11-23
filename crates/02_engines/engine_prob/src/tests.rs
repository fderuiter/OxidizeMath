use crate::bootstrap::{
    bca_bootstrap_mean, bca_bootstrap_median, bootstrap_indices, percentile_bootstrap_mean,
    percentile_bootstrap_median, BootstrapMethod,
};
use crate::cubature::{
    gaussian_copula_richtmyer_cdf, integrate_hyperrect, integrate_unit_cube, CubatureConfig,
    CubatureRule,
};
use crate::error::ProbError;
use crate::math::{cholesky_lower_row_major, standard_normal_cdf, standard_normal_inv_cdf};
use crate::variance_reduction::{
    estimate_antithetic_normal, estimate_antithetic_uniform, estimate_control_variate,
    estimate_self_normalized_is, self_normalized_is_from_slices,
    simulate_correlated_bernoulli_antithetic,
};
use crate::workspace::SolverWorkspace;
use oxidize_core::rng::{OxidizeRng, BASE_SEED};

#[test]
fn test_normal_cdf_and_inv_cdf_roundtrip() {
    for &p in &[
        1e-12,
        1e-6,
        0.01,
        0.025,
        0.25,
        0.5,
        0.75,
        0.975,
        0.99,
        1.0 - 1e-9,
    ] {
        let z = standard_normal_inv_cdf(p);
        let p_back = standard_normal_cdf(z);
        assert!((p_back - p).abs() < 1e-13, "p={p}, z={z}, p_back={p_back}");
    }
}

#[test]
fn test_richtmyer_halton_sobol_cubature_and_shift_bounds() {
    let mut floats = [0.0_f64; 64];
    let mut ws = SolverWorkspace::from_floats(&mut floats);

    // Integral over [0, 1]^3 of (x^2 + y*z + e^x):
    // int_0^1 x^2 dx + int_0^1 y dy * int_0^1 z dz + (e - 1) = 1/3 + 1/4 + (e - 1)
    let exact = (1.0 / 3.0) + 0.25 + (libm::exp(1.0) - 1.0);
    for rule in [
        CubatureRule::Richtmyer { baker_tent: true },
        CubatureRule::Halton,
        CubatureRule::Sobol,
    ] {
        let cfg = CubatureConfig::new(rule, 512).with_shifts(8);
        let est = integrate_unit_cube(3, &cfg, &mut ws, |u| {
            u[0] * u[0] + u[1] * u[2] + libm::exp(u[0])
        })
        .unwrap();
        assert!((est.value - exact).abs() < 5e-3, "{rule:?}: {}", est.value);
        assert!(est.std_error > 0.0 && est.std_error < 5e-3);
        assert!(est.lower_95 < est.upper_95);
    }

    // Hyperrectangle [0, 2] x [1, 3]: int_0^2 int_1^3 x * y dy dx = 2 * 4 = 8
    let cfg = CubatureConfig::new(CubatureRule::Sobol, 256);
    let rect =
        integrate_hyperrect(&[0.0, 1.0], &[2.0, 3.0], &cfg, &mut ws, |x| x[0] * x[1]).unwrap();
    assert!((rect.value - 8.0).abs() < 1e-2);
}

#[test]
fn test_genz_gaussian_copula_richtmyer_cdf() {
    let corr = [1.0, 0.5, 0.5, 1.0];
    let mut chol = [0.0; 4];
    cholesky_lower_row_major(&corr, 2, &mut chol).unwrap();
    let mut floats = [0.0; 32];
    let mut ws = SolverWorkspace::from_floats(&mut floats);

    // For rho = 0.5 at u = (0.5, 0.5), C(0.5, 0.5) = 1/4 + arcsin(0.5)/(2*pi) = 1/3.
    let est =
        gaussian_copula_richtmyer_cdf(&chol, &[0.5, 0.5], 2048, 4, BASE_SEED, &mut ws).unwrap();
    assert!((est.value - (1.0 / 3.0)).abs() < 2e-3);
    assert!(est.std_error >= 0.0);
}

#[test]
fn test_antithetic_control_variate_and_snis_estimators() {
    let mut floats = [0.0_f64; 4096];
    let mut ws = SolverWorkspace::from_floats(&mut floats);
    let mut rng = OxidizeRng::from_seed(BASE_SEED);

    // Antithetic normal on E[e^{0.5 Z}] = e^{0.125}
    let anti_norm =
        estimate_antithetic_normal(1, 1000, &mut rng, &mut ws, |z| libm::exp(0.5 * z[0])).unwrap();
    assert!((anti_norm.mean - libm::exp(0.125)).abs() < 0.02);

    // Antithetic uniform on E[e^U] = e - 1
    let anti_unif =
        estimate_antithetic_uniform(1, 1000, &mut rng, &mut ws, |u| libm::exp(u[0])).unwrap();
    assert!((anti_unif.mean - (libm::exp(1.0) - 1.0)).abs() < 0.01);

    // Control variate on Y = e^U using C = U with E[C] = 0.5: dramatic variance reduction
    let cv = estimate_control_variate(1000, 0.5, &mut rng, &mut ws, |r| {
        let u = r.next_f64_open01();
        (libm::exp(u), u)
    })
    .unwrap();
    assert!((cv.mean - (libm::exp(1.0) - 1.0)).abs() < 5e-3);
    assert!(cv.variance_reduction_ratio < 0.05);
    assert!(cv.std_error < cv.raw_std_error * 0.25);

    // Self-normalized importance sampling: target N(1, 1), proposal N(0, 1), E_target[X] = 1.0
    let snis = estimate_self_normalized_is(1500, &mut rng, &mut ws, |r| {
        let (z, _) = crate::math::box_muller_pair(r);
        let weight = libm::exp(z - 0.5);
        (z, weight)
    })
    .unwrap();
    assert!((snis.mean - 1.0).abs() < 0.1);
    assert!(snis.effective_sample_size > 200.0);
    assert_eq!(
        self_normalized_is_from_slices(&[1.0, 2.0], &[0.0, 0.0]).unwrap_err(),
        ProbError::ZeroEffectiveWeight
    );
}

#[test]
fn test_correlated_bernoulli_antithetic() {
    let corr = [1.0, 0.5, 0.5, 1.0];
    let mut chol = [0.0; 4];
    cholesky_lower_row_major(&corr, 2, &mut chol).unwrap();
    let mut floats = [0.0; 16];
    let mut ws = SolverWorkspace::from_floats(&mut floats);
    let mut hits = [0_usize; 2];
    let mut rng = OxidizeRng::from_seed(BASE_SEED);

    let rep = simulate_correlated_bernoulli_antithetic(
        &chol,
        &[0.0, 0.0],
        5_000,
        &mut rng,
        &mut hits,
        &mut ws,
    )
    .unwrap();
    assert!((rep.joint_prob - (1.0 / 3.0)).abs() < 0.015);
    assert_eq!(hits[0], 5_000);
    assert_eq!(hits[1], 5_000);
}

#[test]
fn test_percentile_and_bca_bootstrap_intervals_and_edge_cases() {
    let skewed = [0.4, 0.5, 0.6, 0.8, 1.1, 1.5, 2.4, 4.2, 7.9, 15.0];
    let mut floats = [0.0_f64; 1024];
    let mut indices = [0_usize; 32];
    let mut ws = SolverWorkspace::new(&mut floats, &mut indices);

    let mut rng_p = OxidizeRng::from_seed(BASE_SEED);
    let pct = percentile_bootstrap_mean(&skewed, 500, 0.95, &mut rng_p, &mut ws).unwrap();
    let mut rng_b = OxidizeRng::from_seed(BASE_SEED);
    let bca = bca_bootstrap_mean(&skewed, 500, 0.95, &mut rng_b, &mut ws).unwrap();

    assert!(
        bca.acceleration > 0.0,
        "right-skewed sample must have positive jackknife acceleration"
    );
    assert!(
        bca.upper > pct.upper,
        "BCa upper bound shifts right for positively skewed data"
    );

    let med_pct = percentile_bootstrap_median(&skewed, 400, 0.95, &mut rng_p, &mut ws).unwrap();
    let med_bca = bca_bootstrap_median(&skewed, 400, 0.95, &mut rng_b, &mut ws).unwrap();
    assert!(med_pct.lower <= med_pct.point_estimate && med_pct.point_estimate <= med_pct.upper);
    assert!(med_bca.lower <= med_bca.point_estimate && med_bca.point_estimate <= med_bca.upper);

    // Index-based BCa bootstrap parity
    let mut rng_idx = OxidizeRng::from_seed(BASE_SEED);
    let idx_bca = bootstrap_indices(
        skewed.len(),
        500,
        0.95,
        BootstrapMethod::BCa,
        &mut rng_idx,
        &mut ws,
        |idxs| idxs.iter().map(|&i| skewed[i]).sum::<f64>() / (idxs.len() as f64),
    )
    .unwrap();
    assert!((idx_bca.acceleration - bca.acceleration).abs() < 1e-12);

    // Constant sample edge case must not produce NaN
    let constant = [42.0; 8];
    let const_ci = bca_bootstrap_mean(&constant, 100, 0.95, &mut rng_b, &mut ws).unwrap();
    assert_eq!(const_ci.lower, 42.0);
    assert_eq!(const_ci.upper, 42.0);
    assert_eq!(const_ci.std_error, 0.0);

    // Undersized workspace returns InsufficientWorkspace
    let mut tiny = [0.0; 4];
    let mut tiny_ws = SolverWorkspace::from_floats(&mut tiny);
    assert!(matches!(
        bca_bootstrap_mean(&skewed, 100, 0.95, &mut rng_b, &mut tiny_ws),
        Err(ProbError::InsufficientWorkspace { .. })
    ));
}
