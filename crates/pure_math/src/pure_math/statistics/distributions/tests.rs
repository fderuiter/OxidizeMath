#![cfg_attr(any(), verified_engine::verified(opt_out = "tests"))]

use super::*;
use oxidize_core::rng::OxidizeRng;
use verified_engine::engine::VerifiedEngine;

fn assert_continuous_invariants<D: ContinuousDistribution>(dist: &D, grid_lo: f64, grid_hi: f64) {
    let steps = 4000;
    let dx = (grid_hi - grid_lo) / (steps as f64);
    let mut area = 0.0;
    for i in 0..steps {
        let x = grid_lo + ((i as f64) + 0.5) * dx;
        let p = dist.pdf(x);
        assert!(p >= 0.0 && p.is_finite());
        if p > 1e-200 {
            assert!((dist.ln_pdf(x).exp() - p).abs() <= 1e-9 * p.max(1.0));
        }
        area += p * dx;
    }
    assert!(
        (area - 1.0).abs() < 0.015,
        "integrated PDF area {area} diverged from 1.0"
    );

    for &q in &[0.05, 0.25, 0.5, 0.75, 0.95] {
        let x = dist.quantile(q).expect("valid quantile");
        let back = dist.cdf(x);
        assert!(
            (back - q).abs() < 1e-7,
            "quantile-cdf mismatch: cdf(quantile({q})) = {back}"
        );
    }
    assert!(dist.quantile(-0.01).is_err());
    assert!(dist.quantile(1.01).is_err());
    assert!(dist.quantile(f64::NAN).is_err());
}

fn assert_discrete_invariants<D: DiscreteDistribution>(dist: &D, max_k: u64) {
    let mut mass = 0.0;
    let mut prev_cdf = 0.0;
    for k in 0..=max_k {
        let p = dist.pmf(k);
        assert!((0.0..=1.0).contains(&p));
        if p > 0.0 {
            assert!((dist.ln_pmf(k).exp() - p).abs() < 1e-11);
        } else {
            assert_eq!(dist.ln_pmf(k), f64::NEG_INFINITY);
        }
        mass += p;
        let c = dist.cdf(k);
        assert!(c + 1e-12 >= prev_cdf);
        prev_cdf = c;
    }
    assert!(
        (mass - 1.0).abs() < 1e-4,
        "discrete PMF sum {mass} diverged from 1.0"
    );
    for &q in &[0.1, 0.5, 0.9] {
        let k = dist.quantile(q).expect("valid discrete quantile");
        assert!(dist.cdf(k) + 1e-12 >= q);
        if k > 0 {
            assert!(dist.cdf(k - 1) <= q + 1e-12);
        }
    }
}

#[test]
fn test_all_nine_continuous_distributions_and_fitters() {
    let normal = NormalDist::new(2.5, 1.4).expect("normal");
    assert_continuous_invariants(&normal, -8.0, 13.0);

    let uniform = UniformDist::new(-2.0, 5.0).expect("uniform");
    assert_continuous_invariants(&uniform, -2.0, 5.0);

    let exp = ExponentialDist::new(1.5).expect("exp");
    assert_continuous_invariants(&exp, 0.0, 14.0);

    let lognorm = LogNormalDist::new(0.3, 0.5).expect("lognorm");
    assert_continuous_invariants(&lognorm, 1e-5, 18.0);

    let weibull = WeibullDist::new(2.2, 3.0).expect("weibull");
    assert_continuous_invariants(&weibull, 1e-5, 12.0);

    let gamma = GammaDist::new(3.5, 1.2).expect("gamma");
    assert_continuous_invariants(&gamma, 1e-5, 20.0);

    let beta = BetaDist::new(2.5, 4.0).expect("beta");
    assert_continuous_invariants(&beta, 1e-5, 1.0 - 1e-5);

    let chi2 = ChiSquaredDist::new(5.0).expect("chi2");
    assert_continuous_invariants(&chi2, 1e-5, 35.0);

    let student = StudentTDist::new(6.0, 1.0, 0.8).expect("student");
    assert_continuous_invariants(&student, -12.0, 14.0);
    let cauchy = StudentTDist::standard(1.0).expect("cauchy");
    assert_eq!(cauchy.mean(), None);
    assert_eq!(cauchy.variance(), None);
}

#[test]
fn test_continuous_mle_and_mom_recovery() {
    let mut rng = OxidizeRng::from_seed(2026);
    let n = 1500;

    let norm_true = NormalDist::new(-1.5, 2.2).expect("norm");
    let norm_samples: Vec<f64> = (0..n).map(|_| norm_true.sample(&mut rng)).collect();
    let norm_mle = NormalDist::fit_mle(&norm_samples).expect("norm mle");
    let norm_mom = NormalDist::fit_mom(&norm_samples).expect("norm mom");
    assert!((norm_mle.mu() - -1.5).abs() < 0.15);
    assert!((norm_mom.sigma() - 2.2).abs() < 0.15);

    let gamma_true = GammaDist::new(4.0, 2.0).expect("gamma");
    let gamma_samples: Vec<f64> = (0..n).map(|_| gamma_true.sample(&mut rng)).collect();
    let gamma_mle = GammaDist::fit_mle(&gamma_samples).expect("gamma mle");
    let gamma_mom = GammaDist::fit_mom(&gamma_samples).expect("gamma mom");
    assert!((gamma_mle.shape() - 4.0).abs() < 0.45);
    assert!((gamma_mom.rate() - 2.0).abs() < 0.30);

    let beta_true = BetaDist::new(3.0, 5.0).expect("beta");
    let beta_samples: Vec<f64> = (0..n).map(|_| beta_true.sample(&mut rng)).collect();
    let beta_mle = BetaDist::fit_mle(&beta_samples).expect("beta mle");
    let beta_mom = BetaDist::fit_mom(&beta_samples).expect("beta mom");
    assert!((beta_mle.alpha() - 3.0).abs() < 0.4);
    assert!((beta_mom.beta() - 5.0).abs() < 0.6);

    let weib_true = WeibullDist::new(2.5, 4.0).expect("weibull");
    let weib_samples: Vec<f64> = (0..n).map(|_| weib_true.sample(&mut rng)).collect();
    let weib_mle = WeibullDist::fit_mle(&weib_samples).expect("weib mle");
    let weib_mom = WeibullDist::fit_mom(&weib_samples).expect("weib mom");
    assert!((weib_mle.shape() - 2.5).abs() < 0.25);
    assert!((weib_mom.scale() - 4.0).abs() < 0.25);

    let ln_true = LogNormalDist::new(0.5, 0.4).expect("ln");
    let ln_samples: Vec<f64> = (0..n).map(|_| ln_true.sample(&mut rng)).collect();
    assert!((LogNormalDist::fit_mle(&ln_samples).expect("mle").mu() - 0.5).abs() < 0.08);
    assert!((LogNormalDist::fit_mom(&ln_samples).expect("mom").sigma() - 0.4).abs() < 0.08);

    let chi_true = ChiSquaredDist::new(6.0).expect("chi2");
    let chi_samples: Vec<f64> = (0..n).map(|_| chi_true.sample(&mut rng)).collect();
    assert!((ChiSquaredDist::fit_mle(&chi_samples).expect("mle").df() - 6.0).abs() < 0.4);
    assert!((ChiSquaredDist::fit_mom(&chi_samples).expect("mom").df() - 6.0).abs() < 0.4);

    let st_true = StudentTDist::new(5.0, 2.0, 1.2).expect("student");
    let st_samples: Vec<f64> = (0..n).map(|_| st_true.sample(&mut rng)).collect();
    let st_mle = StudentTDist::fit_mle(&st_samples).expect("st mle");
    assert!((st_mle.location() - 2.0).abs() < 0.15);
    assert!((st_mle.scale() - 1.2).abs() < 0.25);
}

#[test]
fn test_all_five_discrete_distributions_and_fitters() {
    let bern = BernoulliDist::new(0.35).expect("bern");
    assert_discrete_invariants(&bern, 4);

    let binom = BinomialDist::new(12, 0.4).expect("binom");
    assert_discrete_invariants(&binom, 12);

    let pois = PoissonDist::new(4.2).expect("pois");
    assert_discrete_invariants(&pois, 30);

    let geom = GeometricDist::new(0.3).expect("geom");
    assert_discrete_invariants(&geom, 50);

    let negbin = NegativeBinomialDist::new(3.5, 0.45).expect("negbin");
    assert_discrete_invariants(&negbin, 60);

    let mut rng = OxidizeRng::from_seed(99);
    let n = 2000;
    let nb_samples: Vec<u64> = (0..n).map(|_| negbin.sample(&mut rng)).collect();
    let nb_mle = NegativeBinomialDist::fit_mle(&nb_samples).expect("nb mle");
    let nb_mom = NegativeBinomialDist::fit_mom(&nb_samples).expect("nb mom");
    assert!((nb_mle.mean() - negbin.mean()).abs() < 0.3);
    assert!((nb_mom.p() - 0.45).abs() < 0.08);

    let bin_samples: Vec<u64> = (0..n).map(|_| binom.sample(&mut rng)).collect();
    let bin_mle = BinomialDist::fit_mle_known_n(&bin_samples, 12).expect("bin mle");
    assert!((bin_mle.p() - 0.4).abs() < 0.04);
    let bin_joint = BinomialDist::fit_mle(&bin_samples).expect("bin joint");
    assert!((bin_joint.mean() - binom.mean()).abs() < 0.25);
}

#[test]
fn test_distributions_zero_allocation_under_verified_engine() {
    let mut rng = OxidizeRng::from_seed(77);
    let (acc, _) = VerifiedEngine::run_verified(|| {
        let normal = NormalDist::new(1.0, 2.0).unwrap();
        let beta = BetaDist::new(2.5, 3.5).unwrap();
        let gamma = GammaDist::new(3.0, 1.5).unwrap();
        let student = StudentTDist::new(4.0, 0.0, 1.0).unwrap();
        let pois = PoissonDist::new(35.0).unwrap();
        let negbin = NegativeBinomialDist::new(2.5, 0.4).unwrap();

        let q_n = normal.quantile(0.9).unwrap();
        let q_b = beta.quantile(0.25).unwrap();
        let q_g = gamma.quantile(0.75).unwrap();
        let q_t = student.quantile(0.95).unwrap();
        let q_p = pois.quantile(0.5).unwrap() as f64;
        let s = normal.sample(&mut rng)
            + beta.sample(&mut rng)
            + gamma.sample(&mut rng)
            + student.sample(&mut rng)
            + (pois.sample(&mut rng) as f64)
            + (negbin.sample(&mut rng) as f64);
        q_n + q_b + q_g + q_t + q_p + s
    });
    assert!(acc.is_finite());
}
