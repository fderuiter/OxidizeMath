#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Analytical conjugate posterior curves, credible intervals, and Bayes factors for `BayesianInferenceTool`.

use super::bayesian_tool::{BayesianAnalyticalCache, BayesianToolConfig, ConjugateFamily};
use math_explorer::pure_math::statistics::bayesian::{
    BayesFactor, BetaBinomial, GammaPoisson, NormalInverseGamma, NormalNormal,
    hpd_credible_interval,
};
use math_explorer::pure_math::statistics::distributions::{
    BetaDist, ContinuousDistribution, GammaDist, NormalDist, StudentTDist,
};
use math_explorer::rng::OxidizeRng;

pub(crate) fn eval_log_posterior(cfg: &BayesianToolConfig, theta: f64) -> f64 {
    match cfg.family {
        ConjugateFamily::BetaBinomial => {
            let a0 = cfg.prior_a.clamp(0.2, 20.0);
            let b0 = cfg.prior_b.clamp(0.2, 20.0);
            let n = cfg.obs_count as u64;
            let k = (cfg.obs_stat.round().clamp(0.0, n as f64)) as u64;
            BetaDist::new(a0 + k as f64, b0 + (n - k) as f64)
                .map_or(f64::NEG_INFINITY, |d| d.ln_pdf(theta))
        }
        ConjugateFamily::GammaPoisson => {
            let a0 = cfg.prior_a.clamp(0.2, 20.0);
            let b0 = cfg.prior_b.clamp(0.2, 20.0);
            let per_obs = (cfg.obs_stat.clamp(0.0, 20.0) * 0.25).round() as u64;
            let total = (per_obs * (cfg.obs_count as u64)) as f64;
            GammaDist::new(a0 + total, b0 + cfg.obs_count as f64)
                .map_or(f64::NEG_INFINITY, |d| d.ln_pdf(theta))
        }
        ConjugateFamily::NormalNormal => {
            let prior = NormalNormal::new(
                cfg.prior_a.clamp(-5.0, 10.0),
                cfg.prior_b.clamp(0.2, 10.0),
                1.0,
            )
            .expect("valid NormalNormal");
            let post = prior
                .update_summary(cfg.obs_stat.clamp(-5.0, 10.0), cfg.obs_count)
                .expect("valid Normal update");
            post.posterior_dist()
                .map_or(f64::NEG_INFINITY, |d| d.ln_pdf(theta))
        }
        ConjugateFamily::NormalInverseGamma => {
            let ( _, post) = build_nig_models(cfg);
            post.marginal_mean_dist()
                .map_or(f64::NEG_INFINITY, |d| d.ln_pdf(theta))
        }
    }
}

pub(crate) fn compute_analytical(cfg: &BayesianToolConfig) -> BayesianAnalyticalCache {
    let level = cfg.credible_level.clamp(0.50, 0.99);
    match cfg.family {
        ConjugateFamily::BetaBinomial => analytical_beta_binomial(cfg, level),
        ConjugateFamily::GammaPoisson => analytical_gamma_poisson(cfg, level),
        ConjugateFamily::NormalNormal => analytical_normal_normal(cfg, level),
        ConjugateFamily::NormalInverseGamma => analytical_nig(cfg, level),
    }
}

fn sample_curve<D: ContinuousDistribution>(dist: &D, lo: f64, hi: f64) -> Vec<[f64; 2]> {
    let span = (hi - lo).max(1e-3);
    (0..128)
        .map(|i| {
            let x = lo + ((i as f64 + 0.5) / 128.0) * span;
            let y = dist.pdf(x);
            [x, if y.is_finite() { y.min(30.0) } else { 0.0 }]
        })
        .collect()
}

fn analytical_beta_binomial(cfg: &BayesianToolConfig, level: f64) -> BayesianAnalyticalCache {
    let a0 = cfg.prior_a.clamp(0.2, 20.0);
    let b0 = cfg.prior_b.clamp(0.2, 20.0);
    let n = cfg.obs_count as u64;
    let k = (cfg.obs_stat.round().clamp(0.0, n as f64)) as u64;
    let prior = BetaBinomial::new(a0, b0).expect("valid BetaBinomial");
    let post = prior.update(k, n).expect("valid BetaBinomial update");
    let prior_d = prior.posterior_dist().expect("prior BetaDist");
    let post_d = post.posterior_dist().expect("post BetaDist");
    BayesianAnalyticalCache {
        prior_curve: sample_curve(&prior_d, 0.0, 1.0),
        posterior_curve: sample_curve(&post_d, 0.0, 1.0),
        plot_bounds: (0.0, 1.0),
        posterior_mean: post.posterior_mean(),
        posterior_std: post.posterior_variance().sqrt(),
        equal_tailed: post.credible_interval(level).expect("ET CI"),
        hpd: post.hpd_interval(level).expect("HPD CI"),
        bayes_factor: prior.bayes_factor_vs_point(0.5, k, n).expect("BF10"),
    }
}

fn analytical_gamma_poisson(cfg: &BayesianToolConfig, level: f64) -> BayesianAnalyticalCache {
    let a0 = cfg.prior_a.clamp(0.2, 20.0);
    let b0 = cfg.prior_b.clamp(0.2, 20.0);
    let prior = GammaPoisson::new(a0, b0).expect("valid GammaPoisson");
    let per_obs = (cfg.obs_stat.clamp(0.0, 20.0) * 0.25).round() as u64;
    let counts = vec![per_obs; cfg.obs_count];
    let post = prior.update_counts(&counts).expect("valid Gamma update");
    let prior_d = prior.posterior_dist().expect("prior GammaDist");
    let post_d = post.posterior_dist().expect("post GammaDist");
    let hi = post_d
        .quantile(0.998)
        .unwrap_or(8.0)
        .max(prior_d.quantile(0.98).unwrap_or(5.0));
    BayesianAnalyticalCache {
        prior_curve: sample_curve(&prior_d, 0.0, hi),
        posterior_curve: sample_curve(&post_d, 0.0, hi),
        plot_bounds: (0.0, hi),
        posterior_mean: post.posterior_mean(),
        posterior_std: post.posterior_variance().sqrt(),
        equal_tailed: post.credible_interval(level).expect("ET CI"),
        hpd: post.hpd_interval(level).expect("HPD CI"),
        bayes_factor: prior.bayes_factor_vs_rate(1.0, &counts).expect("Gamma BF10"),
    }
}

fn analytical_normal_normal(cfg: &BayesianToolConfig, level: f64) -> BayesianAnalyticalCache {
    let mu0 = cfg.prior_a.clamp(-5.0, 10.0);
    let tau2 = cfg.prior_b.clamp(0.2, 10.0);
    let x_bar = cfg.obs_stat.clamp(-5.0, 10.0);
    let prior = NormalNormal::new(mu0, tau2, 1.0).expect("valid NormalNormal");
    let data = vec![x_bar; cfg.obs_count];
    let post = prior.update_slice(&data).expect("Normal update");
    let prior_d = prior.posterior_dist().expect("prior Normal");
    let post_d = post.posterior_dist().expect("post Normal");
    let lo = prior_d
        .quantile(0.01)
        .unwrap_or(-4.0)
        .min(post_d.quantile(0.005).unwrap_or(-4.0));
    let hi = prior_d
        .quantile(0.99)
        .unwrap_or(4.0)
        .max(post_d.quantile(0.995).unwrap_or(4.0));
    let et = post.credible_interval(level).expect("Normal CI");
    BayesianAnalyticalCache {
        prior_curve: sample_curve(&prior_d, lo, hi),
        posterior_curve: sample_curve(&post_d, lo, hi),
        plot_bounds: (lo, hi),
        posterior_mean: post.posterior_mean(),
        posterior_std: post.posterior_variance().sqrt(),
        equal_tailed: et,
        hpd: et,
        bayes_factor: prior.bayes_factor_vs_point(0.0, &data).expect("Normal BF10"),
    }
}

fn build_nig_models(cfg: &BayesianToolConfig) -> (Vec<f64>, NormalInverseGamma) {
    let prior = NormalInverseGamma::new(
        cfg.prior_a.clamp(-5.0, 10.0),
        1.5,
        cfg.prior_b.clamp(1.2, 12.0),
        1.5,
    )
    .expect("valid NIG prior");
    let mut rng = OxidizeRng::from_seed(cfg.seed);
    let gen_d = NormalDist::new(cfg.obs_stat.clamp(-5.0, 10.0), 1.1).expect("gen Normal");
    let data: Vec<f64> = (0..cfg.obs_count).map(|_| gen_d.sample(&mut rng)).collect();
    let post = prior.update_slice(&data).expect("NIG update");
    (data, post)
}

fn analytical_nig(cfg: &BayesianToolConfig, level: f64) -> BayesianAnalyticalCache {
    let prior = NormalInverseGamma::new(
        cfg.prior_a.clamp(-5.0, 10.0),
        1.5,
        cfg.prior_b.clamp(1.2, 12.0),
        1.5,
    )
    .expect("valid NIG prior");
    let (data, post) = build_nig_models(cfg);
    let prior_t: StudentTDist = prior.marginal_mean_dist().expect("prior t");
    let post_t: StudentTDist = post.marginal_mean_dist().expect("post t");
    let lo = prior_t
        .quantile(0.02)
        .unwrap_or(-4.0)
        .min(post_t.quantile(0.005).unwrap_or(-4.0));
    let hi = prior_t
        .quantile(0.98)
        .unwrap_or(4.0)
        .max(post_t.quantile(0.995).unwrap_or(4.0));
    let et = post.credible_interval_mean(level).expect("NIG ET CI");
    let hpd = hpd_credible_interval(&post_t, level).unwrap_or(et);
    let ln_m1 = prior.ln_marginal_likelihood(&data).unwrap_or(0.0);
    let null_d = NormalDist::standard();
    let ln_m0: f64 = data.iter().map(|&x| null_d.ln_pdf(x)).sum();
    let bf = BayesFactor::from_ln_marginals(ln_m1, ln_m0).expect("NIG BF");
    BayesianAnalyticalCache {
        prior_curve: sample_curve(&prior_t, lo, hi),
        posterior_curve: sample_curve(&post_t, lo, hi),
        plot_bounds: (lo, hi),
        posterior_mean: post.mu(),
        posterior_std: post_t.variance().unwrap_or(1.0).sqrt(),
        equal_tailed: et,
        hpd,
        bayes_factor: bf,
    }
}

// [cite:clinical_trials_statistics]
