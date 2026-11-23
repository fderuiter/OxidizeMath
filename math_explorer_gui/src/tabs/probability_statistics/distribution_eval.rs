#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Deterministic curve evaluation and MLE/MoM fitting helpers for [`DistributionExplorerTool`].

use super::distribution_tool::{
    DistributionExplorerTool, DistributionFamily, DistributionSnapshot,
};
use math_explorer::pure_math::statistics::distributions::{
    BernoulliDist, BetaDist, BinomialDist, ChiSquaredDist, ContinuousDistribution,
    DiscreteDistribution, ExponentialDist, GammaDist, GeometricDist, LogNormalDist,
    NegativeBinomialDist, NormalDist, PoissonDist, StudentTDist, UniformDist, WeibullDist,
};
use math_explorer::pure_math::statistics::zip_regression::{
    simple_zip_fit, Count, ZipDistribution, ZipParams,
};
use math_explorer::rng::OxidizeRng;

const CURVE_POINTS: usize = 128;
const HIST_BINS: usize = 18;

impl DistributionExplorerTool {
    /// Recomputes theoretical curves, quantile, and seeded MLE/MoM recovery diagnostics.
    pub(crate) fn recompute(&mut self) {
        let mut rng = OxidizeRng::from_seed(self.config.seed);
        let p = self.config.quantile_p.clamp(0.01, 0.99);
        let n = self.config.sample_size.clamp(16, 512);
        let a = self.config.param_a;
        let b = self.config.param_b;

        self.snapshot = match self.config.family {
            DistributionFamily::Normal => eval_normal(a, b.max(0.1), p, n, &mut rng),
            DistributionFamily::Gamma => eval_gamma(a.clamp(0.2, 25.0), b.max(0.1), p, n, &mut rng),
            DistributionFamily::Beta => eval_beta(a.clamp(0.2, 25.0), b.max(0.2), p, n, &mut rng),
            DistributionFamily::Weibull => {
                eval_weibull(a.clamp(0.3, 15.0), b.max(0.1), p, n, &mut rng)
            }
            DistributionFamily::StudentT => {
                eval_student_t(a.clamp(1.2, 30.0), b.max(0.1), p, n, &mut rng)
            }
            DistributionFamily::LogNormal => {
                eval_lognormal(a.clamp(-2.5, 2.5), b.clamp(0.1, 1.8), p, n, &mut rng)
            }
            DistributionFamily::Exponential => eval_exponential(a.clamp(0.1, 15.0), p, n, &mut rng),
            DistributionFamily::ChiSquared => eval_chi_squared(a.clamp(0.5, 30.0), p, n, &mut rng),
            DistributionFamily::Uniform => {
                let lo = a.clamp(-8.0, 20.0);
                let span = b.clamp(0.2, 20.0);
                eval_uniform(lo, lo + span, p, n, &mut rng)
            }
            DistributionFamily::Poisson => eval_poisson(a.clamp(0.2, 25.0), p, n, &mut rng),
            DistributionFamily::Binomial => {
                let trials = (a.round().clamp(2.0, 30.0)) as u64;
                let prob = b.clamp(0.05, 0.95);
                eval_binomial(trials, prob, p, n, &mut rng)
            }
            DistributionFamily::NegativeBinomial => {
                eval_negbin(a.clamp(0.5, 25.0), b.clamp(0.08, 0.92), p, n, &mut rng)
            }
            DistributionFamily::Geometric => eval_geometric(a.clamp(0.05, 0.95), p, n, &mut rng),
            DistributionFamily::Bernoulli => eval_bernoulli(a.clamp(0.05, 0.95), p, n, &mut rng),
            DistributionFamily::ZeroInflatedPoisson => {
                eval_zip(a.clamp(0.02, 0.90), b.clamp(0.3, 20.0), p, n, &mut rng)
            }
        };
    }
}

fn build_continuous_snapshot<D: ContinuousDistribution>(
    dist: &D,
    p: f64,
    n: usize,
    rng: &mut OxidizeRng,
    labels: (&'static str, Option<&'static str>),
) -> (DistributionSnapshot, Vec<f64>) {
    let x_lo = dist.quantile(0.005).unwrap_or(-4.0);
    let x_hi = dist.quantile(0.995).unwrap_or(4.0);
    let (supp_lo, supp_hi) = dist.support();
    let lo = if supp_lo.is_finite() {
        supp_lo
    } else {
        x_lo
    };
    let hi = if supp_hi.is_finite() {
        supp_hi
    } else {
        x_hi.max(lo + 1e-2)
    };
    let span = (hi - lo).max(1e-3);

    let mut density_curve = Vec::with_capacity(CURVE_POINTS);
    let mut cdf_curve = Vec::with_capacity(CURVE_POINTS);
    for idx in 0..CURVE_POINTS {
        let frac = (idx as f64 + 0.5) / (CURVE_POINTS as f64);
        let x = lo + frac * span;
        let pdf = dist.pdf(x);
        density_curve.push([x, if pdf.is_finite() { pdf.min(25.0) } else { 0.0 }]);
        cdf_curve.push([x, dist.cdf(x)]);
    }

    let draws: Vec<f64> = (0..n).map(|_| dist.sample(rng)).collect();
    let histogram_bars = build_empirical_histogram(&draws, lo, hi);
    let snap = DistributionSnapshot {
        density_curve,
        cdf_curve,
        histogram_bars,
        discrete_bars: Vec::new(),
        mean: dist.mean(),
        variance: dist.variance(),
        quantile_x: dist.quantile(p).unwrap_or(lo),
        mle_summary: String::new(),
        mom_summary: String::new(),
        param_labels: labels,
    };
    (snap, draws)
}

fn build_empirical_histogram(draws: &[f64], lo: f64, hi: f64) -> Vec<[f64; 3]> {
    if draws.is_empty() || hi <= lo {
        return Vec::new();
    }
    let width = (hi - lo) / (HIST_BINS as f64);
    let mut counts = [0_usize; HIST_BINS];
    for &x in draws {
        if x >= lo && x <= hi {
            let bin = (((x - lo) / width).floor() as usize).min(HIST_BINS - 1);
            counts[bin] += 1;
        }
    }
    let denom = (draws.len() as f64) * width;
    (0..HIST_BINS)
        .map(|b| {
            let center = lo + (b as f64 + 0.5) * width;
            [center, (counts[b] as f64) / denom, width * 0.88]
        })
        .collect()
}

fn build_discrete_snapshot<D: DiscreteDistribution>(
    dist: &D,
    k_max: u64,
    p: f64,
    n: usize,
    rng: &mut OxidizeRng,
    labels: (&'static str, Option<&'static str>),
) -> (DistributionSnapshot, Vec<u64>) {
    let cap = k_max.clamp(1, 60);
    let mut discrete_bars = Vec::with_capacity((cap + 1) as usize);
    let mut density_curve = Vec::with_capacity((cap + 1) as usize);
    let mut cdf_curve = Vec::with_capacity(((cap + 1) * 2) as usize);
    for k in 0..=cap {
        let pmf = dist.pmf(k);
        let cdf = dist.cdf(k);
        discrete_bars.push((k, pmf));
        density_curve.push([k as f64, pmf]);
        cdf_curve.push([k as f64, cdf]);
        cdf_curve.push([k as f64 + 0.999, cdf]);
    }
    let draws: Vec<u64> = (0..n).map(|_| dist.sample(rng)).collect();
    let mut counts = vec![0_usize; (cap + 1) as usize];
    for &d in &draws {
        if d <= cap {
            counts[d as usize] += 1;
        }
    }
    let histogram_bars = counts
        .into_iter()
        .enumerate()
        .map(|(k, c)| [k as f64 + 0.22, (c as f64) / (n as f64), 0.34])
        .collect();

    let snap = DistributionSnapshot {
        density_curve,
        cdf_curve,
        histogram_bars,
        discrete_bars,
        mean: Some(dist.mean()),
        variance: Some(dist.variance()),
        quantile_x: dist.quantile(p).unwrap_or(0) as f64,
        mle_summary: String::new(),
        mom_summary: String::new(),
        param_labels: labels,
    };
    (snap, draws)
}

fn eval_normal(mu: f64, sigma: f64, p: f64, n: usize, rng: &mut OxidizeRng) -> DistributionSnapshot {
    let dist = NormalDist::new(mu, sigma).expect("clamped Normal parameters");
    let (mut snap, draws) =
        build_continuous_snapshot(&dist, p, n, rng, ("Mean mu", Some("Std Dev sigma")));
    snap.mle_summary = NormalDist::fit_mle(&draws)
        .map_or_else(|e| e.to_string(), |d| format!("mu={:.3}, sigma={:.3}", d.mu(), d.sigma()));
    snap.mom_summary = NormalDist::fit_mom(&draws)
        .map_or_else(|e| e.to_string(), |d| format!("mu={:.3}, sigma={:.3}", d.mu(), d.sigma()));
    snap
}

fn eval_gamma(shape: f64, rate: f64, p: f64, n: usize, rng: &mut OxidizeRng) -> DistributionSnapshot {
    let dist = GammaDist::new(shape, rate).expect("clamped Gamma parameters");
    let (mut snap, draws) =
        build_continuous_snapshot(&dist, p, n, rng, ("Shape alpha", Some("Rate beta")));
    snap.mle_summary = GammaDist::fit_mle(&draws).map_or_else(
        |e| e.to_string(),
        |d| format!("alpha={:.3}, beta={:.3}", d.shape(), d.rate()),
    );
    snap.mom_summary = GammaDist::fit_mom(&draws).map_or_else(
        |e| e.to_string(),
        |d| format!("alpha={:.3}, beta={:.3}", d.shape(), d.rate()),
    );
    snap
}

fn eval_beta(a: f64, b: f64, p: f64, n: usize, rng: &mut OxidizeRng) -> DistributionSnapshot {
    let dist = BetaDist::new(a, b).expect("clamped Beta parameters");
    let (mut snap, draws) =
        build_continuous_snapshot(&dist, p, n, rng, ("Shape alpha", Some("Shape beta")));
    snap.mle_summary = BetaDist::fit_mle(&draws).map_or_else(
        |e| e.to_string(),
        |d| format!("alpha={:.3}, beta={:.3}", d.alpha(), d.beta()),
    );
    snap.mom_summary = BetaDist::fit_mom(&draws).map_or_else(
        |e| e.to_string(),
        |d| format!("alpha={:.3}, beta={:.3}", d.alpha(), d.beta()),
    );
    snap
}

fn eval_weibull(k: f64, scale: f64, p: f64, n: usize, rng: &mut OxidizeRng) -> DistributionSnapshot {
    let dist = WeibullDist::new(k, scale).expect("clamped Weibull parameters");
    let (mut snap, draws) =
        build_continuous_snapshot(&dist, p, n, rng, ("Shape k", Some("Scale lambda")));
    snap.mle_summary = WeibullDist::fit_mle(&draws).map_or_else(
        |e| e.to_string(),
        |d| format!("k={:.3}, lambda={:.3}", d.shape(), d.scale()),
    );
    snap.mom_summary = WeibullDist::fit_mom(&draws).map_or_else(
        |e| e.to_string(),
        |d| format!("k={:.3}, lambda={:.3}", d.shape(), d.scale()),
    );
    snap
}

fn eval_student_t(df: f64, scale: f64, p: f64, n: usize, rng: &mut OxidizeRng) -> DistributionSnapshot {
    let dist = StudentTDist::new(df, 0.0, scale).expect("clamped Student-t parameters");
    let (mut snap, draws) =
        build_continuous_snapshot(&dist, p, n, rng, ("Degrees of Freedom nu", Some("Scale sigma")));
    snap.mle_summary = StudentTDist::fit_mle(&draws).map_or_else(
        |e| e.to_string(),
        |d| format!("nu={:.2}, mu={:.3}, s={:.3}", d.df(), d.location(), d.scale()),
    );
    snap.mom_summary = StudentTDist::fit_mom(&draws).map_or_else(
        |e| e.to_string(),
        |d| format!("nu={:.2}, mu={:.3}, s={:.3}", d.df(), d.location(), d.scale()),
    );
    snap
}

fn eval_lognormal(mu: f64, sigma: f64, p: f64, n: usize, rng: &mut OxidizeRng) -> DistributionSnapshot {
    let dist = LogNormalDist::new(mu, sigma).expect("clamped LogNormal parameters");
    let (mut snap, draws) =
        build_continuous_snapshot(&dist, p, n, rng, ("Log-Mean mu", Some("Log-Scale sigma")));
    snap.mle_summary = LogNormalDist::fit_mle(&draws).map_or_else(
        |e| e.to_string(),
        |d| format!("mu={:.3}, sigma={:.3}", d.mu(), d.sigma()),
    );
    snap.mom_summary = LogNormalDist::fit_mom(&draws).map_or_else(
        |e| e.to_string(),
        |d| format!("mu={:.3}, sigma={:.3}", d.mu(), d.sigma()),
    );
    snap
}

fn eval_exponential(rate: f64, p: f64, n: usize, rng: &mut OxidizeRng) -> DistributionSnapshot {
    let dist = ExponentialDist::new(rate).expect("clamped Exponential rate");
    let (mut snap, draws) = build_continuous_snapshot(&dist, p, n, rng, ("Rate lambda", None));
    snap.mle_summary = ExponentialDist::fit_mle(&draws)
        .map_or_else(|e| e.to_string(), |d| format!("lambda={:.3}", d.rate()));
    snap.mom_summary = ExponentialDist::fit_mom(&draws)
        .map_or_else(|e| e.to_string(), |d| format!("lambda={:.3}", d.rate()));
    snap
}

fn eval_chi_squared(df: f64, p: f64, n: usize, rng: &mut OxidizeRng) -> DistributionSnapshot {
    let dist = ChiSquaredDist::new(df).expect("clamped ChiSquared df");
    let (mut snap, draws) =
        build_continuous_snapshot(&dist, p, n, rng, ("Degrees of Freedom k", None));
    snap.mle_summary = ChiSquaredDist::fit_mle(&draws)
        .map_or_else(|e| e.to_string(), |d| format!("df={:.3}", d.df()));
    snap.mom_summary = ChiSquaredDist::fit_mom(&draws)
        .map_or_else(|e| e.to_string(), |d| format!("df={:.3}", d.df()));
    snap
}

fn eval_uniform(lo: f64, hi: f64, p: f64, n: usize, rng: &mut OxidizeRng) -> DistributionSnapshot {
    let dist = UniformDist::new(lo, hi).expect("clamped Uniform bounds");
    let (mut snap, draws) =
        build_continuous_snapshot(&dist, p, n, rng, ("Lower Bound a", Some("Width b - a")));
    snap.mle_summary = UniformDist::fit_mle(&draws).map_or_else(
        |e| e.to_string(),
        |d| format!("a={:.3}, b={:.3}", d.lower(), d.upper()),
    );
    snap.mom_summary = UniformDist::fit_mom(&draws).map_or_else(
        |e| e.to_string(),
        |d| format!("a={:.3}, b={:.3}", d.lower(), d.upper()),
    );
    snap
}

fn eval_poisson(lambda: f64, p: f64, n: usize, rng: &mut OxidizeRng) -> DistributionSnapshot {
    let dist = PoissonDist::new(lambda).expect("clamped Poisson rate");
    let k_max = dist.quantile(0.998).unwrap_or(20).max(8);
    let (mut snap, draws) = build_discrete_snapshot(&dist, k_max, p, n, rng, ("Rate lambda", None));
    snap.mle_summary = PoissonDist::fit_mle(&draws)
        .map_or_else(|e| e.to_string(), |d| format!("lambda={:.3}", d.lambda()));
    snap.mom_summary = PoissonDist::fit_mom(&draws)
        .map_or_else(|e| e.to_string(), |d| format!("lambda={:.3}", d.lambda()));
    snap
}

fn eval_binomial(trials: u64, prob: f64, p: f64, n: usize, rng: &mut OxidizeRng) -> DistributionSnapshot {
    let dist = BinomialDist::new(trials, prob).expect("clamped Binomial parameters");
    let (mut snap, draws) = build_discrete_snapshot(
        &dist,
        trials,
        p,
        n,
        rng,
        ("Trials n", Some("Success Prob p")),
    );
    snap.mle_summary = BinomialDist::fit_mle_known_n(&draws, trials).map_or_else(
        |e| e.to_string(),
        |d| format!("n={}, p={:.3}", d.n(), d.p()),
    );
    snap.mom_summary = BinomialDist::fit_mom(&draws).map_or_else(
        |e| e.to_string(),
        |d| format!("n={}, p={:.3}", d.n(), d.p()),
    );
    snap
}

fn eval_negbin(r: f64, prob: f64, p: f64, n: usize, rng: &mut OxidizeRng) -> DistributionSnapshot {
    let dist = NegativeBinomialDist::new(r, prob).expect("clamped NegativeBinomial");
    let k_max = dist.quantile(0.995).unwrap_or(25).max(10);
    let (mut snap, draws) =
        build_discrete_snapshot(&dist, k_max, p, n, rng, ("Dispersion r", Some("Success Prob p")));
    snap.mle_summary = NegativeBinomialDist::fit_mle(&draws).map_or_else(
        |e| e.to_string(),
        |d| format!("r={:.3}, p={:.3}", d.r(), d.p()),
    );
    snap.mom_summary = NegativeBinomialDist::fit_mom(&draws).map_or_else(
        |e| e.to_string(),
        |d| format!("r={:.3}, p={:.3}", d.r(), d.p()),
    );
    snap
}

fn eval_geometric(prob: f64, p: f64, n: usize, rng: &mut OxidizeRng) -> DistributionSnapshot {
    let dist = GeometricDist::new(prob).expect("clamped Geometric p");
    let k_max = dist.quantile(0.995).unwrap_or(20).max(8);
    let (mut snap, draws) =
        build_discrete_snapshot(&dist, k_max, p, n, rng, ("Success Prob p", None));
    snap.mle_summary = GeometricDist::fit_mle(&draws)
        .map_or_else(|e| e.to_string(), |d| format!("p={:.3}", d.p()));
    snap.mom_summary = GeometricDist::fit_mom(&draws)
        .map_or_else(|e| e.to_string(), |d| format!("p={:.3}", d.p()));
    snap
}

fn eval_bernoulli(prob: f64, p: f64, n: usize, rng: &mut OxidizeRng) -> DistributionSnapshot {
    let dist = BernoulliDist::new(prob).expect("clamped Bernoulli p");
    let (mut snap, draws) =
        build_discrete_snapshot(&dist, 1, p, n, rng, ("Success Prob p", None));
    snap.mle_summary = BernoulliDist::fit_mle(&draws)
        .map_or_else(|e| e.to_string(), |d| format!("p={:.3}", d.p()));
    snap.mom_summary = BernoulliDist::fit_mom(&draws)
        .map_or_else(|e| e.to_string(), |d| format!("p={:.3}", d.p()));
    snap
}

fn eval_zip(rho: f64, lambda: f64, p: f64, n: usize, rng: &mut OxidizeRng) -> DistributionSnapshot {
    let params = ZipParams::from_values(rho, lambda).expect("clamped ZIP parameters");
    let dist = ZipDistribution::new(params);
    let k_max = <ZipDistribution as DiscreteDistribution>::quantile(&dist, 0.995)
        .unwrap_or(18)
        .max(8);
    let (mut snap, draws) = build_discrete_snapshot(
        &dist,
        k_max,
        p,
        n,
        rng,
        ("Zero-Inflation rho", Some("Poisson Rate lambda")),
    );
    let counts: Vec<Count> = draws.iter().map(|&k| Count::new(k as u32)).collect();
    let mom_str = simple_zip_fit(&counts).map_or_else(
        |e| e.to_string(),
        |est| format!("rho={:.3}, lambda={:.3}", est.rho.value(), est.lambda.value()),
    );
    snap.mle_summary = format!("Unclamped MoM/EM seed: {mom_str}");
    snap.mom_summary = mom_str;
    snap
}

// [cite:clinical_trials_statistics]
