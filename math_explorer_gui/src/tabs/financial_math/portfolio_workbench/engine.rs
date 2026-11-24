#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{BsmPlotMetric, PortfolioWorkbenchState};
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::black_scholes::{
    black_scholes_merton, BlackScholesPair, BlackScholesParams, OptionGreeks,
};
use math_explorer::pure_math::statistics::copula::{
    Correlation, CorrelationMatrix, GaussianCopula, Probability,
};
use math_explorer::pure_math::statistics::kelly::{
    empirical_tail_risk, evaluate_portfolio_kelly_metrics, independent_kelly_weights,
    long_only_kelly_projected_gradient, parametric_lognormal_tail_risk,
    simulated_portfolio_tail_risk, unconstrained_covariance_kelly, PortfolioHorizonSpec,
    PortfolioKellySolution, TailRiskMetrics,
};
use math_explorer::pure_math::statistics::ou_process::{
    estimate_ou_params, EulerMaruyama, MomentumType, OuAnalyzer, OuParams, PerformanceStats,
    TimeStep,
};
use math_explorer::rng::OxidizeRng;

const FRONTIER_STEPS: usize = 100;
const COPULA_SWEEP_STEPS: usize = 48;
const BSM_SWEEP_STEPS: usize = 100;
const TAIL_MC_PATHS: usize = 8_192;

#[derive(Debug, Clone, Default)]
pub struct ImportedDatasetSummary {
    pub source_name: String,
    pub row_count: usize,
    pub column_names: Vec<String>,
    pub equal_weight_losses: Vec<f64>,
    pub ou_calibrated: Option<(f64, f64, f64)>,
}

#[derive(Debug, Clone, Default)]
pub struct KellyEngineCache {
    pub cov_kelly: Option<PortfolioKellySolution>,
    pub long_only_uncapped: Option<PortfolioKellySolution>,
    pub long_only_capped: Option<PortfolioKellySolution>,
    pub indep_kelly: Option<PortfolioKellySolution>,
    pub cov_sim_tail: Option<TailRiskMetrics>,
    pub cov_param_tail: Option<TailRiskMetrics>,
    pub indep_sim_tail: Option<TailRiskMetrics>,
    pub long_sim_tail: Option<TailRiskMetrics>,
    pub empirical_tail: Option<TailRiskMetrics>,
    pub cov_frontier: Vec<PlotPoint>,
    pub long_frontier: Vec<PlotPoint>,
    pub indep_frontier: Vec<PlotPoint>,
    pub zero_line: Vec<PlotPoint>,
    pub cov_marker: Vec<PlotPoint>,
    pub long_marker: Vec<PlotPoint>,
    pub indep_marker: Vec<PlotPoint>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct CopulaEngineCache {
    pub active_sgp_prob: f64,
    pub naive_sgp_prob: f64,
    pub active_crash_prob: f64,
    pub naive_crash_prob: f64,
    pub crash_multiplier: f64,
    pub sgp_curve: Vec<PlotPoint>,
    pub naive_sgp_line: Vec<PlotPoint>,
    pub crash_curve: Vec<PlotPoint>,
    pub naive_crash_line: Vec<PlotPoint>,
    pub active_crash_marker: Vec<PlotPoint>,
    pub active_sgp_marker: Vec<PlotPoint>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct OuEngineCache {
    pub forecast_stats: Option<PerformanceStats>,
    pub momentum: Option<MomentumType>,
    pub half_life: f64,
    pub p05_band: Vec<PlotPoint>,
    pub p50_band: Vec<PlotPoint>,
    pub p95_band: Vec<PlotPoint>,
    pub mean_curve: Vec<PlotPoint>,
    pub eq_line: Vec<PlotPoint>,
    pub sample_paths: Vec<(String, Vec<PlotPoint>)>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct BsmEngineCache {
    pub pair: Option<BlackScholesPair>,
    pub call_curve: Vec<PlotPoint>,
    pub put_curve: Vec<PlotPoint>,
    pub call_marker: Vec<PlotPoint>,
    pub put_marker: Vec<PlotPoint>,
    pub error: Option<String>,
}

pub fn recompute_kelly(
    state: &PortfolioWorkbenchState,
    imported: Option<&ImportedDatasetSummary>,
) -> KellyEngineCache {
    let mut out = KellyEngineCache {
        zero_line: vec![PlotPoint::new(0.0, 0.0), PlotPoint::new(2.5, 0.0)],
        ..Default::default()
    };
    let (_, cov) = match state.build_portfolio_matrices() {
        Ok(pair) => pair,
        Err(e) => {
            out.error = Some(e);
            return out;
        }
    };
    let mu: Vec<f64> = state.assets.iter().map(|a| a.mu).collect();
    let r = state.risk_free_rate;

    let cov_sol = match unconstrained_covariance_kelly(&mu, &cov, r) {
        Ok(sol) => sol,
        Err(e) => {
            out.error = Some(format!("Covariance Kelly failed: {e}"));
            return out;
        }
    };
    out.cov_kelly = Some(cov_sol);
    out.long_only_uncapped = long_only_kelly_projected_gradient(&mu, &cov, r, None).ok();
    out.long_only_capped =
        long_only_kelly_projected_gradient(&mu, &cov, r, Some(state.leverage_cap)).ok();
    out.indep_kelly = independent_kelly_weights(&mu, &cov, r).ok();

    populate_kelly_frontiers(&mut out, &mu, &cov, r);
    populate_tail_metrics(&mut out, state, &mu, &cov, imported);
    out
}

fn eval_scaled_frontier(
    weights: &[f64],
    mu: &[f64],
    cov: &math_explorer::nalgebra::DMatrix<f64>,
    r: f64,
) -> Vec<PlotPoint> {
    let mut scaled = vec![0.0; mu.len()];
    let mut pts = Vec::with_capacity(FRONTIER_STEPS + 1);
    for step in 0..=FRONTIER_STEPS {
        let scale = 2.5 * (step as f64 / FRONTIER_STEPS as f64);
        for (dst, &w) in scaled.iter_mut().zip(weights.iter()) {
            *dst = scale * w;
        }
        if let Ok(m) = evaluate_portfolio_kelly_metrics(&scaled, mu, cov, r) {
            pts.push(PlotPoint::new(scale, m.excess_log_growth));
        }
    }
    pts
}

fn populate_kelly_frontiers(
    out: &mut KellyEngineCache,
    mu: &[f64],
    cov: &math_explorer::nalgebra::DMatrix<f64>,
    r: f64,
) {
    if let Some(ref cov_sol) = out.cov_kelly {
        out.cov_frontier = eval_scaled_frontier(&cov_sol.weights, mu, cov, r);
        out.cov_marker
            .push(PlotPoint::new(1.0, cov_sol.metrics.excess_log_growth));
    }
    if let Some(ref lo) = out.long_only_capped {
        out.long_frontier = eval_scaled_frontier(&lo.weights, mu, cov, r);
        out.long_marker
            .push(PlotPoint::new(1.0, lo.metrics.excess_log_growth));
    }
    if let Some(ref ind) = out.indep_kelly {
        out.indep_frontier = eval_scaled_frontier(&ind.weights, mu, cov, r);
        out.indep_marker
            .push(PlotPoint::new(1.0, ind.metrics.excess_log_growth));
    }
}

fn populate_tail_metrics(
    out: &mut KellyEngineCache,
    state: &PortfolioWorkbenchState,
    mu: &[f64],
    cov: &math_explorer::nalgebra::DMatrix<f64>,
    imported: Option<&ImportedDatasetSummary>,
) {
    let Ok(spec) = PortfolioHorizonSpec::new(state.risk_free_rate, state.bankroll, state.horizon)
    else {
        return;
    };
    if let Some(ref cov_sol) = out.cov_kelly {
        out.cov_param_tail = parametric_lognormal_tail_risk(&cov_sol.weights, mu, cov, spec).ok();
        out.cov_sim_tail = simulated_portfolio_tail_risk(
            &cov_sol.weights,
            mu,
            cov,
            spec,
            TAIL_MC_PATHS,
            state.seed,
        )
        .ok();
    }
    if let Some(ref lo) = out.long_only_capped {
        out.long_sim_tail = simulated_portfolio_tail_risk(
            &lo.weights,
            mu,
            cov,
            spec,
            TAIL_MC_PATHS,
            state.seed ^ 0x55AA,
        )
        .ok();
    }
    if let Some(ref ind) = out.indep_kelly {
        out.indep_sim_tail = simulated_portfolio_tail_risk(
            &ind.weights,
            mu,
            cov,
            spec,
            TAIL_MC_PATHS,
            state.seed ^ 0xAA55,
        )
        .ok();
    }
    populate_empirical_tail(out, state.bankroll, imported);
}

fn populate_empirical_tail(
    out: &mut KellyEngineCache,
    bankroll: f64,
    imported: Option<&ImportedDatasetSummary>,
) {
    if let Some(imp) = imported {
        if !imp.equal_weight_losses.is_empty() {
            let scaled_losses: Vec<f64> = imp
                .equal_weight_losses
                .iter()
                .map(|&l| l * bankroll)
                .collect();
            out.empirical_tail = empirical_tail_risk(&scaled_losses).ok();
        }
    }
}

pub fn recompute_copula(state: &PortfolioWorkbenchState) -> CopulaEngineCache {
    let mut out = CopulaEngineCache::default();
    let legs = state.copula_legs.clamp(2, 12);
    let sgp_u: Vec<Probability> = (0..legs)
        .map(|i| {
            let u = state
                .assets
                .get(i % state.assets.len())
                .map_or(0.55, |a| a.marginal_u);
            Probability::new(u.clamp(0.001, 0.999)).unwrap()
        })
        .collect();
    let crash_p = match Probability::new(state.crash_quantile.clamp(0.001, 0.50)) {
        Ok(p) => p,
        Err(e) => {
            out.error = Some(format!("Invalid crash quantile: {e}"));
            return out;
        }
    };
    let crash_u = vec![crash_p; legs];
    out.naive_sgp_prob = sgp_u.iter().map(|p| p.value()).product();
    out.naive_crash_prob = crash_p.value().powi(legs as i32);

    let rho_min = PortfolioWorkbenchState::min_equicorrelation_for(legs);
    let rho_max = 0.95_f64;
    let active_rho = state.rho.clamp(rho_min, rho_max);
    eval_active_copula_point(&mut out, legs, active_rho, &sgp_u, &crash_u, state.seed);
    populate_copula_sweep(
        &mut out, legs, &sgp_u, &crash_u, rho_min, rho_max, state.seed,
    );
    out
}

fn eval_active_copula_point(
    out: &mut CopulaEngineCache,
    legs: usize,
    active_rho: f64,
    sgp_u: &[Probability],
    crash_u: &[Probability],
    seed: u64,
) {
    if let Ok(c) = Correlation::new(active_rho) {
        if let Ok(mat) = CorrelationMatrix::equicorrelated(legs, c) {
            let copula = GaussianCopula::new(mat);
            if let Ok(p) = copula.cdf_with_seed(sgp_u, seed, 512) {
                out.active_sgp_prob = p.value();
                out.active_sgp_marker
                    .push(PlotPoint::new(active_rho, p.value()));
            }
            if let Ok(p) = copula.cdf_with_seed(crash_u, seed, 512) {
                out.active_crash_prob = p.value();
                out.active_crash_marker
                    .push(PlotPoint::new(active_rho, p.value()));
            }
        }
    }
    out.crash_multiplier = if out.naive_crash_prob > 0.0 {
        out.active_crash_prob / out.naive_crash_prob
    } else {
        1.0
    };
}

fn populate_copula_sweep(
    out: &mut CopulaEngineCache,
    legs: usize,
    sgp_u: &[Probability],
    crash_u: &[Probability],
    rho_min: f64,
    rho_max: f64,
    seed: u64,
) {
    for step in 0..=COPULA_SWEEP_STEPS {
        let rho_val = rho_min + (rho_max - rho_min) * (step as f64 / COPULA_SWEEP_STEPS as f64);
        if let Ok(c) = Correlation::new(rho_val) {
            if let Ok(mat) = CorrelationMatrix::equicorrelated(legs, c) {
                let copula = GaussianCopula::new(mat);
                if let Ok(ps) = copula.cdf_with_seed(sgp_u, seed, 256) {
                    out.sgp_curve.push(PlotPoint::new(rho_val, ps.value()));
                }
                if let Ok(pc) = copula.cdf_with_seed(crash_u, seed, 256) {
                    out.crash_curve.push(PlotPoint::new(rho_val, pc.value()));
                }
            }
        }
    }
    out.naive_sgp_line = vec![
        PlotPoint::new(rho_min, out.naive_sgp_prob),
        PlotPoint::new(rho_max, out.naive_sgp_prob),
    ];
    out.naive_crash_line = vec![
        PlotPoint::new(rho_min, out.naive_crash_prob),
        PlotPoint::new(rho_max, out.naive_crash_prob),
    ];
}

pub fn recompute_ou(
    state: &PortfolioWorkbenchState,
    imported: Option<&ImportedDatasetSummary>,
) -> OuEngineCache {
    let mut out = OuEngineCache::default();
    let (mu, theta, sigma) = imported.and_then(|imp| imp.ou_calibrated).unwrap_or((
        state.ou_mu,
        state.ou_theta,
        state.ou_sigma,
    ));

    let params = match OuParams::from_values(mu, theta, sigma) {
        Ok(p) => p,
        Err(e) => {
            out.error = Some(format!("Invalid OU parameters: {e}"));
            return out;
        }
    };
    let dt = match TimeStep::new(state.ou_dt) {
        Ok(d) => d,
        Err(e) => {
            out.error = Some(format!("Invalid OU dt: {e}"));
            return out;
        }
    };
    let analyzer = OuAnalyzer::new(params, dt);
    out.momentum = Some(analyzer.momentum_type());
    out.half_life = std::f64::consts::LN_2 / theta.max(1e-6);

    let mut forecast_rng = OxidizeRng::new(state.seed);
    out.forecast_stats = Some(analyzer.monte_carlo_forecast(
        state.ou_x0,
        state.ou_horizon,
        state.ou_paths,
        Some(mu),
        &mut forecast_rng,
    ));

    populate_ou_cones(
        &mut out,
        params,
        dt,
        state.ou_x0,
        state.ou_horizon,
        state.ou_paths,
        state.seed,
    );
    out
}

fn populate_ou_cones(
    out: &mut OuEngineCache,
    params: OuParams,
    dt: TimeStep,
    x0: f64,
    horizon: f64,
    n_paths: usize,
    seed: u64,
) {
    let n_steps = ((horizon / dt.value()).ceil() as usize).clamp(4, 1_000);
    let solver = EulerMaruyama::new(params, dt);
    let mut rng = OxidizeRng::new(seed ^ 0x0D15_EA5E);
    let paths = solver.simulate_paths(x0, n_steps, n_paths, &mut rng);
    let mut slice_vals = vec![0.0; n_paths];
    let mu = params.mu.value();
    let theta = params.theta.value();

    for step in 0..=n_steps {
        let t = step as f64 * dt.value();
        for (p_idx, path) in paths.iter().enumerate() {
            slice_vals[p_idx] = path[step];
        }
        slice_vals.sort_unstable_by(f64::total_cmp);
        let p05 = slice_vals[((n_paths as f64 * 0.05) as usize).min(n_paths - 1)];
        let p50 = slice_vals[n_paths / 2];
        let p95 = slice_vals[((n_paths as f64 * 0.95) as usize).min(n_paths - 1)];
        let exact_mean = mu + (x0 - mu) * (-theta * t).exp();

        out.p05_band.push(PlotPoint::new(t, p05));
        out.p50_band.push(PlotPoint::new(t, p50));
        out.p95_band.push(PlotPoint::new(t, p95));
        out.mean_curve.push(PlotPoint::new(t, exact_mean));
    }
    let t_end = n_steps as f64 * dt.value();
    out.eq_line = vec![PlotPoint::new(0.0, mu), PlotPoint::new(t_end, mu)];
    for (idx, path) in paths.iter().take(3).enumerate() {
        let pts = path
            .iter()
            .enumerate()
            .map(|(s, &v)| PlotPoint::new(s as f64 * dt.value(), v))
            .collect();
        out.sample_paths
            .push((format!("Sample Path {}", idx + 1), pts));
    }
}

pub fn recompute_bsm(state: &PortfolioWorkbenchState) -> BsmEngineCache {
    let mut out = BsmEngineCache::default();
    let params = match BlackScholesParams::with_dividend_yield(
        state.bsm_spot,
        state.bsm_strike,
        state.risk_free_rate,
        state.bsm_dividend,
        state.bsm_vol,
        state.bsm_expiry,
    ) {
        Ok(p) => p,
        Err(e) => {
            out.error = Some(format!("Invalid Black-Scholes parameters: {e}"));
            return out;
        }
    };
    let pair = black_scholes_merton(&params);
    let extract = |g: &OptionGreeks| match state.bsm_metric {
        BsmPlotMetric::Prices => g.price,
        BsmPlotMetric::Delta => g.delta,
        BsmPlotMetric::Gamma => g.gamma,
        BsmPlotMetric::Vega => g.vega,
        BsmPlotMetric::Theta => g.theta,
        BsmPlotMetric::Rho => g.rho,
    };
    out.call_marker
        .push(PlotPoint::new(state.bsm_spot, extract(&pair.call)));
    out.put_marker
        .push(PlotPoint::new(state.bsm_spot, extract(&pair.put)));

    let s_min = (0.45 * state.bsm_strike).max(0.5);
    let s_max = 1.55 * state.bsm_strike;
    for i in 0..=BSM_SWEEP_STEPS {
        let s = s_min + (s_max - s_min) * (i as f64 / BSM_SWEEP_STEPS as f64);
        if let Ok(sp) = BlackScholesParams::with_dividend_yield(
            s,
            state.bsm_strike,
            state.risk_free_rate,
            state.bsm_dividend,
            state.bsm_vol,
            state.bsm_expiry,
        ) {
            let p = black_scholes_merton(&sp);
            out.call_curve.push(PlotPoint::new(s, extract(&p.call)));
            out.put_curve.push(PlotPoint::new(s, extract(&p.put)));
        }
    }
    out.pair = Some(pair);
    out
}

pub fn calibrate_ou_from_returns(returns: &[f64], dt: f64) -> Option<(f64, f64, f64)> {
    if returns.len() < 4 {
        return None;
    }
    let est = estimate_ou_params(returns, dt).ok()?;
    Some((est.mu.value(), est.theta.value(), est.sigma.value()))
}

// [cite:clinical_trials_statistics]
