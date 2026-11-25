//!Interactive Volatility Drag, Solvency Barrier, Push Settlement, and Non-Ergodic Ensemble Lab.
//!
//! Combines [`BankrollEnsembleWorkspace`] and [`kelly_barrier_risk`] solvency bounds with
//! three-state push settlements, Busseti-Ryu-Boyd drawdown-constrained medians, per-step
//! 5th/25th/50th/75th/95th wealth percentiles, closed polygon fan bands, underwater drawdown
//! curves, maximum underwater duration, and terminal 95% `VaR` and `CVaR` with zero heap
//! allocations inside simulation updates.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "bankroll_ensemble_push.rs"]
mod push_sim;
#[path = "bankroll_ensemble_ui.rs"]
mod ui;

use crate::framework::CalloutManager;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::bootstrap::{sample_median_in_place, BootstrapWorkspace};
use math_explorer::pure_math::statistics::kelly::{
    drawdown_constrained_kelly, expected_growth_rate, expected_value, kelly_fraction,
    BankrollFraction, DrawdownConstraint, Odds, UnitInterval,
};
use math_explorer::pure_math::statistics::risk::{
    drawdown_series_from_wealth, historical_cvar, historical_var, max_underwater_duration_wealth,
};
use math_explorer::pure_math::statistics::ruin::{
    kelly_barrier_risk, BankrollEnsembleWorkspace, EnsembleSolvencyReport, KellyBarrierRisk,
    ProportionalPortfolioBet,
};
use oxidize_core::rng::{OxidizeRng, BASE_SEED};
use rand::RngCore;

/// Maximum number of sequential wagers preallocated for zero-allocation evaluation.
pub const MAX_ENSEMBLE_BETS: usize = 1000;
/// Number of Monte Carlo ensemble trajectories simulated per rerun.
pub const ENSEMBLE_PATH_COUNT: usize = 1000;
/// Number of nonparametric bootstrap resamples for terminal median confidence bounds.
pub const BOOTSTRAP_REPLICATES: usize = 256;
/// Number of preallocated trajectories used for push/drawdown median overlays.
pub(super) const PUSH_PREVIEW_PATH_COUNT: usize = 128;
/// Upper bound on custom wager fraction `f` so `ln(1 - f)` remains finite.
pub const MAX_CUSTOM_FRACTION: f64 = 0.95;
/// Default custom wager fraction demonstrating overbetting volatility drag.
pub const DEFAULT_CUSTOM_FRACTION: f64 = 0.30;
/// Default wealth barrier floor ratio `B_floor / W_0` (50% drawdown floor).
pub const DEFAULT_BARRIER_FLOOR_RATIO: f64 = 0.50;
/// Minimum configurable wealth barrier floor ratio.
pub const MIN_BARRIER_FLOOR_RATIO: f64 = 0.05;
/// Maximum configurable wealth barrier floor ratio.
pub const MAX_BARRIER_FLOOR_RATIO: f64 = 0.95;

pub(super) const MIN_POSITIVE_WEALTH: f64 = 1e-300;
const INV_TWO_POW_53: f64 = 1.0 / 9_007_199_254_740_992.0;

/// Active view mode inside [`super::bankroll_growth::BankrollGrowthTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BankrollLabMode {
    /// Classic Full / Half / Quarter / Drawdown-Constrained Kelly compounding trajectories.
    #[default]
    KellyStrategies,
    /// Interactive non-ergodic volatility-drag, percentile fan-band, and drawdown barrier laboratory.
    VolatilityDrag,
}

/// User hypothesis choice made before inspecting the non-ergodic trajectory split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum VolatilityHypothesis {
    /// User predicts typical wealth compounds upward alongside expected wealth.
    TypicalGrows,
    /// User predicts volatility drag drives typical wealth down while `E[W_t]` grows.
    TypicalShrinksDiverges,
}

pub(super) fn sorted_percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = q.clamp(0.0, 1.0) * ((sorted.len() - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (pos - (lo as f64)) * (sorted[hi] - sorted[lo])
}

/// Preallocated simulator and UI controller for Kelly volatility-drag, barrier, and tail-risk experiments.
pub struct BankrollEnsembleLab {
    /// Custom allocation fraction `f` clamped to `0.0..=0.95`.
    pub custom_fraction: f64,
    /// Wealth floor ratio `B_floor / W_0` clamped to `0.05..=0.95` (default `0.50`).
    pub barrier_floor_ratio: f64,
    /// User's predict-before-run hypothesis selection.
    pub hypothesis: Option<VolatilityHypothesis>,
    /// Plot callout overlay manager for growth rate, ruin, and `VaR`/`CVaR` annotations.
    pub callout_manager: CalloutManager,
    /// Analytical expected wealth trajectory `E[W_t] = W_0 * (1 + f * EV)^t`.
    pub expected_points: Vec<PlotPoint>,
    /// Analytical asymptotic median trajectory $W_{\text{median}}(t) = W_0 e^{t g(f)}$.
    pub median_points: Vec<PlotPoint>,
    /// Empirical median trajectory across `ENSEMBLE_PATH_COUNT` stochastic paths.
    pub empirical_median_points: Vec<PlotPoint>,
    /// Trajectory under the Busseti-Ryu-Boyd drawdown-constrained fraction.
    pub drawdown_median_points: Vec<PlotPoint>,
    /// Horizontal drawdown floor line `[0, B_floor] .. [T, B_floor]`.
    pub barrier_points: Vec<PlotPoint>,
    /// Per-step 5th percentile wealth trajectory.
    pub p05_points: Vec<PlotPoint>,
    /// Per-step 25th percentile wealth trajectory.
    pub p25_points: Vec<PlotPoint>,
    /// Per-step 75th percentile wealth trajectory.
    pub p75_points: Vec<PlotPoint>,
    /// Per-step 95th percentile wealth trajectory.
    pub p95_points: Vec<PlotPoint>,
    /// Closed polygon boundary for the 5th–95th percentile fan band.
    pub fan_5_95_poly: Vec<PlotPoint>,
    /// Closed polygon boundary for the 25th–75th interquartile fan band.
    pub fan_25_75_poly: Vec<PlotPoint>,
    /// Median-trajectory underwater percentage curve (`-100 * dd_t`).
    pub underwater_points: Vec<PlotPoint>,
    workspace: BankrollEnsembleWorkspace,
    bootstrap_ws: BootstrapWorkspace,
    pub(super) path_wealths: Vec<f64>,
    pub(super) peak_wealths: Vec<f64>,
    pub(super) max_drawdowns: Vec<f64>,
    pub(super) barrier_hits: Vec<u8>,
    pub(super) drawdown_path_wealths: Vec<f64>,
    pub(super) scratch_sort: Vec<f64>,
    pub(super) terminal_returns: Vec<f64>,
    pub(super) median_wealth_buf: Vec<f64>,
    pub(super) drawdown_buf: Vec<f64>,
    /// Latest initial capital $W_0$ evaluated.
    pub last_initial_bankroll: f64,
    /// Latest logarithmic growth rate $g(f)$.
    pub last_growth_rate: f64,
    /// Latest optimal Kelly fraction $f^*$ when positive edge exists.
    pub last_f_star: Option<f64>,
    /// Latest Busseti-Ryu-Boyd risk-constrained Kelly fraction when active.
    pub last_drawdown_f: Option<f64>,
    /// Latest terminal expected wealth `E[W_T]`.
    pub last_expected_terminal: f64,
    /// Latest terminal median wealth $W_{\text{median}}(T)$.
    pub last_median_terminal: f64,
    /// Lower 95% BCa bootstrap confidence bound on terminal empirical median wealth.
    pub last_empirical_median_ci_lower: f64,
    /// Upper 95% BCa bootstrap confidence bound on terminal empirical median wealth.
    pub last_empirical_median_ci_upper: f64,
    /// Latest terminal empirical median wealth for the drawdown-constrained strategy.
    pub last_drawdown_terminal: f64,
    /// Peak-to-trough fractional maximum drawdown along the empirical median trajectory.
    pub last_max_drawdown: f64,
    /// Maximum consecutive underwater steps strictly below prior running peak.
    pub last_underwater_duration: usize,
    /// Terminal 95% Historical Value-at-Risk as a fractional capital loss.
    pub last_terminal_var_95: f64,
    /// Terminal 95% Historical Expected Shortfall (`CVaR`) as a fractional capital loss.
    pub last_terminal_cvar_95: f64,
    /// Whether `f > 2 * f*` (or `g(f) < 0` despite positive edge), flagging asymptotic decay.
    pub is_ruin_zone: bool,
    /// Analytical diffusion and log-return Lundberg barrier bounds for `barrier_floor_ratio`.
    pub last_barrier_risk: Option<KellyBarrierRisk>,
    /// Finite-horizon ensemble barrier report with 95% Wilson CI and max-drawdown quantiles.
    pub last_solvency_report: Option<EnsembleSolvencyReport>,
}

impl Default for BankrollEnsembleLab {
    fn default() -> Self {
        Self::new()
    }
}

impl BankrollEnsembleLab {
    /// Constructs a new ensemble lab backed by a preallocated [`BankrollEnsembleWorkspace`].
    pub fn new() -> Self {
        let cap = MAX_ENSEMBLE_BETS + 1;
        let workspace = BankrollEnsembleWorkspace::with_capacity(ENSEMBLE_PATH_COUNT, 0)
            .expect("positive ensemble path capacity");
        Self {
            custom_fraction: DEFAULT_CUSTOM_FRACTION,
            barrier_floor_ratio: DEFAULT_BARRIER_FLOOR_RATIO,
            hypothesis: None,
            callout_manager: CalloutManager::new(),
            expected_points: Vec::with_capacity(cap),
            median_points: Vec::with_capacity(cap),
            empirical_median_points: Vec::with_capacity(cap),
            drawdown_median_points: Vec::with_capacity(cap),
            barrier_points: Vec::with_capacity(2),
            p05_points: Vec::with_capacity(cap),
            p25_points: Vec::with_capacity(cap),
            p75_points: Vec::with_capacity(cap),
            p95_points: Vec::with_capacity(cap),
            fan_5_95_poly: Vec::with_capacity(2 * cap),
            fan_25_75_poly: Vec::with_capacity(2 * cap),
            underwater_points: Vec::with_capacity(cap),
            workspace,
            bootstrap_ws: BootstrapWorkspace::new(ENSEMBLE_PATH_COUNT, BOOTSTRAP_REPLICATES)
                .expect("valid preallocated bootstrap capacity"),
            path_wealths: vec![0.0; ENSEMBLE_PATH_COUNT],
            peak_wealths: vec![0.0; PUSH_PREVIEW_PATH_COUNT],
            max_drawdowns: vec![0.0; PUSH_PREVIEW_PATH_COUNT],
            barrier_hits: vec![0; PUSH_PREVIEW_PATH_COUNT],
            drawdown_path_wealths: vec![0.0; PUSH_PREVIEW_PATH_COUNT],
            scratch_sort: vec![0.0; ENSEMBLE_PATH_COUNT],
            terminal_returns: vec![0.0; ENSEMBLE_PATH_COUNT],
            median_wealth_buf: vec![1.0; cap],
            drawdown_buf: vec![0.0; cap],
            last_initial_bankroll: 100.0,
            last_growth_rate: 0.0,
            last_f_star: None,
            last_drawdown_f: None,
            last_expected_terminal: 0.0,
            last_median_terminal: 0.0,
            last_empirical_median_ci_lower: 0.0,
            last_empirical_median_ci_upper: 0.0,
            last_drawdown_terminal: 0.0,
            last_max_drawdown: 0.0,
            last_underwater_duration: 0,
            last_terminal_var_95: 0.0,
            last_terminal_cvar_95: 0.0,
            is_ruin_zone: false,
            last_barrier_risk: None,
            last_solvency_report: None,
        }
    }

    /// Resets custom fraction, barrier floor, hypothesis state, and callouts to defaults.
    pub fn reset_defaults(&mut self) {
        self.custom_fraction = DEFAULT_CUSTOM_FRACTION;
        self.barrier_floor_ratio = DEFAULT_BARRIER_FLOOR_RATIO;
        self.hypothesis = None;
        self.callout_manager.clear_auto();
    }

    /// Clamps `custom_fraction` and `barrier_floor_ratio` into safe finite intervals.
    pub fn clamp_fraction(&mut self) {
        self.custom_fraction = if self.custom_fraction.is_finite() {
            self.custom_fraction.clamp(0.0, MAX_CUSTOM_FRACTION)
        } else {
            DEFAULT_CUSTOM_FRACTION
        };
        self.barrier_floor_ratio = if self.barrier_floor_ratio.is_finite() {
            self.barrier_floor_ratio
                .clamp(MIN_BARRIER_FLOOR_RATIO, MAX_BARRIER_FLOOR_RATIO)
        } else {
            DEFAULT_BARRIER_FLOOR_RATIO
        };
    }

    pub(super) fn reset_step_zero(&mut self, w0: f64, barrier_floor: f64, steps: usize) {
        self.expected_points.clear();
        self.median_points.clear();
        self.empirical_median_points.clear();
        self.drawdown_median_points.clear();
        self.barrier_points.clear();
        self.p05_points.clear();
        self.p25_points.clear();
        self.p75_points.clear();
        self.p95_points.clear();
        self.fan_5_95_poly.clear();
        self.fan_25_75_poly.clear();
        self.underwater_points.clear();
        self.path_wealths.fill(w0);

        let origin = PlotPoint::new(0.0, w0);
        self.expected_points.push(origin);
        self.median_points.push(origin);
        self.empirical_median_points.push(origin);
        self.drawdown_median_points.push(origin);
        self.p05_points.push(origin);
        self.p25_points.push(origin);
        self.p75_points.push(origin);
        self.p95_points.push(origin);
        self.barrier_points.push(PlotPoint::new(0.0, barrier_floor));
        self.barrier_points
            .push(PlotPoint::new(steps as f64, barrier_floor));
        self.median_wealth_buf[0] = w0.max(MIN_POSITIVE_WEALTH);
    }

    /// Evaluates analytical curves, barrier bounds, per-step wealth quantiles, underwater
    /// drawdowns, and terminal `VaR`/`CVaR` with zero heap allocations.
    pub fn evaluate_in_place(
        &mut self,
        initial_bankroll: f64,
        p: &UnitInterval,
        o: &Odds,
        num_bets: usize,
        seed_counter: u64,
    ) {
        let dc = DrawdownConstraint::new(0.70, 0.10).ok();
        self.eval_binary(initial_bankroll, p, o, dc.as_ref(), num_bets, seed_counter);
    }

    pub(super) fn eval_binary(
        &mut self,
        initial_bankroll: f64,
        p: &UnitInterval,
        o: &Odds,
        constraint: Option<&DrawdownConstraint>,
        num_bets: usize,
        seed_counter: u64,
    ) {
        self.clamp_fraction();
        let steps = num_bets.min(MAX_ENSEMBLE_BETS);
        let w0 = initial_bankroll.max(1e-6);
        self.last_initial_bankroll = w0;
        let f_clamped = self.custom_fraction;
        let Ok(frac) = BankrollFraction::new(f_clamped) else {
            return;
        };
        let g = expected_growth_rate(p, o, &frac);
        let f_star = kelly_fraction(p, o).ok().map(|fk| fk.value());
        let f_rc = constraint
            .and_then(|dc| drawdown_constrained_kelly(p, o, dc).ok())
            .map(|fr| fr.value());
        self.last_growth_rate = g;
        self.last_f_star = f_star;
        self.last_drawdown_f = f_rc;
        self.is_ruin_zone = f_star.is_some_and(|fs| f_clamped > 2.0 * fs);
        self.last_barrier_risk = kelly_barrier_risk(p, o, &frac, self.barrier_floor_ratio).ok();

        let barrier_floor = w0 * self.barrier_floor_ratio;
        let mut ws_rng = OxidizeRng::new(BASE_SEED).with_stream(seed_counter);
        self.last_solvency_report = ProportionalPortfolioBet::unit_weight(*p, *o, frac)
            .and_then(|bet| {
                self.workspace
                    .simulate_proportional(w0, barrier_floor, &bet, steps, &mut ws_rng)
            })
            .ok();
        let g_rc = f_rc
            .and_then(|fr| BankrollFraction::new(fr).ok())
            .map(|bf| expected_growth_rate(p, o, &bf));
        self.reset_step_zero(w0, barrier_floor, steps);
        self.populate_quantile_series(w0, p, o, g_rc, steps, seed_counter);
        self.rebuild_fan_polygons_and_tail_risk(steps, w0, ENSEMBLE_PATH_COUNT);
        self.update_terminal_readouts(w0);
        self.update_terminal_median_bca(false, &mut ws_rng);
    }

    fn populate_quantile_series(
        &mut self,
        w0: f64,
        p: &UnitInterval,
        o: &Odds,
        g_rc: Option<f64>,
        steps: usize,
        seed_counter: u64,
    ) {
        let (win_prob, net_b, ev) = (p.value(), o.net_profit_multiplier(), expected_value(p, o));
        let f_clamped = self.custom_fraction;
        let g = self.last_growth_rate;
        let mut rng = OxidizeRng::new(BASE_SEED).with_stream(seed_counter);
        let win_mult = 1.0 + f_clamped * net_b;
        let loss_mult = (1.0 - f_clamped).max(0.0);
        let step_ev_factor = (1.0 + f_clamped * ev).max(0.0);
        let mut ew = w0;

        for t in 1..=steps {
            ew = (ew * step_ev_factor).max(0.0);
            let x = t as f64;
            let w_med = if g.is_finite() {
                (w0 * (x * g).exp()).max(0.0)
            } else {
                0.0
            };
            let rc_med = g_rc
                .filter(|gr| gr.is_finite())
                .map_or(w0, |gr| (w0 * (x * gr).exp()).max(0.0));
            let (p05, p25, emp_med, p75, p95) =
                self.step_ensemble_quantiles(&mut rng, win_prob, win_mult, loss_mult);
            self.expected_points.push(PlotPoint::new(x, ew));
            self.median_points.push(PlotPoint::new(x, w_med));
            self.empirical_median_points
                .push(PlotPoint::new(x, emp_med));
            self.drawdown_median_points.push(PlotPoint::new(x, rc_med));
            self.p05_points.push(PlotPoint::new(x, p05));
            self.p25_points.push(PlotPoint::new(x, p25));
            self.p75_points.push(PlotPoint::new(x, p75));
            self.p95_points.push(PlotPoint::new(x, p95));
            self.median_wealth_buf[t] = emp_med.max(MIN_POSITIVE_WEALTH);
        }
    }

    fn step_ensemble_quantiles(
        &mut self,
        rng: &mut OxidizeRng,
        win_prob: f64,
        win_mult: f64,
        loss_mult: f64,
    ) -> (f64, f64, f64, f64, f64) {
        for w in &mut self.path_wealths {
            let u = ((rng.next_u64() >> 11) as f64) * INV_TWO_POW_53;
            *w = (*w * if u < win_prob { win_mult } else { loss_mult }).max(0.0);
        }
        self.scratch_sort.copy_from_slice(&self.path_wealths);
        self.scratch_sort.sort_unstable_by(f64::total_cmp);
        let mid = self.scratch_sort.len() / 2;
        let emp_med = 0.5 * (self.scratch_sort[mid - 1] + self.scratch_sort[mid]);
        let p05 = sorted_percentile(&self.scratch_sort, 0.05);
        let p25 = sorted_percentile(&self.scratch_sort, 0.25);
        let p75 = sorted_percentile(&self.scratch_sort, 0.75);
        let p95 = sorted_percentile(&self.scratch_sort, 0.95);
        (p05, p25, emp_med, p75, p95)
    }

    pub(super) fn rebuild_fan_polygons_and_tail_risk(
        &mut self,
        steps: usize,
        w0: f64,
        path_count: usize,
    ) {
        for &pt in &self.p95_points {
            self.fan_5_95_poly.push(pt);
        }
        for &pt in self.p05_points.iter().rev() {
            self.fan_5_95_poly.push(pt);
        }
        for &pt in &self.p75_points {
            self.fan_25_75_poly.push(pt);
        }
        for &pt in self.p25_points.iter().rev() {
            self.fan_25_75_poly.push(pt);
        }

        let len = steps + 1;
        let wealth_slice = &self.median_wealth_buf[..len];
        let dd_slice = &mut self.drawdown_buf[..len];
        self.last_max_drawdown = drawdown_series_from_wealth(wealth_slice, dd_slice).unwrap_or(0.0);
        self.last_underwater_duration = max_underwater_duration_wealth(wealth_slice).unwrap_or(0);
        for (idx, &dd) in dd_slice.iter().enumerate() {
            self.underwater_points
                .push(PlotPoint::new(idx as f64, -100.0 * dd));
        }

        if w0 > 0.0 && path_count > 0 {
            let ret_slice = &mut self.terminal_returns[..path_count];
            let scratch_slice = &mut self.scratch_sort[..path_count];
            for (dst, &w_end) in ret_slice.iter_mut().zip(&self.path_wealths[..path_count]) {
                *dst = (w_end - w0) / w0;
            }
            self.last_terminal_var_95 =
                historical_var(ret_slice, 0.95, scratch_slice).unwrap_or(0.0);
            self.last_terminal_cvar_95 =
                historical_cvar(ret_slice, 0.95, scratch_slice).unwrap_or(0.0);
        } else {
            self.last_terminal_var_95 = 0.0;
            self.last_terminal_cvar_95 = 0.0;
        }
    }

    pub(super) fn update_terminal_readouts(&mut self, w0: f64) {
        self.last_expected_terminal = self.expected_points.last().map_or(w0, |pt| pt.y);
        self.last_median_terminal = self.median_points.last().map_or(w0, |pt| pt.y);
        self.last_drawdown_terminal = self.drawdown_median_points.last().map_or(w0, |pt| pt.y);
    }
    pub(super) fn update_terminal_median_bca(&mut self, push_mode: bool, rng: &mut OxidizeRng) {
        let samples = if push_mode {
            &self.path_wealths[..PUSH_PREVIEW_PATH_COUNT]
        } else {
            self.workspace.terminal_wealths()
        };
        if let Ok(ci) = self.bootstrap_ws.bca_ci_with_scratch(
            samples,
            BOOTSTRAP_REPLICATES,
            0.95,
            rng,
            sample_median_in_place,
        ) {
            self.last_empirical_median_ci_lower = ci.ci_lower;
            self.last_empirical_median_ci_upper = ci.ci_upper;
        } else {
            self.last_empirical_median_ci_lower = self.last_median_terminal;
            self.last_empirical_median_ci_upper = self.last_median_terminal;
        }
    }
}

// [cite:clinical_trials_statistics]
