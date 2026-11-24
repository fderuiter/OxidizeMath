//! Interactive Volatility Drag, Solvency Barrier, Push Settlement, and Drawdown-Constrained Lab.
//!
//! Powered by [`BankrollEnsembleWorkspace`] and [`kelly_barrier_risk`] alongside preallocated
//! three-state push and Busseti-Ryu-Boyd drawdown-constrained trajectory buffers.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::CalloutManager;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::bootstrap::{sample_median_in_place, BootstrapWorkspace};
use math_explorer::pure_math::statistics::kelly::{
    drawdown_constrained_kelly, drawdown_constrained_push_kelly, expected_growth_rate,
    expected_value, kelly_fraction, push_expected_growth_rate, push_expected_value,
    push_kelly_fraction, BankrollFraction, DrawdownConstraint, Odds, PushOutcome, PushWager,
    UnitInterval,
};
use math_explorer::pure_math::statistics::ruin::{
    kelly_barrier_risk, wilson_interval_95, BankrollEnsembleWorkspace, DrawdownQuantiles,
    EnsembleSolvencyReport, KellyBarrierRisk, ProportionalPortfolioBet,
};
use oxidize_core::rng::{OxidizeRng, BASE_SEED};
use rand::Rng;

#[path = "bankroll_ensemble_ui.rs"]
mod ui;

/// Maximum number of sequential wagers preallocated for zero-allocation evaluation.
pub const MAX_ENSEMBLE_BETS: usize = 1000;
/// Number of Monte Carlo ensemble trajectories simulated per rerun.
pub const ENSEMBLE_PATH_COUNT: usize = 1000;
/// Number of nonparametric bootstrap resamples for terminal median confidence bounds.
pub const BOOTSTRAP_REPLICATES: usize = 256;
/// Number of preallocated trajectories used for push/drawdown median overlays.
const PUSH_PREVIEW_PATH_COUNT: usize = 128;
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

/// Active view mode inside [`super::bankroll_growth::BankrollGrowthTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BankrollLabMode {
    /// Classic Full / Half / Quarter / Drawdown-Constrained Kelly compounding trajectories.
    #[default]
    KellyStrategies,
    /// Interactive non-ergodic volatility-drag and drawdown barrier laboratory.
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

/// Preallocated simulator and UI controller for Kelly volatility-drag, barrier, and drawdown experiments.
pub struct BankrollEnsembleLab {
    /// Custom wager fraction `f` clamped to `0.0..=0.95`.
    pub custom_fraction: f64,
    /// Wealth floor ratio `B_floor / W_0` clamped to `0.05..=0.95` (default `0.50`).
    pub barrier_floor_ratio: f64,
    /// User's predict-before-run hypothesis selection.
    pub hypothesis: Option<VolatilityHypothesis>,
    /// Plot callout overlay manager for growth rate and ruin annotations.
    pub callout_manager: CalloutManager,
    /// Analytical expected wealth trajectory `E[W_t] = W_0 * (1 + f * EV)^t`.
    pub expected_points: Vec<PlotPoint>,
    /// Analytical asymptotic median trajectory $W_{\text{median}}(t) = W_0 e^{t g(f)}$.
    pub median_points: Vec<PlotPoint>,
    /// Empirical median trajectory across `ENSEMBLE_PATH_COUNT` stochastic paths.
    pub empirical_median_points: Vec<PlotPoint>,
    /// Empirical median trajectory under the Busseti-Ryu-Boyd drawdown-constrained fraction.
    pub drawdown_median_points: Vec<PlotPoint>,
    /// Horizontal drawdown floor line `[0, B_floor] .. [T, B_floor]`.
    pub barrier_points: Vec<PlotPoint>,
    workspace: BankrollEnsembleWorkspace,
    bootstrap_ws: BootstrapWorkspace,
    path_wealths: Vec<f64>,
    peak_wealths: Vec<f64>,
    max_drawdowns: Vec<f64>,
    barrier_hits: Vec<u8>,
    drawdown_path_wealths: Vec<f64>,
    scratch_sort: Vec<f64>,
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
        let workspace =
            BankrollEnsembleWorkspace::with_capacity(ENSEMBLE_PATH_COUNT, MAX_ENSEMBLE_BETS)
                .expect("positive ensemble path capacity");
        Self {
            custom_fraction: DEFAULT_CUSTOM_FRACTION,
            barrier_floor_ratio: DEFAULT_BARRIER_FLOOR_RATIO,
            hypothesis: None,
            callout_manager: CalloutManager::new(),
            expected_points: Vec::with_capacity(MAX_ENSEMBLE_BETS + 1),
            median_points: Vec::with_capacity(MAX_ENSEMBLE_BETS + 1),
            empirical_median_points: Vec::with_capacity(MAX_ENSEMBLE_BETS + 1),
            drawdown_median_points: Vec::with_capacity(MAX_ENSEMBLE_BETS + 1),
            barrier_points: Vec::with_capacity(2),
            workspace,
            bootstrap_ws: BootstrapWorkspace::new(ENSEMBLE_PATH_COUNT, BOOTSTRAP_REPLICATES)
                .expect("valid preallocated bootstrap capacity"),
            path_wealths: vec![0.0; PUSH_PREVIEW_PATH_COUNT],
            peak_wealths: vec![0.0; PUSH_PREVIEW_PATH_COUNT],
            max_drawdowns: vec![0.0; PUSH_PREVIEW_PATH_COUNT],
            barrier_hits: vec![0; PUSH_PREVIEW_PATH_COUNT],
            drawdown_path_wealths: vec![0.0; PUSH_PREVIEW_PATH_COUNT],
            scratch_sort: vec![0.0; PUSH_PREVIEW_PATH_COUNT],
            last_growth_rate: 0.0,
            last_f_star: None,
            last_drawdown_f: None,
            last_expected_terminal: 0.0,
            last_median_terminal: 0.0,
            last_empirical_median_ci_lower: 0.0,
            last_empirical_median_ci_upper: 0.0,
            last_drawdown_terminal: 0.0,
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

    /// Evaluates analytical curves, barrier bounds, and stochastic paths with zero heap allocations.
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

    fn eval_binary(
        &mut self,
        initial_bankroll: f64,
        p: &UnitInterval,
        o: &Odds,
        constraint: Option<&DrawdownConstraint>,
        num_bets: usize,
        seed_counter: u64,
    ) {
        self.clamp_fraction();
        let (steps, w0, f_clamped) = (
            num_bets.min(MAX_ENSEMBLE_BETS),
            initial_bankroll.max(1e-6),
            self.custom_fraction,
        );
        let Ok(frac) = BankrollFraction::new(f_clamped) else {
            return;
        };
        let (ev, g) = (expected_value(p, o), expected_growth_rate(p, o, &frac));
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
        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ seed_counter.wrapping_mul(0x9E37_79B9));
        self.last_solvency_report = ProportionalPortfolioBet::unit_weight(*p, *o, frac)
            .and_then(|bet| {
                self.workspace
                    .simulate_proportional(w0, barrier_floor, &bet, steps, &mut rng)
            })
            .ok();
        let g_rc = f_rc
            .and_then(|fr| BankrollFraction::new(fr).ok())
            .map(|bf| expected_growth_rate(p, o, &bf));
        self.populate_binary_series(w0, ev, g_rc, steps);
        self.update_terminal_median_bca(false, &mut rng);
    }

    fn populate_binary_series(&mut self, w0: f64, ev: f64, g_rc: Option<f64>, steps: usize) {
        let (f_clamped, g, barrier_floor) = (
            self.custom_fraction,
            self.last_growth_rate,
            w0 * self.barrier_floor_ratio,
        );
        self.expected_points.clear();
        self.median_points.clear();
        self.empirical_median_points.clear();
        self.drawdown_median_points.clear();
        self.barrier_points.clear();
        self.expected_points.push(PlotPoint::new(0.0, w0));
        self.median_points.push(PlotPoint::new(0.0, w0));
        self.drawdown_median_points.push(PlotPoint::new(0.0, w0));
        self.barrier_points.push(PlotPoint::new(0.0, barrier_floor));
        self.barrier_points
            .push(PlotPoint::new(steps as f64, barrier_floor));

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
            self.expected_points.push(PlotPoint::new(x, ew));
            self.median_points.push(PlotPoint::new(x, w_med));
            self.drawdown_median_points.push(PlotPoint::new(x, rc_med));
        }
        for (t, &emp_w) in self.workspace.median_trajectory().iter().enumerate() {
            self.empirical_median_points
                .push(PlotPoint::new(t as f64, emp_w));
        }
        self.update_terminal_readouts(w0);
    }

    /// Evaluates three-state push settlements and drawdown-constrained ensemble paths with zero allocations.
    pub fn evaluate_push_and_drawdown_in_place(
        &mut self,
        initial_bankroll: f64,
        wager: &PushWager,
        constraint: Option<&DrawdownConstraint>,
        num_bets: usize,
        seed_counter: u64,
    ) {
        if wager.p_push() <= 1e-12 {
            if let Ok(p) = UnitInterval::new(wager.p_win()) {
                self.eval_binary(
                    initial_bankroll,
                    &p,
                    &wager.odds(),
                    constraint,
                    num_bets,
                    seed_counter,
                );
                return;
            }
        }
        self.clamp_fraction();
        let (steps, w0) = (num_bets.min(MAX_ENSEMBLE_BETS), initial_bankroll.max(1e-6));
        let Ok(frac) = BankrollFraction::new(self.custom_fraction) else {
            return;
        };
        let f_star = push_kelly_fraction(wager).ok().map(|fk| fk.value());
        self.last_growth_rate = push_expected_growth_rate(wager, &frac);
        self.last_f_star = f_star;
        self.last_drawdown_f = constraint
            .and_then(|dc| drawdown_constrained_push_kelly(wager, dc).ok())
            .map(|fr| fr.value());
        self.is_ruin_zone = f_star.is_some_and(|fs| self.custom_fraction > 2.0 * fs);
        self.last_barrier_risk = None;
        self.run_push_series(w0, wager, push_expected_value(wager), steps, seed_counter);
    }

    fn run_push_series(
        &mut self,
        w0: f64,
        wager: &PushWager,
        ev: f64,
        steps: usize,
        seed_counter: u64,
    ) {
        let (f_clamped, g, rc_frac, barrier_floor) = (
            self.custom_fraction,
            self.last_growth_rate,
            self.last_drawdown_f.unwrap_or(0.0),
            w0 * self.barrier_floor_ratio,
        );
        self.reset_push_buffers(w0, barrier_floor, steps);
        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ seed_counter.wrapping_mul(0x9E37_79B9));
        let step_ev_factor = (1.0 + f_clamped * ev).max(0.0);
        let mut ew = w0;
        for t in 1..=steps {
            ew = (ew * step_ev_factor).max(0.0);
            let w_med = if g.is_finite() {
                (w0 * ((t as f64) * g).exp()).max(0.0)
            } else {
                0.0
            };
            let (emp_med, rc_med) =
                self.step_push_medians(&mut rng, wager, f_clamped, rc_frac, barrier_floor);
            let x = t as f64;
            self.expected_points.push(PlotPoint::new(x, ew));
            self.median_points.push(PlotPoint::new(x, w_med));
            self.empirical_median_points
                .push(PlotPoint::new(x, emp_med));
            self.drawdown_median_points.push(PlotPoint::new(x, rc_med));
        }
        self.last_solvency_report = self.build_push_solvency_report(steps);
        self.update_terminal_readouts(w0);
        self.update_terminal_median_bca(true, &mut rng);
    }

    fn update_terminal_readouts(&mut self, w0: f64) {
        self.last_expected_terminal = self.expected_points.last().map_or(w0, |pt| pt.y);
        self.last_median_terminal = self.median_points.last().map_or(w0, |pt| pt.y);
        self.last_drawdown_terminal = self.drawdown_median_points.last().map_or(w0, |pt| pt.y);
    }

    fn update_terminal_median_bca(&mut self, push_mode: bool, rng: &mut OxidizeRng) {
        let samples = if push_mode {
            &self.path_wealths[..]
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

    fn reset_push_buffers(&mut self, w0: f64, barrier_floor: f64, steps: usize) {
        self.expected_points.clear();
        self.median_points.clear();
        self.empirical_median_points.clear();
        self.drawdown_median_points.clear();
        self.barrier_points.clear();
        self.path_wealths.fill(w0);
        self.peak_wealths.fill(w0);
        self.max_drawdowns.fill(0.0);
        self.barrier_hits.fill(0);
        self.drawdown_path_wealths.fill(w0);

        self.expected_points.push(PlotPoint::new(0.0, w0));
        self.median_points.push(PlotPoint::new(0.0, w0));
        self.empirical_median_points.push(PlotPoint::new(0.0, w0));
        self.drawdown_median_points.push(PlotPoint::new(0.0, w0));
        self.barrier_points.push(PlotPoint::new(0.0, barrier_floor));
        self.barrier_points
            .push(PlotPoint::new(steps as f64, barrier_floor));
    }

    fn step_push_medians(
        &mut self,
        rng: &mut OxidizeRng,
        wager: &PushWager,
        f_custom: f64,
        f_rc: f64,
        barrier_floor: f64,
    ) -> (f64, f64) {
        let net_b = wager.odds().net_profit_multiplier();
        for idx in 0..PUSH_PREVIEW_PATH_COUNT {
            let outcome = wager.sample_outcome(rng.r#gen::<f64>());
            let w_new = settle_raw_push(self.path_wealths[idx], f_custom, net_b, outcome);
            self.path_wealths[idx] = w_new;
            if w_new > self.peak_wealths[idx] {
                self.peak_wealths[idx] = w_new;
            }
            let peak = self.peak_wealths[idx];
            if peak > 0.0 {
                let dd = (peak - w_new) / peak;
                if dd > self.max_drawdowns[idx] {
                    self.max_drawdowns[idx] = dd;
                }
            }
            if w_new <= barrier_floor {
                self.barrier_hits[idx] = 1;
            }
            self.drawdown_path_wealths[idx] =
                settle_raw_push(self.drawdown_path_wealths[idx], f_rc, net_b, outcome);
        }
        let emp_med = fast_even_median(&self.path_wealths, &mut self.scratch_sort);
        let rc_med = fast_even_median(&self.drawdown_path_wealths, &mut self.scratch_sort);
        (emp_med, rc_med)
    }

    fn build_push_solvency_report(&mut self, steps: usize) -> Option<EnsembleSolvencyReport> {
        let n = self.path_wealths.len();
        let ruin_count = self.barrier_hits.iter().filter(|&&h| h != 0).count();
        let wilson_95 = wilson_interval_95(ruin_count, n).ok()?;
        let terminal_mean_wealth = self.path_wealths.iter().sum::<f64>() / (n as f64);
        let terminal_median_wealth = fast_even_median(&self.path_wealths, &mut self.scratch_sort);
        self.scratch_sort.copy_from_slice(&self.max_drawdowns);
        self.scratch_sort.sort_unstable_by(f64::total_cmp);
        Some(EnsembleSolvencyReport {
            num_paths: n,
            num_steps: steps,
            ruin_count,
            ruin_probability: wilson_95.estimate,
            wilson_95,
            terminal_median_wealth,
            terminal_mean_wealth,
            drawdown_quantiles: DrawdownQuantiles {
                p50: sorted_quantile(&self.scratch_sort, 0.50),
                p90: sorted_quantile(&self.scratch_sort, 0.90),
                p95: sorted_quantile(&self.scratch_sort, 0.95),
                p99: sorted_quantile(&self.scratch_sort, 0.99),
            },
        })
    }
}

fn settle_raw_push(wealth: f64, frac: f64, net_b: f64, outcome: PushOutcome) -> f64 {
    let stake = wealth * frac;
    match outcome {
        PushOutcome::Win => (wealth + stake * net_b).max(0.0),
        PushOutcome::Push => wealth.max(0.0),
        PushOutcome::Loss => (wealth - stake).max(0.0),
    }
}

fn fast_even_median(values: &[f64], scratch: &mut [f64]) -> f64 {
    scratch.copy_from_slice(values);
    let mid = scratch.len() / 2;
    let (left, mid_val, _) = scratch.select_nth_unstable_by(mid, f64::total_cmp);
    let max_left = left.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    0.5 * (max_left + *mid_val)
}

fn sorted_quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q.clamp(0.0, 1.0) * ((sorted.len() - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - (lo as f64);
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

// [cite:clinical_trials_statistics]
