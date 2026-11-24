//! Interactive Volatility Drag, Solvency Barrier, Push Settlement, and Drawdown-Constrained Lab.
//!
//! Powered by [`BankrollEnsembleWorkspace`] and [`kelly_barrier_risk`] alongside preallocated
//! three-state push, Busseti-Ryu-Boyd drawdown-constrained trajectory buffers, and
//! `#![no_std]` [`math_explorer::prob`] nonparametric Percentile / Efron BCa bootstrap bands.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::CalloutManager;
use egui_plot::PlotPoint;
use math_explorer::prob::{bootstrap::bca_workspace_floats, SolverWorkspace};
use math_explorer::pure_math::statistics::kelly::{
    drawdown_constrained_kelly, expected_growth_rate, expected_value, kelly_fraction,
    BankrollFraction, DrawdownConstraint, Odds, UnitInterval,
};
use math_explorer::pure_math::statistics::ruin::{
    kelly_barrier_risk, BankrollEnsembleWorkspace, BootstrapInterval, BootstrapMethod,
    EnsembleSolvencyReport, KellyBarrierRisk, ProportionalPortfolioBet,
};
use oxidize_core::rng::{OxidizeRng, BASE_SEED};

#[path = "bankroll_ensemble_push.rs"]
mod push;
#[path = "bankroll_ensemble_ui.rs"]
mod ui;

/// Maximum number of sequential wagers preallocated for zero-allocation evaluation.
pub const MAX_ENSEMBLE_BETS: usize = 1000;
/// Number of Monte Carlo ensemble trajectories simulated per rerun.
pub const ENSEMBLE_PATH_COUNT: usize = 1000;
/// Bootstrap resamples evaluated per trajectory knot for interactive frame rates.
pub const ENSEMBLE_BOOT_RESAMPLES: usize = 128;
/// Maximum bootstrap knot evaluations along `1..=steps` (`<= 40` knots for 60+ FPS).
pub const MAX_BOOTSTRAP_KNOTS: usize = 40;
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
    /// Lower 95% nonparametric bootstrap confidence band on the ensemble median path.
    pub median_ci_lower_points: Vec<PlotPoint>,
    /// Upper 95% nonparametric bootstrap confidence band on the ensemble median path.
    pub median_ci_upper_points: Vec<PlotPoint>,
    /// Trajectory under the Busseti-Ryu-Boyd drawdown-constrained fraction.
    pub drawdown_median_points: Vec<PlotPoint>,
    /// Horizontal drawdown floor line `[0, B_floor] .. [T, B_floor]`.
    pub barrier_points: Vec<PlotPoint>,
    /// Selected nonparametric bootstrap estimator (`Percentile` or `BCa`).
    pub bootstrap_method: BootstrapMethod,
    workspace: BankrollEnsembleWorkspace,
    boot_floats: Vec<f64>,
    boot_indices: Vec<usize>,
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
    /// Latest terminal 95% bootstrap confidence interval on the empirical median wealth.
    pub last_terminal_median_ci: Option<BootstrapInterval>,
    /// Latest terminal median wealth for the drawdown-constrained strategy.
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

pub(super) fn interp_band_point(prev_y: f64, curr_y: f64, alpha: f64) -> f64 {
    if prev_y > 1e-12 && curr_y > 1e-12 {
        (prev_y.ln() * (1.0 - alpha) + curr_y.ln() * alpha).exp()
    } else {
        prev_y * (1.0 - alpha) + curr_y * alpha
    }
}

impl BankrollEnsembleLab {
    /// Constructs a new ensemble lab backed by preallocated [`BankrollEnsembleWorkspace`], push, and bootstrap slices.
    pub fn new() -> Self {
        let workspace =
            BankrollEnsembleWorkspace::with_capacity(ENSEMBLE_PATH_COUNT, MAX_ENSEMBLE_BETS)
                .expect("positive ensemble path capacity");
        let n_boot_floats = bca_workspace_floats(ENSEMBLE_PATH_COUNT, ENSEMBLE_BOOT_RESAMPLES);
        Self {
            custom_fraction: DEFAULT_CUSTOM_FRACTION,
            barrier_floor_ratio: DEFAULT_BARRIER_FLOOR_RATIO,
            hypothesis: None,
            callout_manager: CalloutManager::new(),
            expected_points: Vec::with_capacity(MAX_ENSEMBLE_BETS + 1),
            median_points: Vec::with_capacity(MAX_ENSEMBLE_BETS + 1),
            empirical_median_points: Vec::with_capacity(MAX_ENSEMBLE_BETS + 1),
            median_ci_lower_points: Vec::with_capacity(MAX_ENSEMBLE_BETS + 1),
            median_ci_upper_points: Vec::with_capacity(MAX_ENSEMBLE_BETS + 1),
            drawdown_median_points: Vec::with_capacity(MAX_ENSEMBLE_BETS + 1),
            barrier_points: Vec::with_capacity(2),
            bootstrap_method: BootstrapMethod::Percentile,
            workspace,
            boot_floats: vec![0.0_f64; n_boot_floats],
            boot_indices: vec![0_usize; ENSEMBLE_PATH_COUNT],
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
            last_terminal_median_ci: None,
            last_drawdown_terminal: 0.0,
            is_ruin_zone: false,
            last_barrier_risk: None,
            last_solvency_report: None,
        }
    }

    /// Resets custom fraction, barrier floor, bootstrap method, hypothesis state, and callouts to defaults.
    pub fn reset_defaults(&mut self) {
        self.custom_fraction = DEFAULT_CUSTOM_FRACTION;
        self.barrier_floor_ratio = DEFAULT_BARRIER_FLOOR_RATIO;
        self.bootstrap_method = BootstrapMethod::Percentile;
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

    /// Evaluates analytical curves, barrier bounds, stochastic paths, and bootstrap median bands with zero heap allocations.
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
        let mut boot_rng = OxidizeRng::from_seed(
            BASE_SEED ^ 0xB007_57AB ^ seed_counter.wrapping_mul(0x517C_C1B7_2722_0A95),
        );
        self.last_solvency_report = ProportionalPortfolioBet::unit_weight(*p, *o, frac)
            .and_then(|bet| {
                self.simulate_and_bootstrap_band(
                    w0,
                    barrier_floor,
                    &bet,
                    steps,
                    &mut rng,
                    &mut boot_rng,
                )
            })
            .ok();
        let g_rc = f_rc
            .and_then(|fr| BankrollFraction::new(fr).ok())
            .map(|bf| expected_growth_rate(p, o, &bf));
        self.populate_binary_series(w0, ev, g_rc, steps);
    }

    fn simulate_and_bootstrap_band(
        &mut self,
        w0: f64,
        barrier_floor: f64,
        bet: &ProportionalPortfolioBet,
        steps: usize,
        rng: &mut OxidizeRng,
        boot_rng: &mut OxidizeRng,
    ) -> Result<EnsembleSolvencyReport, math_explorer::error::KellyError> {
        self.workspace.reset(w0);
        self.median_ci_lower_points.clear();
        self.median_ci_upper_points.clear();
        self.median_ci_lower_points.push(PlotPoint::new(0.0, w0));
        self.median_ci_upper_points.push(PlotPoint::new(0.0, w0));
        self.last_terminal_median_ci = None;

        let stride = (steps / MAX_BOOTSTRAP_KNOTS).max(1);
        let (mut prev_t, mut prev_lo, mut prev_hi) = (0usize, w0, w0);
        for t in 1..=steps {
            let emp_med = self
                .workspace
                .step_proportional_once(bet, barrier_floor, rng);
            if t % stride == 0 || t == steps {
                let mut ws = SolverWorkspace::new(&mut self.boot_floats, &mut self.boot_indices);
                let ci = self
                    .workspace
                    .bootstrap_median_wealth(
                        self.bootstrap_method,
                        ENSEMBLE_BOOT_RESAMPLES,
                        0.95,
                        boot_rng,
                        &mut ws,
                    )
                    .unwrap_or(BootstrapInterval {
                        point_estimate: emp_med,
                        lower: emp_med,
                        upper: emp_med,
                        std_error: 0.0,
                        confidence: 0.95,
                        bias_z0: 0.0,
                        acceleration: 0.0,
                    });
                let span = (t - prev_t) as f64;
                for s in (prev_t + 1)..=t {
                    let alpha = ((s - prev_t) as f64) / span;
                    let lo = interp_band_point(prev_lo, ci.lower.max(0.0), alpha);
                    let hi = interp_band_point(prev_hi, ci.upper.max(lo), alpha);
                    self.median_ci_lower_points
                        .push(PlotPoint::new(s as f64, lo));
                    self.median_ci_upper_points
                        .push(PlotPoint::new(s as f64, hi));
                }
                prev_t = t;
                prev_lo = ci.lower.max(0.0);
                prev_hi = ci.upper.max(prev_lo);
                if t == steps {
                    self.last_terminal_median_ci = Some(ci);
                }
            }
        }
        self.workspace.current_solvency_report(steps)
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

    pub(super) fn update_terminal_readouts(&mut self, w0: f64) {
        self.last_expected_terminal = self.expected_points.last().map_or(w0, |pt| pt.y);
        self.last_median_terminal = self.median_points.last().map_or(w0, |pt| pt.y);
        self.last_drawdown_terminal = self.drawdown_median_points.last().map_or(w0, |pt| pt.y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use math_explorer::pure_math::statistics::kelly::PushWager;

    #[test]
    fn test_bootstrap_median_bands_deterministic_and_zero_alloc() {
        let mut lab_a = BankrollEnsembleLab::new();
        let mut lab_b = BankrollEnsembleLab::new();
        let p = UnitInterval::new(0.55).unwrap();
        let o = Odds::from_net_odds(1.0).unwrap();

        lab_a.evaluate_in_place(100.0, &p, &o, 80, 0);
        lab_b.evaluate_in_place(100.0, &p, &o, 80, 0);
        {
            let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            verified_engine::engine::VerifiedEngine::run_verified(|| {
                lab_a.evaluate_in_place(100.0, &p, &o, 80, 0);
                lab_b.evaluate_in_place(100.0, &p, &o, 80, 0);
            });
        }
        assert_eq!(lab_a.median_ci_lower_points.len(), 81);
        assert_eq!(lab_a.median_ci_upper_points.len(), 81);
        assert_eq!(lab_a.drawdown_median_points.len(), 81);
        assert_eq!(lab_a.last_terminal_median_ci, lab_b.last_terminal_median_ci);
        let term_ci = lab_a.last_terminal_median_ci.unwrap();
        let emp_term = lab_a.empirical_median_points.last().unwrap().y;
        assert!(term_ci.lower <= emp_term + 1e-9 && emp_term <= term_ci.upper + 1e-9);

        lab_a.bootstrap_method = BootstrapMethod::BCa;
        {
            let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            verified_engine::engine::VerifiedEngine::run_verified(|| {
                lab_a.evaluate_in_place(100.0, &p, &o, 80, 0);
            });
        }
        let bca_ci = lab_a.last_terminal_median_ci.unwrap();
        assert!(bca_ci.lower <= bca_ci.upper);

        let pw = PushWager::from_win_and_push(0.50, 0.10, o).unwrap();
        let dc = DrawdownConstraint::new(0.70, 0.10).unwrap();
        {
            let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            verified_engine::engine::VerifiedEngine::run_verified(|| {
                lab_a.evaluate_push_and_drawdown_in_place(100.0, &pw, Some(&dc), 80, 0);
            });
        }
        assert_eq!(lab_a.median_ci_lower_points.len(), 81);
        assert_eq!(lab_a.drawdown_median_points.len(), 81);
        assert!(lab_a.last_terminal_median_ci.is_some());
    }
}

// [cite:clinical_trials_statistics]
