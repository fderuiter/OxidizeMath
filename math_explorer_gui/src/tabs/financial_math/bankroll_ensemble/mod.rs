//! Interactive Volatility Drag and Non-Ergodic Capital Ensemble Laboratory.
//!
//! Tracks per-step 5th, 25th, 50th, 75th, and 95th wealth percentiles, closed
//! polygon fan bands, underwater drawdown curves, maximum underwater duration,
//! and terminal 95% `VaR` and `CVaR` with zero dynamic heap allocations.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod plots;

use crate::framework::CalloutManager;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::kelly::{
    expected_growth_rate, expected_value, kelly_fraction, BankrollFraction, Odds, UnitInterval,
};
use math_explorer::pure_math::statistics::risk::{
    drawdown_series_from_wealth, historical_cvar, historical_var, max_underwater_duration_wealth,
};
use oxidize_core::rng::{OxidizeRng, BASE_SEED};
use rand::Rng;

/// Maximum number of sequential allocations preallocated for zero-allocation evaluation.
pub const MAX_ENSEMBLE_BETS: usize = 1000;
/// Number of Monte Carlo ensemble trajectories simulated per rerun.
pub const ENSEMBLE_PATH_COUNT: usize = 64;
/// Upper bound on custom allocation fraction `f` so `ln(1 - f)` remains finite.
pub const MAX_CUSTOM_FRACTION: f64 = 0.95;
/// Default custom allocation fraction demonstrating overbetting volatility drag.
pub const DEFAULT_CUSTOM_FRACTION: f64 = 0.30;

const MIN_POSITIVE_WEALTH: f64 = 1e-300;

/// Active view mode inside [`super::bankroll_growth::BankrollGrowthTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BankrollLabMode {
    /// Classic Full / Half / Quarter Kelly compounding trajectories.
    #[default]
    KellyStrategies,
    /// Interactive non-ergodic volatility-drag and percentile fan-band laboratory.
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

fn sorted_percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = q.clamp(0.0, 1.0) * ((sorted.len() - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (pos - (lo as f64)) * (sorted[hi] - sorted[lo])
}

/// Preallocated simulator and UI controller for Kelly volatility-drag and tail-risk experiments.
pub struct BankrollEnsembleLab {
    /// Custom allocation fraction `f` clamped to `0.0..=0.95`.
    pub custom_fraction: f64,
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
    path_wealths: Vec<f64>,
    scratch_sort: Vec<f64>,
    terminal_returns: Vec<f64>,
    median_wealth_buf: Vec<f64>,
    drawdown_buf: Vec<f64>,
    /// Latest initial capital $W_0$ evaluated.
    pub last_initial_bankroll: f64,
    /// Latest logarithmic growth rate $g(f)$ from `expected_growth_rate`.
    pub last_growth_rate: f64,
    /// Latest optimal Kelly fraction $f^*$ when positive edge exists.
    pub last_f_star: Option<f64>,
    /// Latest terminal expected wealth `E[W_T]`.
    pub last_expected_terminal: f64,
    /// Latest terminal median wealth $W_{\text{median}}(T)$.
    pub last_median_terminal: f64,
    /// Peak-to-trough fractional maximum drawdown along the empirical median trajectory.
    pub last_max_drawdown: f64,
    /// Maximum consecutive underwater steps strictly below prior running peak.
    pub last_underwater_duration: usize,
    /// Terminal 95% Historical Value-at-Risk as a fractional capital loss.
    pub last_terminal_var_95: f64,
    /// Terminal 95% Historical Expected Shortfall (`CVaR`) as a fractional capital loss.
    pub last_terminal_cvar_95: f64,
    /// Whether `f > 2 * f*` (or `g(f) < 0` despite positive edge), flagging ruin.
    pub is_ruin_zone: bool,
}

impl Default for BankrollEnsembleLab {
    fn default() -> Self {
        Self::new()
    }
}

impl BankrollEnsembleLab {
    /// Constructs a new ensemble lab with preallocated trajectory, polygon, and sorting buffers.
    pub fn new() -> Self {
        let cap = MAX_ENSEMBLE_BETS + 1;
        Self {
            custom_fraction: DEFAULT_CUSTOM_FRACTION,
            hypothesis: None,
            callout_manager: CalloutManager::new(),
            expected_points: Vec::with_capacity(cap),
            median_points: Vec::with_capacity(cap),
            empirical_median_points: Vec::with_capacity(cap),
            p05_points: Vec::with_capacity(cap),
            p25_points: Vec::with_capacity(cap),
            p75_points: Vec::with_capacity(cap),
            p95_points: Vec::with_capacity(cap),
            fan_5_95_poly: Vec::with_capacity(2 * cap),
            fan_25_75_poly: Vec::with_capacity(2 * cap),
            underwater_points: Vec::with_capacity(cap),
            path_wealths: vec![0.0; ENSEMBLE_PATH_COUNT],
            scratch_sort: vec![0.0; ENSEMBLE_PATH_COUNT],
            terminal_returns: vec![0.0; ENSEMBLE_PATH_COUNT],
            median_wealth_buf: vec![1.0; cap],
            drawdown_buf: vec![0.0; cap],
            last_initial_bankroll: 100.0,
            last_growth_rate: 0.0,
            last_f_star: None,
            last_expected_terminal: 0.0,
            last_median_terminal: 0.0,
            last_max_drawdown: 0.0,
            last_underwater_duration: 0,
            last_terminal_var_95: 0.0,
            last_terminal_cvar_95: 0.0,
            is_ruin_zone: false,
        }
    }

    /// Resets custom fraction, hypothesis state, and callouts to defaults.
    pub fn reset_defaults(&mut self) {
        self.custom_fraction = DEFAULT_CUSTOM_FRACTION;
        self.hypothesis = None;
        self.callout_manager.clear_auto();
    }

    /// Clamps `custom_fraction` into the safe finite interval `0.0..=0.95`.
    pub fn clamp_fraction(&mut self) {
        if !self.custom_fraction.is_finite() {
            self.custom_fraction = DEFAULT_CUSTOM_FRACTION;
        } else {
            self.custom_fraction = self.custom_fraction.clamp(0.0, MAX_CUSTOM_FRACTION);
        }
    }

    fn reset_step_zero(&mut self, w0: f64) {
        self.expected_points.clear();
        self.median_points.clear();
        self.empirical_median_points.clear();
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
        self.p05_points.push(origin);
        self.p25_points.push(origin);
        self.p75_points.push(origin);
        self.p95_points.push(origin);
        self.median_wealth_buf[0] = w0.max(MIN_POSITIVE_WEALTH);
    }

    /// Evaluates analytical curves, per-step wealth quantiles, underwater drawdowns,
    /// and terminal `VaR`/`CVaR` with zero heap allocations.
    pub fn evaluate_in_place(
        &mut self,
        initial_bankroll: f64,
        p: &UnitInterval,
        o: &Odds,
        num_bets: usize,
        seed_counter: u64,
    ) {
        self.clamp_fraction();
        let steps = num_bets.min(MAX_ENSEMBLE_BETS);
        let w0 = initial_bankroll.max(0.0);
        self.last_initial_bankroll = w0;
        let f_clamped = self.custom_fraction;
        let Ok(frac) = BankrollFraction::new(f_clamped) else {
            return;
        };

        let ev = expected_value(p, o);
        let g = expected_growth_rate(p, o, &frac);
        let f_star = kelly_fraction(p, o).ok().map(|fk| fk.value());
        self.last_growth_rate = g;
        self.last_f_star = f_star;
        self.is_ruin_zone = f_star.is_some_and(|fs| f_clamped > 2.0 * fs);
        self.reset_step_zero(w0);

        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ seed_counter.wrapping_mul(0x9E37_79B9));
        let net_b = o.net_profit_multiplier();
        let step_ev_factor = (1.0 + f_clamped * ev).max(0.0);
        let mut ew = w0;

        for t in 1..=steps {
            ew = (ew * step_ev_factor).max(0.0);
            let w_med = if g.is_finite() {
                (w0 * ((t as f64) * g).exp()).max(0.0)
            } else {
                0.0
            };
            let (p05, p25, emp_med, p75, p95) =
                self.step_ensemble_quantiles(&mut rng, p.value(), f_clamped, net_b);
            let x = t as f64;
            self.expected_points.push(PlotPoint::new(x, ew));
            self.median_points.push(PlotPoint::new(x, w_med));
            self.empirical_median_points
                .push(PlotPoint::new(x, emp_med));
            self.p05_points.push(PlotPoint::new(x, p05));
            self.p25_points.push(PlotPoint::new(x, p25));
            self.p75_points.push(PlotPoint::new(x, p75));
            self.p95_points.push(PlotPoint::new(x, p95));
            self.median_wealth_buf[t] = emp_med.max(MIN_POSITIVE_WEALTH);
        }

        self.rebuild_fan_polygons_and_tail_risk(steps, w0);
        self.last_expected_terminal = self.expected_points.last().map_or(w0, |pt| pt.y);
        self.last_median_terminal = self.median_points.last().map_or(w0, |pt| pt.y);
    }

    fn step_ensemble_quantiles(
        &mut self,
        rng: &mut OxidizeRng,
        win_prob: f64,
        f_clamped: f64,
        net_b: f64,
    ) -> (f64, f64, f64, f64, f64) {
        for w in &mut self.path_wealths {
            let stake = *w * f_clamped;
            if rng.r#gen::<f64>() < win_prob {
                *w = (*w + stake * net_b).max(0.0);
            } else {
                *w = (*w - stake).max(0.0);
            }
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

    fn rebuild_fan_polygons_and_tail_risk(&mut self, steps: usize, w0: f64) {
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

        if w0 > 0.0 {
            for (dst, &w_end) in self.terminal_returns.iter_mut().zip(&self.path_wealths) {
                *dst = (w_end - w0) / w0;
            }
            self.last_terminal_var_95 =
                historical_var(&self.terminal_returns, 0.95, &mut self.scratch_sort).unwrap_or(0.0);
            self.last_terminal_cvar_95 =
                historical_cvar(&self.terminal_returns, 0.95, &mut self.scratch_sort)
                    .unwrap_or(0.0);
        } else {
            self.last_terminal_var_95 = 0.0;
            self.last_terminal_cvar_95 = 0.0;
        }
    }

    /// Synchronizes automated `CalloutManager` overlays for growth rate, ruin, and 95% `VaR`/`CVaR`.
    pub fn sync_callouts(&mut self, num_bets: usize) {
        let steps = num_bets.min(MAX_ENSEMBLE_BETS);
        if self.median_points.is_empty() {
            self.callout_manager.clear_auto();
            return;
        }
        let mid_idx = (steps / 2).min(self.median_points.len() - 1);
        let mid_pt = self.median_points[mid_idx];
        self.callout_manager.set_auto_callout(
            "growth_rate",
            [mid_pt.x, mid_pt.y],
            format!(
                "g(f={:.2}) = {:+.4}/step",
                self.custom_fraction, self.last_growth_rate
            ),
            [24.0, -26.0],
        );

        let end_x = steps as f64;
        let p05_end = self.p05_points.last().map_or(0.0, |pt| pt.y);
        let var_dollars = self.last_initial_bankroll * self.last_terminal_var_95;
        let cvar_dollars = self.last_initial_bankroll * self.last_terminal_cvar_95;
        self.callout_manager.set_auto_callout(
            "var_95",
            [end_x, p05_end],
            format!(
                "95% VaR: {:.1}% (${:.2})",
                self.last_terminal_var_95 * 100.0,
                var_dollars
            ),
            [-84.0, 18.0],
        );
        self.callout_manager.set_auto_callout(
            "cvar_95",
            [(0.78 * end_x).max(0.0), p05_end],
            format!(
                "95% CVaR: {:.1}% (${:.2})",
                self.last_terminal_cvar_95 * 100.0,
                cvar_dollars
            ),
            [-84.0, 38.0],
        );

        if let Some(fs) = self
            .last_f_star
            .filter(|&fs| self.custom_fraction > 2.0 * fs)
        {
            self.callout_manager.set_auto_callout(
                "ruin_warning",
                [end_x, self.last_median_terminal],
                format!(
                    "Ruin Trap (f={:.2} > 2f*={:.2}): W_med={:.4} vs E[W]={:.2}",
                    self.custom_fraction,
                    2.0 * fs,
                    self.last_median_terminal,
                    self.last_expected_terminal
                ),
                [-90.0, -34.0],
            );
        } else {
            self.remove_auto_tag("ruin_warning");
        }
    }

    fn remove_auto_tag(&mut self, tag: &str) {
        if let Some(id) = self
            .callout_manager
            .callouts()
            .iter()
            .find(|c| c.tag.as_deref() == Some(tag))
            .map(|c| c.id)
        {
            self.callout_manager.remove_callout(id);
        }
    }
}

// [cite:clinical_trials_statistics]
