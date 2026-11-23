//! Interactive Volatility Drag, Solvency Barrier, and Non-Ergodic Bankroll Ensemble Laboratory.
//!
//! Powered by [`BankrollEnsembleWorkspace`] and [`kelly_barrier_risk`] to evaluate both
//! non-ergodic overbetting divergence (`f > 2 * f*`) and finite-drawdown barrier risks
//! (`0 < f <= f*`) with zero heap allocations inside simulation updates.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::CalloutManager;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::kelly::{
    expected_growth_rate, expected_value, kelly_fraction, BankrollFraction, Odds, UnitInterval,
};
use math_explorer::pure_math::statistics::ruin::{
    kelly_barrier_risk, BankrollEnsembleWorkspace, EnsembleSolvencyReport, KellyBarrierRisk,
    ProportionalPortfolioBet,
};
use oxidize_core::rng::{OxidizeRng, BASE_SEED};

#[path = "bankroll_ensemble_ui.rs"]
mod ui;

/// Maximum number of sequential wagers preallocated for zero-allocation evaluation.
pub const MAX_ENSEMBLE_BETS: usize = 1000;
/// Number of Monte Carlo ensemble trajectories simulated per rerun.
pub const ENSEMBLE_PATH_COUNT: usize = 1000;
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
    /// Classic Full / Half / Quarter Kelly compounding trajectories.
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

/// Preallocated simulator and UI controller for Kelly volatility-drag and barrier experiments.
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
    /// Horizontal drawdown floor line `[0, B_floor] .. [T, B_floor]`.
    pub barrier_points: Vec<PlotPoint>,
    workspace: BankrollEnsembleWorkspace,
    /// Latest logarithmic growth rate $g(f)$ from `expected_growth_rate`.
    pub last_growth_rate: f64,
    /// Latest optimal Kelly fraction $f^*$ when positive edge exists.
    pub last_f_star: Option<f64>,
    /// Latest terminal expected wealth `E[W_T]`.
    pub last_expected_terminal: f64,
    /// Latest terminal median wealth $W_{\text{median}}(T)$.
    pub last_median_terminal: f64,
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
            barrier_points: Vec::with_capacity(2),
            workspace,
            last_growth_rate: 0.0,
            last_f_star: None,
            last_expected_terminal: 0.0,
            last_median_terminal: 0.0,
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
        if !self.custom_fraction.is_finite() {
            self.custom_fraction = DEFAULT_CUSTOM_FRACTION;
        } else {
            self.custom_fraction = self.custom_fraction.clamp(0.0, MAX_CUSTOM_FRACTION);
        }
        if !self.barrier_floor_ratio.is_finite() {
            self.barrier_floor_ratio = DEFAULT_BARRIER_FLOOR_RATIO;
        } else {
            self.barrier_floor_ratio = self
                .barrier_floor_ratio
                .clamp(MIN_BARRIER_FLOOR_RATIO, MAX_BARRIER_FLOOR_RATIO);
        }
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
        self.clamp_fraction();
        let steps = num_bets.min(MAX_ENSEMBLE_BETS);
        let w0 = initial_bankroll.max(1e-6);
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
        self.last_barrier_risk = kelly_barrier_risk(p, o, &frac, self.barrier_floor_ratio).ok();

        let barrier_floor = w0 * self.barrier_floor_ratio;
        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ seed_counter.wrapping_mul(0x9E37_79B9));
        self.last_solvency_report = ProportionalPortfolioBet::unit_weight(*p, *o, frac)
            .and_then(|bet| {
                self.workspace
                    .simulate_proportional(w0, barrier_floor, &bet, steps, &mut rng)
            })
            .ok();

        self.populate_plot_series(w0, barrier_floor, ev, f_clamped, g, steps);
    }

    fn populate_plot_series(
        &mut self,
        w0: f64,
        barrier_floor: f64,
        ev: f64,
        f_clamped: f64,
        g: f64,
        steps: usize,
    ) {
        self.expected_points.clear();
        self.median_points.clear();
        self.empirical_median_points.clear();
        self.barrier_points.clear();

        self.expected_points.push(PlotPoint::new(0.0, w0));
        self.median_points.push(PlotPoint::new(0.0, w0));
        self.barrier_points.push(PlotPoint::new(0.0, barrier_floor));
        self.barrier_points
            .push(PlotPoint::new(steps as f64, barrier_floor));

        let step_ev_factor = (1.0 + f_clamped * ev).max(0.0);
        let mut ew = w0;
        for t in 1..=steps {
            ew = (ew * step_ev_factor).max(0.0);
            let w_med = if g.is_finite() {
                (w0 * ((t as f64) * g).exp()).max(0.0)
            } else {
                0.0
            };
            let x = t as f64;
            self.expected_points.push(PlotPoint::new(x, ew));
            self.median_points.push(PlotPoint::new(x, w_med));
        }

        for (t, &emp_w) in self.workspace.median_trajectory().iter().enumerate() {
            self.empirical_median_points
                .push(PlotPoint::new(t as f64, emp_w));
        }

        self.last_expected_terminal = self.expected_points.last().map_or(w0, |pt| pt.y);
        self.last_median_terminal = self.median_points.last().map_or(w0, |pt| pt.y);
    }

    /// Synchronizes automated `CalloutManager` overlays for `expected_growth_rate` and ruin.
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
                "g(f={:.2}) = {:+.4}/bet",
                self.custom_fraction, self.last_growth_rate
            ),
            [24.0, -26.0],
        );

        if let Some(fs) = self
            .last_f_star
            .filter(|&fs| self.custom_fraction > 2.0 * fs)
        {
            self.callout_manager.set_auto_callout(
                "ruin_warning",
                [steps as f64, self.last_median_terminal],
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
