//! Interactive Tail-Risk (`VaR` / `CVaR`), Underwater Drawdown, and Performance-Ratio Workbench.
//!
//! Compares Historical empirical, Parametric Gaussian, and 4th-order Cornish-Fisher tail-loss
//! estimators alongside Sharpe, Sortino, Calmar, and underwater recovery metrics with zero
//! heap allocations in [`TailRiskDrawdownTool::evaluate_in_place`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod plots;
mod state;

#[cfg(test)]
mod tests;

use crate::framework::{CalloutManager, ExportCapabilities, InteractiveTool};
use crate::web_io::PlatformInstant;
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::risk::{
    cornish_fisher_cvar, cornish_fisher_var, drawdown_series_from_returns, gaussian_cvar,
    gaussian_var, historical_cvar, historical_var, summarize_return_risk, RiskSummary,
};
use oxidize_core::rng::{OxidizeRng, BASE_SEED};
use rand::Rng;

/// Maximum simulated periodic return horizon preallocated on startup.
pub const MAX_TAIL_HORIZON: usize = 504;
/// Number of confidence-level grid samples across `[0.55, 0.99]`.
pub const CONFIDENCE_GRID_POINTS: usize = 45;

const PERIODS_PER_YEAR: f64 = 252.0;
const INITIAL_CAPITAL_INDEX: f64 = 100.0;

/// Interactive workbench for tail-loss quantiles, expected shortfall, and underwater drawdowns.
pub struct TailRiskDrawdownTool {
    pub(super) annual_volatility: f64,
    pub(super) annual_drift: f64,
    pub(super) tail_shock_prob: f64,
    pub(super) tail_shock_scale: f64,
    pub(super) confidence_level: f64,
    pub(super) risk_free_rate: f64,
    pub(super) horizon_periods: usize,
    pub(super) seed: u64,
    pub(super) callout_manager: CalloutManager,
    pub(super) summary: Option<RiskSummary>,
    pub(super) last_compute_ms: f64,
    returns_buf: Vec<f64>,
    drawdown_buf: Vec<f64>,
    scratch_buf: Vec<f64>,
    pub(super) hist_var_curve: Vec<PlotPoint>,
    pub(super) hist_cvar_curve: Vec<PlotPoint>,
    pub(super) gauss_var_curve: Vec<PlotPoint>,
    pub(super) gauss_cvar_curve: Vec<PlotPoint>,
    pub(super) cf_var_curve: Vec<PlotPoint>,
    pub(super) cf_cvar_curve: Vec<PlotPoint>,
    pub(super) wealth_points: Vec<PlotPoint>,
    pub(super) underwater_points: Vec<PlotPoint>,
}

impl Default for TailRiskDrawdownTool {
    fn default() -> Self {
        let mut tool = Self {
            annual_volatility: 0.22,
            annual_drift: 0.10,
            tail_shock_prob: 0.06,
            tail_shock_scale: 0.09,
            confidence_level: 0.95,
            risk_free_rate: 0.03,
            horizon_periods: 252,
            seed: 42,
            callout_manager: CalloutManager::new(),
            summary: None,
            last_compute_ms: 0.0,
            returns_buf: vec![0.0; MAX_TAIL_HORIZON],
            drawdown_buf: vec![0.0; MAX_TAIL_HORIZON],
            scratch_buf: vec![0.0; MAX_TAIL_HORIZON],
            hist_var_curve: Vec::with_capacity(CONFIDENCE_GRID_POINTS),
            hist_cvar_curve: Vec::with_capacity(CONFIDENCE_GRID_POINTS),
            gauss_var_curve: Vec::with_capacity(CONFIDENCE_GRID_POINTS),
            gauss_cvar_curve: Vec::with_capacity(CONFIDENCE_GRID_POINTS),
            cf_var_curve: Vec::with_capacity(CONFIDENCE_GRID_POINTS),
            cf_cvar_curve: Vec::with_capacity(CONFIDENCE_GRID_POINTS),
            wealth_points: Vec::with_capacity(MAX_TAIL_HORIZON + 1),
            underwater_points: Vec::with_capacity(MAX_TAIL_HORIZON + 1),
        };
        tool.recompute();
        tool
    }
}

impl TailRiskDrawdownTool {
    /// Evaluates synthetic return trajectory, tail-loss curves, drawdowns, and ratios in place
    /// with zero dynamic heap allocations.
    pub fn evaluate_in_place(&mut self) {
        let n = self.horizon_periods.clamp(30, MAX_TAIL_HORIZON);
        self.generate_returns_in_place(n);
        self.populate_wealth_and_drawdown_in_place(n);
        self.populate_tail_curves_in_place(n);
        let rf_step = self.risk_free_rate / PERIODS_PER_YEAR;
        self.summary = summarize_return_risk(
            &self.returns_buf[..n],
            self.confidence_level.clamp(0.55, 0.995),
            rf_step,
            PERIODS_PER_YEAR,
            &mut self.scratch_buf[..n],
        )
        .ok();
    }

    fn generate_returns_in_place(&mut self, n: usize) {
        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ self.seed);
        let dt = 1.0 / PERIODS_PER_YEAR;
        let sigma_step = self.annual_volatility.max(0.02) * dt.sqrt();
        let jump_comp = self.tail_shock_prob * self.tail_shock_scale;
        let mu_step = self.annual_drift * dt + jump_comp * 0.35;

        for slot in &mut self.returns_buf[..n] {
            let u1 = rng.gen::<f64>().clamp(1e-12, 1.0 - 1e-12);
            let u2 = rng.gen::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            let shock = if rng.gen::<f64>() < self.tail_shock_prob {
                let tail_u = rng.gen::<f64>().clamp(1e-6, 1.0);
                -self.tail_shock_scale * (0.5 + 0.5 * (-tail_u.ln()).min(3.0))
            } else {
                0.0
            };
            *slot = (mu_step + sigma_step * z + shock).clamp(-0.75, 1.50);
        }
    }

    fn populate_wealth_and_drawdown_in_place(&mut self, n: usize) {
        self.wealth_points.clear();
        self.underwater_points.clear();
        self.wealth_points
            .push(PlotPoint::new(0.0, INITIAL_CAPITAL_INDEX));
        self.underwater_points.push(PlotPoint::new(0.0, 0.0));

        let _ = drawdown_series_from_returns(&self.returns_buf[..n], &mut self.drawdown_buf[..n]);
        let mut wealth = INITIAL_CAPITAL_INDEX;
        for step in 0..n {
            wealth *= 1.0 + self.returns_buf[step];
            let x = (step + 1) as f64;
            self.wealth_points.push(PlotPoint::new(x, wealth));
            self.underwater_points
                .push(PlotPoint::new(x, -100.0 * self.drawdown_buf[step]));
        }
    }

    fn populate_tail_curves_in_place(&mut self, n: usize) {
        self.hist_var_curve.clear();
        self.hist_cvar_curve.clear();
        self.gauss_var_curve.clear();
        self.gauss_cvar_curve.clear();
        self.cf_var_curve.clear();
        self.cf_cvar_curve.clear();

        let returns = &self.returns_buf[..n];
        for idx in 0..CONFIDENCE_GRID_POINTS {
            let alpha = 0.55 + 0.01 * (idx as f64);
            if let Ok(v) = historical_var(returns, alpha, &mut self.scratch_buf[..n]) {
                self.hist_var_curve.push(PlotPoint::new(alpha, v * 100.0));
            }
            if let Ok(v) = historical_cvar(returns, alpha, &mut self.scratch_buf[..n]) {
                self.hist_cvar_curve.push(PlotPoint::new(alpha, v * 100.0));
            }
            if let Ok(v) = gaussian_var(returns, alpha) {
                self.gauss_var_curve.push(PlotPoint::new(alpha, v * 100.0));
            }
            if let Ok(v) = gaussian_cvar(returns, alpha) {
                self.gauss_cvar_curve.push(PlotPoint::new(alpha, v * 100.0));
            }
            if let Ok(v) = cornish_fisher_var(returns, alpha) {
                self.cf_var_curve.push(PlotPoint::new(alpha, v * 100.0));
            }
            if let Ok(v) = cornish_fisher_cvar(returns, alpha) {
                self.cf_cvar_curve.push(PlotPoint::new(alpha, v * 100.0));
            }
        }
    }

    pub(super) fn recompute(&mut self) {
        let start = PlatformInstant::now();
        self.evaluate_in_place();
        self.refresh_callouts();
        self.last_compute_ms = start.elapsed().as_secs_f64() * 1000.0;
    }

    fn refresh_callouts(&mut self) {
        self.callout_manager.clear_auto();
        if let Some(sum) = self.summary {
            let alpha = self.confidence_level;
            self.callout_manager.set_auto_callout(
                "hist_cvar",
                [alpha, sum.historical_cvar * 100.0],
                "Historical CVaR",
                [-26.0, -28.0],
            );
            self.callout_manager.set_auto_callout(
                "gauss_var",
                [alpha, sum.gaussian_var * 100.0],
                "Gaussian VaR",
                [-24.0, 22.0],
            );
        }
    }
}

impl InteractiveTool for TailRiskDrawdownTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Tail Risk & Drawdown"
    }
    fn domain(&self) -> &'static str {
        "financial_math"
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }
    fn observables(&self) -> Vec<(&'static str, f64)> {
        self.collect_observables()
    }
    fn save_state(&self) -> Option<String> {
        self.save_state_json()
    }
    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.load_state_json(state)
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_export_datasets()
    }
    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        self.build_export_primitives()
    }
    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_workspace(ui);
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "TailRiskDrawdownTool",
        display_name: "Tail Risk & Drawdown",
        domain: "financial_math",
        tags: &["risk", "var", "cvar", "drawdown", "sharpe"],
        build: || Box::new(TailRiskDrawdownTool::default()),
    }
}

// [cite:clinical_trials_statistics]
