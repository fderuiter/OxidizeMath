//! Interactive Markowitz Mean-Variance Efficient Frontier & Active-Set Allocation Workbench.
//!
//! Compares unconstrained analytical Merton-Lagrangian portfolios against long-only and
//! concentration-capped active-set quadratic-programming frontiers across up to
//! [`MAX_PORTFOLIO_ASSETS`] (`16`) correlated assets with zero heap allocations in
//! [`MarkowitzFrontierTool::evaluate_in_place`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod plots;
mod state;

#[cfg(test)]
mod tests;

use crate::framework::{CalloutManager, ExportCapabilities, InteractiveTool};
use crate::web_io::PlatformInstant;
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::applied::portfolio::{
    constrained_frontier_in_place, constrained_min_variance, constrained_tangency,
    unconstrained_frontier_in_place, unconstrained_min_variance, unconstrained_tangency,
    PortfolioAllocation, PortfolioError, PortfolioSpec, MAX_PORTFOLIO_ASSETS,
};

/// Number of target-return grid points along each efficient frontier curve.
pub const FRONTIER_GRID_POINTS: usize = 41;

const ZERO_ALLOCATION: PortfolioAllocation = PortfolioAllocation {
    asset_count: 2,
    weights: [0.0; MAX_PORTFOLIO_ASSETS],
    expected_return: 0.0,
    variance: 0.0,
    volatility: 0.0,
    sharpe_ratio: 0.0,
};

/// Interactive workbench for unconstrained and constrained Markowitz portfolio optimization.
pub struct MarkowitzFrontierTool {
    pub(super) risk_free_rate: f64,
    pub(super) max_asset_weight: f64,
    pub(super) asset_names: Vec<String>,
    pub(super) expected_returns: Vec<f64>,
    pub(super) volatilities: Vec<f64>,
    pub(super) upper_correlations: Vec<f64>,
    pub(super) callout_manager: CalloutManager,
    pub(super) unc_gmv: Option<PortfolioAllocation>,
    pub(super) unc_tangency: Option<PortfolioAllocation>,
    pub(super) con_gmv: Option<PortfolioAllocation>,
    pub(super) con_tangency: Option<PortfolioAllocation>,
    pub(super) last_error: Option<PortfolioError>,
    pub(super) last_compute_ms: f64,
    unc_frontier_buf: [PortfolioAllocation; FRONTIER_GRID_POINTS],
    con_frontier_buf: [PortfolioAllocation; FRONTIER_GRID_POINTS],
    pub(super) unc_frontier_pts: Vec<PlotPoint>,
    pub(super) con_frontier_pts: Vec<PlotPoint>,
    pub(super) asset_pts: Vec<PlotPoint>,
    pub(super) cml_pts: Vec<PlotPoint>,
}

impl Default for MarkowitzFrontierTool {
    fn default() -> Self {
        let mut tool = Self {
            risk_free_rate: 0.03,
            max_asset_weight: 1.0,
            asset_names: vec![
                "Global Equities".into(),
                "Sovereign Bonds".into(),
                "Commodities".into(),
                "Real Estate".into(),
            ],
            expected_returns: vec![0.11, 0.045, 0.075, 0.085],
            volatilities: vec![0.18, 0.06, 0.21, 0.15],
            upper_correlations: vec![0.10, 0.25, 0.45, -0.15, 0.12, 0.30],
            callout_manager: CalloutManager::new(),
            unc_gmv: None,
            unc_tangency: None,
            con_gmv: None,
            con_tangency: None,
            last_error: None,
            last_compute_ms: 0.0,
            unc_frontier_buf: [ZERO_ALLOCATION; FRONTIER_GRID_POINTS],
            con_frontier_buf: [ZERO_ALLOCATION; FRONTIER_GRID_POINTS],
            unc_frontier_pts: Vec::with_capacity(FRONTIER_GRID_POINTS),
            con_frontier_pts: Vec::with_capacity(FRONTIER_GRID_POINTS),
            asset_pts: Vec::with_capacity(MAX_PORTFOLIO_ASSETS),
            cml_pts: Vec::with_capacity(2),
        };
        tool.recompute();
        tool
    }
}

impl MarkowitzFrontierTool {
    /// Solves unconstrained and constrained Markowitz frontiers and optimal portfolios in place
    /// with zero dynamic heap allocations.
    pub fn evaluate_in_place(&mut self) {
        self.clear_plot_buffers();
        let n = self.expected_returns.len();
        let spec = match PortfolioSpec::from_upper_correlations(
            &self.expected_returns,
            &self.volatilities,
            &self.upper_correlations,
        ) {
            Ok(s) => s,
            Err(err) => {
                self.last_error = Some(err);
                return;
            }
        };
        self.last_error = None;
        for i in 0..n {
            self.asset_pts.push(PlotPoint::new(
                self.volatilities[i] * 100.0,
                self.expected_returns[i] * 100.0,
            ));
        }
        self.solve_unconstrained_in_place(&spec);
        self.solve_constrained_in_place(&spec, n);
    }

    fn clear_plot_buffers(&mut self) {
        self.unc_gmv = None;
        self.unc_tangency = None;
        self.con_gmv = None;
        self.con_tangency = None;
        self.unc_frontier_pts.clear();
        self.con_frontier_pts.clear();
        self.asset_pts.clear();
        self.cml_pts.clear();
    }

    fn solve_unconstrained_in_place(&mut self, spec: &PortfolioSpec) {
        self.unc_gmv = unconstrained_min_variance(spec, self.risk_free_rate).ok();
        self.unc_tangency = unconstrained_tangency(spec, self.risk_free_rate).ok();
        let max_asset_ret = self
            .expected_returns
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_ret = self
            .unc_gmv
            .map_or(0.0, |gmv| gmv.expected_return.min(max_asset_ret));
        let max_ret = (max_asset_ret * 1.15).max(min_ret + 0.01);
        if let Ok(cnt) = unconstrained_frontier_in_place(
            spec,
            min_ret,
            max_ret,
            self.risk_free_rate,
            &mut self.unc_frontier_buf,
        ) {
            for alloc in &self.unc_frontier_buf[..cnt] {
                self.unc_frontier_pts.push(PlotPoint::new(
                    alloc.volatility * 100.0,
                    alloc.expected_return * 100.0,
                ));
            }
        }
    }

    fn solve_constrained_in_place(&mut self, spec: &PortfolioSpec, n: usize) {
        let min_cap = (1.0 / (n as f64) + 1e-4).min(1.0);
        let cap = self.max_asset_weight.clamp(min_cap, 1.0);
        let upper = [cap; MAX_PORTFOLIO_ASSETS];

        self.con_gmv = constrained_min_variance(spec, self.risk_free_rate, &upper[..n]).ok();
        self.con_tangency = constrained_tangency(spec, self.risk_free_rate, &upper[..n]).ok();
        if let Ok(cnt) = constrained_frontier_in_place(
            spec,
            self.risk_free_rate,
            &upper[..n],
            &mut self.con_frontier_buf,
        ) {
            for alloc in &self.con_frontier_buf[..cnt] {
                self.con_frontier_pts.push(PlotPoint::new(
                    alloc.volatility * 100.0,
                    alloc.expected_return * 100.0,
                ));
            }
        }
        if let Some(tan) = self.con_tangency {
            let max_vol = self
                .volatilities
                .iter()
                .copied()
                .fold(tan.volatility, f64::max)
                * 1.15;
            let top_ret = self.risk_free_rate + tan.sharpe_ratio * max_vol;
            self.cml_pts
                .push(PlotPoint::new(0.0, self.risk_free_rate * 100.0));
            self.cml_pts
                .push(PlotPoint::new(max_vol * 100.0, top_ret * 100.0));
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
        if let Some(gmv) = self.con_gmv {
            self.callout_manager.set_auto_callout(
                "con_gmv",
                [gmv.volatility * 100.0, gmv.expected_return * 100.0],
                "Constrained GMV",
                [-20.0, 22.0],
            );
        }
        if let Some(tan) = self.con_tangency {
            self.callout_manager.set_auto_callout(
                "con_tangency",
                [tan.volatility * 100.0, tan.expected_return * 100.0],
                "Max-Sharpe Tangency",
                [-22.0, -28.0],
            );
        }
    }
}

impl InteractiveTool for MarkowitzFrontierTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Markowitz Efficient Frontier"
    }
    fn domain(&self) -> &'static str {
        "financial_math"
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }
    fn has_finite_parameters(&self) -> bool {
        self.risk_free_rate.is_finite()
            && self.max_asset_weight.is_finite()
            && self.expected_returns.iter().all(|v| v.is_finite())
            && self.volatilities.iter().all(|v| v.is_finite())
            && self.upper_correlations.iter().all(|v| v.is_finite())
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
        id: "MarkowitzFrontierTool",
        display_name: "Markowitz Efficient Frontier",
        domain: "financial_math",
        tags: &["markowitz", "portfolio", "frontier", "sharpe", "qp"],
        build: || Box::new(MarkowitzFrontierTool::default()),
    }
}

// [cite:clinical_trials_statistics]
