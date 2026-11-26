#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use std::sync::mpsc::Receiver;

#[path = "portfolio_workbench/engine.rs"]
pub(crate) mod engine;
#[path = "portfolio_workbench/io.rs"]
pub(crate) mod io;
#[path = "portfolio_workbench/state.rs"]
pub(crate) mod state;
#[path = "portfolio_workbench/tearsheet.rs"]
mod tearsheet;
#[path = "portfolio_workbench/ui.rs"]
mod ui;
#[path = "portfolio_workbench/ui_panels.rs"]
mod ui_panels;

use engine::{
    recompute_bsm, recompute_copula, recompute_kelly, recompute_ou, BsmEngineCache,
    CopulaEngineCache, ImportedDatasetSummary, KellyEngineCache, OuEngineCache,
};
use state::{PortfolioWorkbenchState, WorkbenchView};

pub struct PortfolioCopulaWorkbenchTool {
    pub(crate) state: PortfolioWorkbenchState,
    pub(crate) imported: Option<ImportedDatasetSummary>,
    pub(crate) kelly_cache: KellyEngineCache,
    pub(crate) copula_cache: CopulaEngineCache,
    pub(crate) ou_cache: OuEngineCache,
    pub(crate) bsm_cache: BsmEngineCache,
    pub(crate) open_rx: Option<Receiver<Result<(String, String), String>>>,
    pub(crate) save_rx: Option<Receiver<Result<String, String>>>,
    pub(crate) io_status: Option<(String, bool)>,
}

impl Default for PortfolioCopulaWorkbenchTool {
    fn default() -> Self {
        let mut tool = Self {
            state: PortfolioWorkbenchState::default(),
            imported: None,
            kelly_cache: KellyEngineCache::default(),
            copula_cache: CopulaEngineCache::default(),
            ou_cache: OuEngineCache::default(),
            bsm_cache: BsmEngineCache::default(),
            open_rx: None,
            save_rx: None,
            io_status: None,
        };
        tool.recalculate_all();
        tool
    }
}

fn named_points(label: &str, pts: &[egui_plot::PlotPoint]) -> (String, Vec<[f64; 2]>) {
    (
        String::from(label),
        pts.iter().map(|p| [p.x, p.y]).collect(),
    )
}

impl PortfolioCopulaWorkbenchTool {
    pub(crate) fn recalculate_all(&mut self) {
        self.kelly_cache = recompute_kelly(&self.state, self.imported.as_ref());
        self.copula_cache = recompute_copula(&self.state);
        self.ou_cache = recompute_ou(&self.state, self.imported.as_ref());
        self.bsm_cache = recompute_bsm(&self.state);
    }

    fn collect_view_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        match self.state.view {
            WorkbenchView::KellyFrontier => vec![
                named_points("Covariance Kelly Frontier", &self.kelly_cache.cov_frontier),
                named_points("Long-Only Kelly Frontier", &self.kelly_cache.long_frontier),
                named_points(
                    "Independent Kelly Frontier",
                    &self.kelly_cache.indep_frontier,
                ),
            ],
            WorkbenchView::CopulaTailCrash => vec![
                named_points("Copula Joint SGP Probability", &self.copula_cache.sgp_curve),
                named_points(
                    "Simultaneous Tail Crash Probability",
                    &self.copula_cache.crash_curve,
                ),
            ],
            WorkbenchView::OuMeanReversion => vec![
                named_points("OU 5th Percentile Cone", &self.ou_cache.p05_band),
                named_points("OU Median Trajectory", &self.ou_cache.p50_band),
                named_points("OU 95th Percentile Cone", &self.ou_cache.p95_band),
                named_points("OU Analytical Expectation", &self.ou_cache.mean_curve),
            ],
            WorkbenchView::BlackScholesGreeks => vec![
                named_points("European Call Profile", &self.bsm_cache.call_curve),
                named_points("European Put Profile", &self.bsm_cache.put_curve),
            ],
        }
    }
}

impl InteractiveTool for PortfolioCopulaWorkbenchTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Portfolio & Copula Risk Workbench"
    }

    fn domain(&self) -> &'static str {
        "financial_math"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.collect_view_datasets()
            .into_iter()
            .filter(|(_, pts)| !pts.is_empty())
            .collect()
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        let mut obs = Vec::with_capacity(6);
        if let Some(ref sol) = self.kelly_cache.cov_kelly {
            obs.push(("cov_kelly_leverage", sol.metrics.gross_leverage));
            obs.push(("cov_kelly_excess_growth", sol.metrics.excess_log_growth));
        }
        if let Some(t) = self.kelly_cache.cov_sim_tail {
            obs.push(("cvar_95_usd", t.cvar_95));
        }
        obs.push(("copula_crash_prob", self.copula_cache.active_crash_prob));
        obs.push((
            "copula_crash_multiplier",
            self.copula_cache.crash_multiplier,
        ));
        obs
    }

    fn has_finite_parameters(&self) -> bool {
        self.state.has_finite_parameters()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.state.clamp_in_place();
        self.recalculate_all();
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: PortfolioWorkbenchState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid PortfolioCopulaWorkbenchTool state: {e}"))?;
        parsed.validate()?;
        self.state = parsed;
        self.recalculate_all();
        Ok(())
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        self.poll_io_channels(parent_ui.ctx());
        let panel_id =
            crate::framework::scoped_panel_id(parent_ui, "portfolio_copula_workbench_side_panel");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            self.show_controls_panel(ui);
        });
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.show_main_workbench(ui);
        });
    }
}

// [cite:clinical_trials_statistics]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "PortfolioCopulaWorkbenchTool",
        display_name: "Portfolio & Copula Risk Workbench",
        domain: "financial_math",
        tags: &["kelly", "copula", "var", "cvar", "black-scholes", "greeks", "ornstein-uhlenbeck"],
        build: || Box::new(PortfolioCopulaWorkbenchTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for PortfolioCopulaWorkbenchTool {
    fn theory_description(&self) -> String {
        format!(
            "Multi-asset covariance Kelly allocation $w^* = \\Sigma^{{-1}}(\\mu - r\\mathbf{{1}})$ with projected-gradient long-only constraints, `95%`/`99%` Value-at-Risk and Conditional Value-at-Risk, Genz Cholesky QMC Gaussian copula tail dependence ($d = {}$, $\\rho = {:.2}$), Ornstein-Uhlenbeck forecast cones, and closed-form Black-Scholes-Merton Greeks.",
            self.state.copula_legs, self.state.rho
        )
    }

    fn phonetic_description(&self) -> String {
        "Multi asset covariance Kelly portfolio sizing, Gaussian copula tail risk, Ornstein Uhlenbeck mean reversion, and Black Scholes Merton option Greeks".into()
    }

    fn theory_citation(&self) -> String {
        "Thorp (2006) The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market; Genz (1992); Black & Scholes (1973); Merton (1973)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Covariance Kelly & Tail Risk".into(),
            "Solves Sigma^-1(mu - r*1) and projected-gradient long-only weights alongside 95%/99% VaR and CVaR".into(),
        );
        map.insert(
            "Gaussian Copula QMC Tail Crash".into(),
            "Evaluates 2..=12 leg joint parlay and simultaneous crash probabilities via Cholesky Halton QMC".into(),
        );
        map.insert(
            "Black-Scholes-Merton Greeks".into(),
            "Closed-form European Call and Put valuation with Delta, Gamma, Vega, Theta, and Rho sensitivities".into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Correlated Portfolio Crash & Kelly Leverage Audit",
    "domain": "financial_math",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "view": "KellyFrontier",
    "assets": [
      { "label": "Tech Momentum", "mu": 0.15, "sigma": 0.24, "marginal_u": 0.58 },
      { "label": "Credit Carry", "mu": 0.10, "sigma": 0.16, "marginal_u": 0.62 },
      { "label": "Commodity Trend", "mu": 0.11, "sigma": 0.20, "marginal_u": 0.54 },
      { "label": "EM Rates", "mu": 0.09, "sigma": 0.18, "marginal_u": 0.52 }
    ],
    "rho": 0.45,
    "risk_free_rate": 0.04,
    "bankroll": 2500.0,
    "horizon": 1.0,
    "leverage_cap": 1.0,
    "copula_legs": 4,
    "crash_quantile": 0.05,
    "ou_mu": 0.06,
    "ou_theta": 1.80,
    "ou_sigma": 0.12,
    "ou_x0": 0.16,
    "ou_dt": 0.02,
    "ou_horizon": 2.0,
    "ou_paths": 256,
    "bsm_spot": 105.0,
    "bsm_strike": 100.0,
    "bsm_vol": 0.22,
    "bsm_expiry": 0.75,
    "bsm_dividend": 0.015,
    "bsm_metric": "Delta",
    "seed": 2026
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "portfolio_copula_workbench_starter",
        title: "Correlated Portfolio Crash & Kelly Leverage Audit",
        description: "Audit covariance vs independent Kelly overbetting, multi-leg Gaussian copula tail crashes, OU forecast cones, and Black-Scholes Greeks.",
        category: "Risk Management",
        icon: "📊",
        domain: "financial_math",
        tool_name: Some("PortfolioCopulaWorkbenchTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

#[cfg(test)]
#[path = "portfolio_workbench/tests.rs"]
mod tests;
