#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::ou_process::MomentumType;
use std::sync::mpsc::Receiver;

#[path = "correlated_risk/engine.rs"]
pub(crate) mod engine;
#[path = "correlated_risk/export.rs"]
pub(crate) mod export;
#[path = "correlated_risk/ui.rs"]
mod ui;

use engine::{CopulaCacheKey, CopulaSweepRow, CorrelatedRiskState, OuCacheKey, OuSummaryStats};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RiskAnalysisMode {
    CopulaCorrelation,
    OrnsteinUhlenbeck,
}

pub struct CorrelatedRiskScenarioTool {
    pub(crate) scenario_label: String,
    pub(crate) mode: RiskAnalysisMode,

    // Bivariate Gaussian copula / SGP parameters
    pub(crate) prob_a: f64,
    pub(crate) prob_b: f64,
    pub(crate) rho: f64,

    // Ornstein-Uhlenbeck SDE parameters
    pub(crate) ou_mu: f64,
    pub(crate) ou_theta: f64,
    pub(crate) ou_sigma: f64,
    pub(crate) ou_x0: f64,
    pub(crate) ou_dt: f64,
    pub(crate) ou_steps: usize,
    pub(crate) ou_paths: usize,
    pub(crate) seed: u64,

    // Cached Copula sweep & active point (zero steady-state frame allocations)
    pub(crate) last_copula_key: Option<CopulaCacheKey>,
    pub(crate) copula_curve_points: Vec<PlotPoint>,
    pub(crate) naive_line_points: [PlotPoint; 2],
    pub(crate) active_rho_marker: Vec<PlotPoint>,
    pub(crate) copula_rows: Vec<CopulaSweepRow>,
    pub(crate) naive_prob: f64,
    pub(crate) active_sgp_prob: f64,
    pub(crate) copula_error: Option<String>,

    // Cached OU Monte Carlo trajectories & summary
    pub(crate) last_ou_key: Option<OuCacheKey>,
    pub(crate) ou_sample_paths: Vec<(String, Vec<PlotPoint>)>,
    pub(crate) ou_mean_path: Vec<PlotPoint>,
    pub(crate) ou_equilibrium_line: [PlotPoint; 2],
    pub(crate) ou_stats: Option<OuSummaryStats>,
    pub(crate) ou_momentum: MomentumType,
    pub(crate) ou_error: Option<String>,

    // Async file save channel & status banner
    pub(crate) save_rx: Option<Receiver<Result<String, String>>>,
    pub(crate) io_status: Option<(String, bool)>,
}

impl Default for CorrelatedRiskScenarioTool {
    fn default() -> Self {
        let mut tool = Self {
            scenario_label: String::from("Baseline Correlated Risk"),
            mode: RiskAnalysisMode::CopulaCorrelation,
            prob_a: 0.60,
            prob_b: 0.55,
            rho: 0.35,
            ou_mu: 0.50,
            ou_theta: 2.00,
            ou_sigma: 0.25,
            ou_x0: 0.85,
            ou_dt: 0.02,
            ou_steps: 150,
            ou_paths: 16,
            seed: 42,
            last_copula_key: None,
            copula_curve_points: Vec::new(),
            naive_line_points: [PlotPoint::new(-0.99, 0.33), PlotPoint::new(0.99, 0.33)],
            active_rho_marker: Vec::new(),
            copula_rows: Vec::new(),
            naive_prob: 0.33,
            active_sgp_prob: 0.33,
            copula_error: None,
            last_ou_key: None,
            ou_sample_paths: Vec::new(),
            ou_mean_path: Vec::new(),
            ou_equilibrium_line: [PlotPoint::new(0.0, 0.50), PlotPoint::new(3.0, 0.50)],
            ou_stats: None,
            ou_momentum: MomentumType::Normal,
            ou_error: None,
            save_rx: None,
            io_status: None,
        };
        tool.recalculate_if_needed();
        tool
    }
}

impl InteractiveTool for CorrelatedRiskScenarioTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Correlated Risk Scenarios"
    }
    fn domain(&self) -> &'static str {
        "financial_math"
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn has_finite_parameters(&self) -> bool {
        [
            self.prob_a,
            self.prob_b,
            self.rho,
            self.ou_mu,
            self.ou_theta,
            self.ou_sigma,
            self.ou_x0,
            self.ou_dt,
        ]
        .iter()
        .all(|v| v.is_finite())
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&CorrelatedRiskState {
            prob_a: self.prob_a,
            prob_b: self.prob_b,
            rho: self.rho,
            ou_mu: self.ou_mu,
            ou_theta: self.ou_theta,
            ou_sigma: self.ou_sigma,
            ou_x0: self.ou_x0,
            ou_dt: self.ou_dt,
            ou_steps: self.ou_steps,
            ou_paths: self.ou_paths,
            seed: self.seed,
            mode: self.mode,
            scenario_label: self.scenario_label.clone(),
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: CorrelatedRiskState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid CorrelatedRiskScenarioTool state: {e}"))?;
        parsed.validate()?;
        self.apply_validated_state(parsed);
        Ok(())
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        self.poll_save_channel(parent_ui.ctx());
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "correlated_risk_side_panel");
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
        id: "CorrelatedRiskScenarioTool",
        display_name: "Correlated Risk Scenarios",
        domain: "financial_math",
        tags: &["copula", "sgp", "ornstein-uhlenbeck", "monte-carlo", "correlation"],
        build: || Box::new(CorrelatedRiskScenarioTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for CorrelatedRiskScenarioTool {
    fn theory_description(&self) -> String {
        format!(
            "Bivariate Gaussian copula $C(u_1, u_2; \\rho) = \\Phi_\\rho(\\Phi^{{-1}}(u_1), \\Phi^{{-1}}(u_2))$ ($p_A = {:.2}, p_B = {:.2}, \\rho = {:.2}$) paired with Ornstein-Uhlenbeck mean-reverting SDE $dX_t = \\theta(\\mu - X_t)\\,dt + \\sigma\\,dW_t$ ($\\mu = {:.2}, \\theta = {:.2}, \\sigma = {:.2}$) integrated via Euler-Maruyama.",
            self.prob_a, self.prob_b, self.rho, self.ou_mu, self.ou_theta, self.ou_sigma
        )
    }
    fn phonetic_description(&self) -> String {
        "Bivariate Gaussian copula dependence and Ornstein Uhlenbeck mean reversion stochastic process theory".into()
    }
    fn theory_citation(&self) -> String {
        "An Introduction to Copulas (Nelsen, 2006); On the Theory of the Brownian Motion (Uhlenbeck & Ornstein, 1930)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Gaussian Copula Joint Probability".into(),
            "Couples marginal probabilities (p_A, p_B) with latent normal correlation rho".into(),
        );
        map.insert(
            "Ornstein-Uhlenbeck Euler-Maruyama".into(),
            "Seeded discrete-time trajectories reverting toward equilibrium mu at rate theta"
                .into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Negative-Correlation Parlay & Streak Reversion",
    "domain": "financial_math",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "prob_a": 0.75,
    "prob_b": 0.60,
    "rho": -0.25,
    "ou_mu": 0.45,
    "ou_theta": 1.20,
    "ou_sigma": 0.18,
    "ou_x0": 0.68,
    "ou_dt": 0.02,
    "ou_steps": 200,
    "ou_paths": 24,
    "seed": 2026,
    "mode": "CopulaCorrelation",
    "scenario_label": "Cannibalized Parlay & Hot-Streak Decay"
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "correlated_risk_starter",
        title: "Negative-Correlation Parlay & Streak Reversion",
        description: "Compare Gaussian copula joint event pricing against naive independence and simulate seeded Ornstein-Uhlenbeck mean reversion.",
        category: "Risk Management",
        icon: "🔗",
        domain: "financial_math",
        tool_name: Some("CorrelatedRiskScenarioTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

#[cfg(test)]
#[path = "correlated_risk/tests.rs"]
mod tests;
