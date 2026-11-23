//! Multivariate Gaussian copula tail-risk and SGP pricing workbench (`CopulaRiskTool`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::copula::MAX_COPULA_DIMENSION;

#[path = "copula_risk/engine.rs"]
mod engine;
#[path = "copula_risk/io.rs"]
pub(crate) mod io;
#[path = "copula_risk/ui.rs"]
mod ui;

#[cfg(test)]
#[path = "copula_risk/tests.rs"]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CopulaPlotView {
    CorrelationSweep,
    BivariateContours,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CorrelationStructureMode {
    Equicorrelated,
    PairwiseMatrix,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CopulaRiskState {
    pub equicorrelation: f64,
    pub book_vig_pct: f64,
    pub num_legs: usize,
    pub marginals: Vec<f64>,
    pub leg_labels: Vec<String>,
    pub mode: CorrelationStructureMode,
    pub pairwise_upper: Vec<f64>,
    pub contour_leg_a: usize,
    pub contour_leg_b: usize,
    pub plot_view: CopulaPlotView,
    pub rng_seed: u64,
}

#[derive(Clone, Debug)]
pub struct SweepPricingRow {
    pub rho: f64,
    pub joint_prob: f64,
    pub naive_prob: f64,
    pub fair_decimal_odds: f64,
    pub offered_decimal_odds: f64,
    pub fair_american_odds: f64,
}

pub struct CopulaRiskTool {
    pub(crate) equicorrelation: f64,
    pub(crate) book_vig_pct: f64,
    pub(crate) num_legs: usize,
    pub(crate) marginals: Vec<f64>,
    pub(crate) leg_labels: Vec<String>,
    pub(crate) mode: CorrelationStructureMode,
    pub(crate) pairwise_upper: Vec<f64>,
    pub(crate) contour_leg_a: usize,
    pub(crate) contour_leg_b: usize,
    pub(crate) plot_view: CopulaPlotView,
    pub(crate) rng_seed: u64,
    pub(crate) state_rev: u64,

    pub(crate) joint_probability: Option<f64>,
    pub(crate) naive_probability: f64,
    pub(crate) effective_pair_rho: f64,
    pub(crate) calc_error: Option<String>,
    pub(crate) sweep_copula_points: Vec<PlotPoint>,
    pub(crate) sweep_naive_points: [PlotPoint; 2],
    pub(crate) active_marker_points: Vec<PlotPoint>,
    pub(crate) contour_curves: Vec<(String, Vec<PlotPoint>)>,
    pub(crate) sampled_cloud_points: Vec<PlotPoint>,
    pub(crate) pricing_rows: Vec<SweepPricingRow>,
    pub(crate) io_state: io::CopulaIoState,
}

impl Default for CopulaRiskTool {
    fn default() -> Self {
        let mut tool = Self {
            equicorrelation: 0.25,
            book_vig_pct: 4.5,
            num_legs: 3,
            marginals: vec![0.62, 0.58, 0.70],
            leg_labels: vec![
                "Leg 1 (QB Pass Yds)".to_string(),
                "Leg 2 (WR Rec Yds)".to_string(),
                "Leg 3 (Team Win)".to_string(),
            ],
            mode: CorrelationStructureMode::Equicorrelated,
            pairwise_upper: vec![0.45, 0.25, 0.20],
            contour_leg_a: 0,
            contour_leg_b: 1,
            plot_view: CopulaPlotView::CorrelationSweep,
            rng_seed: 2026,
            state_rev: 1,
            joint_probability: None,
            naive_probability: 0.0,
            effective_pair_rho: 0.25,
            calc_error: None,
            sweep_copula_points: Vec::new(),
            sweep_naive_points: [PlotPoint::new(-0.4, 0.25), PlotPoint::new(0.95, 0.25)],
            active_marker_points: Vec::new(),
            contour_curves: Vec::new(),
            sampled_cloud_points: Vec::new(),
            pricing_rows: Vec::new(),
            io_state: io::CopulaIoState::default(),
        };
        tool.recalculate();
        tool
    }
}

impl CopulaRiskTool {
    pub(crate) fn bump_revision(&mut self) {
        self.state_rev = self.state_rev.wrapping_add(1).max(1);
    }

    pub(crate) fn to_state(&self) -> CopulaRiskState {
        CopulaRiskState {
            equicorrelation: self.equicorrelation,
            book_vig_pct: self.book_vig_pct,
            num_legs: self.num_legs,
            marginals: self.marginals.clone(),
            leg_labels: self.leg_labels.clone(),
            mode: self.mode,
            pairwise_upper: self.pairwise_upper.clone(),
            contour_leg_a: self.contour_leg_a,
            contour_leg_b: self.contour_leg_b,
            plot_view: self.plot_view,
            rng_seed: self.rng_seed,
        }
    }
}

impl InteractiveTool for CopulaRiskTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Copula Tail Risk"
    }

    fn domain(&self) -> &'static str {
        "financial_math"
    }

    fn state_revision(&self) -> u64 {
        self.state_rev
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn has_finite_parameters(&self) -> bool {
        Self::validate_state(&self.to_state()).is_ok()
    }

    fn clamp_parameters(&mut self) -> bool {
        let mut changed = false;
        let clamped_legs = self.num_legs.clamp(2, MAX_COPULA_DIMENSION);
        if clamped_legs != self.num_legs {
            self.resize_legs(clamped_legs);
            changed = true;
        }
        let min_equi = (-1.0 / (self.num_legs.saturating_sub(1) as f64) + 0.02).max(-0.95);
        let eq = if self.equicorrelation.is_finite() {
            self.equicorrelation.clamp(min_equi, 0.95)
        } else {
            0.25_f64.max(min_equi)
        };
        if (eq - self.equicorrelation).abs() > f64::EPSILON {
            self.equicorrelation = eq;
            changed = true;
        }
        let vig = if self.book_vig_pct.is_finite() {
            self.book_vig_pct.clamp(0.0, 25.0)
        } else {
            4.5
        };
        if (vig - self.book_vig_pct).abs() > f64::EPSILON {
            self.book_vig_pct = vig;
            changed = true;
        }
        for m in &mut self.marginals {
            let next = if m.is_finite() {
                m.clamp(0.01, 0.99)
            } else {
                0.5
            };
            if (next - *m).abs() > f64::EPSILON {
                *m = next;
                changed = true;
            }
        }
        if changed {
            self.recalculate();
        }
        changed
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.to_state()).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: CopulaRiskState =
            serde_json::from_str(state).map_err(|e| format!("Invalid copula risk state: {e}"))?;
        self.apply_state(parsed)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.render_workbench_ui(ui);
        });
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "CopulaRiskTool",
        display_name: "Copula Tail Risk",
        domain: "financial_math",
        tags: &[
            "copula",
            "gaussian copula",
            "cholesky",
            "tail risk",
            "same game parlay",
            "joint probability",
        ],
        build: || Box::new(CopulaRiskTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for CopulaRiskTool {
    fn theory_description(&self) -> String {
        format!(
            "Multivariate Gaussian copula $C(u_1, \\dots, u_d; R) = \\Phi_R(\\Phi^{{-1}}(u_1), \\dots, \\Phi^{{-1}}(u_d))$ for $d = {}$ correlated legs via Cholesky factorization $R = L L^\\top$.",
            self.num_legs
        )
    }

    fn phonetic_description(&self) -> String {
        "Multivariate Gaussian copula Cholesky correlation pricing and joint probability theory"
            .into()
    }

    fn theory_citation(&self) -> String {
        "Fonctions de repartition a n dimensions et leurs marges (Sklar, 1959)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Gaussian Copula".into(),
            "Couples marginal probabilities through a positive-definite latent normal correlation matrix".into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Four-Leg Correlated Parlay",
    "domain": "financial_math",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "equicorrelation": 0.35,
    "book_vig_pct": 5.0,
    "num_legs": 4,
    "marginals": [0.65, 0.58, 0.72, 0.55],
    "leg_labels": ["QB 275+ Yds", "WR1 80+ Yds", "Game Over 48.5", "Favorite ML"],
    "mode": "Equicorrelated",
    "pairwise_upper": [0.55, 0.40, 0.35, 0.45, 0.25, 0.30],
    "contour_leg_a": 0,
    "contour_leg_b": 1,
    "plot_view": "CorrelationSweep",
    "rng_seed": 4242
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "copula_risk_starter",
        title: "Four-Leg Correlated Parlay",
        description: "C(u1..u4; R): Compare Cholesky-sampled 4-leg Gaussian copula fair odds against naive independent multiplication.",
        category: "Risk Management",
        icon: "🎲",
        domain: "financial_math",
        tool_name: Some("CopulaRiskTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

// [cite:clinical_trials_statistics]
