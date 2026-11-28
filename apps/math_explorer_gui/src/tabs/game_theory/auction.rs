//! Interactive Myerson Optimal Auction & Bayes-Nash Bid Shading Laboratory.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "auction/atlas.rs"]
pub mod atlas;
#[path = "auction/challenge.rs"]
pub mod challenge;
/// Re-exported verified mechanism-design quadrature and Bayes-Nash evaluators.
pub mod math {
    pub use super::presets::AuctionPlotView;
    pub use math_explorer::applied::game_theory::mechanism_design::{
        AuctionDistEval, AuctionDistributionKind, AuctionEvalInputs, AuctionEvalWorkspace,
        AuctionRuleMode, AuctionSummaryMetrics,
    };
}
#[path = "auction/presets.rs"]
mod presets;
#[path = "auction/ui.rs"]
mod ui;

#[cfg(test)]
#[path = "auction/tests.rs"]
mod tests;

use crate::async_sim::atlas::ParameterAtlasController;
use crate::framework::{
    CalloutManager, CalloutSpec, ChallengeEvaluation, InteractiveTool, TraceSeries,
    TraceSnapshotManager,
};
use atlas::AUCTION_ATLAS_AXIS_PAIRS;
use eframe::egui;
pub use math::{
    AuctionDistEval, AuctionDistributionKind, AuctionEvalInputs, AuctionEvalWorkspace,
    AuctionPlotView, AuctionRuleMode, AuctionSummaryMetrics,
};
pub use presets::MyersonAuctionPresetConfig;
use scientific_metadata::theory::TheoryDescribable;
use std::hash::{Hash, Hasher};

const CURVE_SAMPLES: usize = 161;

/// Interactive Myerson Optimal Auction and Bayes-Nash Bid Shading tool.
pub struct MyersonAuctionTool {
    pub(crate) distribution: AuctionDistributionKind,
    pub(crate) rule_mode: AuctionRuleMode,
    pub(crate) plot_view: AuctionPlotView,
    pub(crate) uniform_min: f64,
    pub(crate) uniform_max: f64,
    pub(crate) exp_rate: f64,
    pub(crate) beta_alpha: f64,
    pub(crate) beta_beta: f64,
    pub(crate) n_bidders: usize,
    pub(crate) reserve_price: f64,
    pub(crate) focal_valuation: f64,
    pub(crate) candidate_bid: f64,
    pub(crate) mc_samples: usize,
    pub(crate) mc_seed: u64,
    pub(crate) dist_eval: AuctionDistEval,
    pub(crate) workspace: AuctionEvalWorkspace,
    pub(crate) summary: AuctionSummaryMetrics,
    pub(crate) jv_points: Vec<[f64; 2]>,
    pub(crate) exclusion_polygon: Vec<[f64; 2]>,
    pub(crate) second_price_points: Vec<[f64; 2]>,
    pub(crate) first_price_points: Vec<[f64; 2]>,
    pub(crate) revenue_curve: Vec<[f64; 2]>,
    pub(crate) welfare_curve: Vec<[f64; 2]>,
    pub(crate) deadweight_curve: Vec<[f64; 2]>,
    pub(crate) snapshot_manager: TraceSnapshotManager,
    pub(crate) callout_manager: CalloutManager,
    pub(crate) atlas: ParameterAtlasController,
    pub(crate) active_challenge: Option<&'static str>,
    revision: u64,
}

impl Default for MyersonAuctionTool {
    fn default() -> Self {
        let dist_eval = AuctionDistEval::from_params(
            AuctionDistributionKind::Uniform,
            0.0,
            100.0,
            0.1,
            2.0,
            2.0,
        );
        let mut tool = Self {
            distribution: AuctionDistributionKind::Uniform,
            rule_mode: AuctionRuleMode::SecondPrice,
            plot_view: AuctionPlotView::VirtualValuation,
            uniform_min: 0.0,
            uniform_max: 100.0,
            exp_rate: 0.1,
            beta_alpha: 2.0,
            beta_beta: 2.0,
            n_bidders: 2,
            reserve_price: 50.0,
            focal_valuation: 80.0,
            candidate_bid: 80.0,
            mc_samples: presets::default_mc_samples(),
            mc_seed: presets::default_mc_seed(),
            dist_eval,
            workspace: AuctionEvalWorkspace::default(),
            summary: AuctionSummaryMetrics::default(),
            jv_points: Vec::with_capacity(CURVE_SAMPLES),
            exclusion_polygon: Vec::new(),
            second_price_points: Vec::with_capacity(CURVE_SAMPLES),
            first_price_points: Vec::with_capacity(CURVE_SAMPLES),
            revenue_curve: Vec::with_capacity(CURVE_SAMPLES),
            welfare_curve: Vec::with_capacity(CURVE_SAMPLES),
            deadweight_curve: Vec::with_capacity(CURVE_SAMPLES),
            snapshot_manager: TraceSnapshotManager::new(),
            callout_manager: CalloutManager::new(),
            atlas: ParameterAtlasController::new(AUCTION_ATLAS_AXIS_PAIRS.to_vec()),
            active_challenge: None,
            revision: 1,
        };
        tool.recalculate();
        tool
    }
}

impl MyersonAuctionTool {
    pub(crate) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    /// Concise parameter summary for pinned [`TraceSnapshotManager`] overlays.
    #[must_use]
    pub fn param_summary(&self) -> String {
        format!(
            "{}, n={}, r={:.2}",
            self.distribution.label(),
            self.n_bidders,
            self.reserve_price
        )
    }

    pub(crate) fn current_trace_series(&self) -> Vec<TraceSeries> {
        match self.plot_view {
            AuctionPlotView::VirtualValuation => {
                vec![TraceSeries::new("J(v)", self.jv_points.clone())]
            }
            AuctionPlotView::BidFunctions => vec![
                TraceSeries::new("First-Price b*(v)", self.first_price_points.clone()),
                TraceSeries::new("Second-Price b(v)", self.second_price_points.clone()),
            ],
            AuctionPlotView::RevenueWelfare => vec![
                TraceSeries::new("Expected Revenue R(r)", self.revenue_curve.clone()),
                TraceSeries::new("Social Welfare W(r)", self.welfare_curve.clone()),
            ],
        }
    }

    pub(crate) fn record_history_and_recalculate(&mut self) {
        let prev_summary = self.param_summary();
        let prev_series = self.current_trace_series();
        self.snapshot_manager
            .push_history_run(&prev_summary, prev_series);
        self.recalculate();
    }

    /// Rebuilds all mechanism-design curves, Monte Carlo revenue, and equilibrium metrics.
    pub fn recalculate(&mut self) {
        self.enforce_challenge_locks();
        self.dist_eval = AuctionDistEval::from_params(
            self.distribution,
            self.uniform_min,
            self.uniform_max,
            self.exp_rate,
            self.beta_alpha,
            self.beta_beta,
        );
        let (v_min, v_max) = self.dist_eval.plot_bounds();
        self.n_bidders = self.n_bidders.clamp(1, 20);
        self.reserve_price = self.reserve_price.clamp(v_min, v_max);
        self.focal_valuation = self.focal_valuation.clamp(v_min, v_max);
        self.candidate_bid = self.candidate_bid.clamp(0.0, v_max);
        self.mc_samples = self.mc_samples.clamp(100, 25_000);

        let mut summary = self.workspace.evaluate_analytical(
            &self.dist_eval,
            AuctionEvalInputs {
                rule: self.rule_mode,
                reserve: self.reserve_price,
                n_bidders: self.n_bidders,
                focal_v: self.focal_valuation,
                candidate_bid: self.candidate_bid,
                fixed_point_tol: challenge::SHADING_FIXED_POINT_TOL,
            },
        );
        summary.mc_optimal_revenue = self.workspace.simulate_mc_optimal_revenue(
            &self.dist_eval,
            self.n_bidders,
            self.mc_samples,
            self.mc_seed,
        );
        self.summary = summary;
        self.rebuild_curves(v_min, v_max);
        self.sync_auction_callouts();
        self.bump_revision();
    }

    fn rebuild_curves(&mut self, v_min: f64, v_max: f64) {
        self.jv_points.clear();
        self.exclusion_polygon.clear();
        self.second_price_points.clear();
        self.first_price_points.clear();
        self.revenue_curve.clear();
        self.welfare_curve.clear();
        self.deadweight_curve.clear();

        let eps = if self.distribution == AuctionDistributionKind::Beta {
            1e-3
        } else {
            0.0
        };
        let lo = v_min + eps;
        let hi = (v_max - eps).max(lo + 1e-6);
        let r_star = self.summary.optimal_reserve.clamp(lo, hi);

        self.exclusion_polygon.push([lo, 0.0]);
        for i in 0..CURVE_SAMPLES {
            let t = i as f64 / (CURVE_SAMPLES - 1) as f64;
            let v = lo + t * (hi - lo);
            let jv = self
                .dist_eval
                .virtual_valuation(v)
                .clamp(-5.0 * v_max, 5.0 * v_max);
            self.jv_points.push([v, jv]);
            if v <= r_star && jv <= 0.0 {
                self.exclusion_polygon.push([v, jv]);
            }

            let v_grid = v_min + t * (v_max - v_min);
            let sp = if v_grid >= self.reserve_price {
                v_grid
            } else {
                0.0
            };
            let fp = self.dist_eval.first_price_equilibrium_bid(
                v_grid,
                self.reserve_price,
                self.n_bidders,
            );
            self.second_price_points.push([v_grid, sp]);
            self.first_price_points.push([v_grid, fp]);

            let (rev, wel, eff_wel, _) = self
                .dist_eval
                .expected_revenue_and_welfare(v_grid, self.n_bidders);
            self.revenue_curve.push([v_grid, rev]);
            self.welfare_curve.push([v_grid, wel]);
            self.deadweight_curve
                .push([v_grid, (eff_wel - wel).max(0.0)]);
        }
        self.exclusion_polygon.push([r_star, 0.0]);
    }
}

impl InteractiveTool for MyersonAuctionTool {
    fn name(&self) -> &'static str {
        "Myerson Optimal Auction"
    }

    fn domain(&self) -> &'static str {
        "game_theory"
    }

    fn set_suspended(&mut self, suspended: bool) {
        self.atlas.set_suspended(suspended);
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.revision.hash(&mut h);
        self.url_state_revision().hash(&mut h);
        self.snapshot_manager.len().hash(&mut h);
        h.finish().max(1)
    }

    fn url_state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.uniform_min.to_bits().hash(&mut h);
        self.uniform_max.to_bits().hash(&mut h);
        self.exp_rate.to_bits().hash(&mut h);
        self.beta_alpha.to_bits().hash(&mut h);
        self.beta_beta.to_bits().hash(&mut h);
        self.distribution.hash(&mut h);
        self.rule_mode.hash(&mut h);
        self.plot_view.hash(&mut h);
        self.n_bidders.hash(&mut h);
        self.reserve_price.to_bits().hash(&mut h);
        self.focal_valuation.to_bits().hash(&mut h);
        self.candidate_bid.to_bits().hash(&mut h);
        self.mc_samples.hash(&mut h);
        self.mc_seed.hash(&mut h);
        self.atlas.is_visible().hash(&mut h);
        self.atlas.axis_keys().hash(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        [
            self.uniform_min,
            self.uniform_max,
            self.exp_rate,
            self.beta_alpha,
            self.beta_beta,
            self.reserve_price,
            self.focal_valuation,
            self.candidate_bid,
            self.summary.optimal_reserve,
            self.summary.expected_revenue,
        ]
        .into_iter()
        .all(f64::is_finite)
    }

    fn clamp_parameters(&mut self) -> bool {
        let fin =
            |v: f64, lo: f64, hi: f64, d: f64| if v.is_finite() { v.clamp(lo, hi) } else { d };
        self.uniform_min = fin(self.uniform_min, 0.0, 499.0, 0.0);
        self.uniform_max = fin(self.uniform_max, self.uniform_min + 1.0, 500.0, 100.0);
        self.exp_rate = fin(self.exp_rate, 0.05, 5.0, 0.1);
        self.beta_alpha = fin(self.beta_alpha, 0.5, 8.0, 2.0);
        self.beta_beta = fin(self.beta_beta, 0.5, 8.0, 2.0);
        self.reserve_price = fin(self.reserve_price, 0.0, 500.0, 50.0);
        self.focal_valuation = fin(self.focal_valuation, 0.0, 500.0, 80.0);
        self.candidate_bid = fin(self.candidate_bid, 0.0, 500.0, 80.0);
        self.recalculate();
        true
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        let s = &self.summary;
        vec![
            ("optimal_reserve", s.optimal_reserve),
            ("expected_revenue", s.expected_revenue),
            ("optimal_revenue", s.optimal_revenue),
            ("mc_optimal_revenue", s.mc_optimal_revenue),
            ("social_welfare", s.social_welfare),
            ("trade_probability", s.trade_probability),
            ("equilibrium_bid", s.equilibrium_bid),
            ("candidate_utility", s.candidate_utility),
        ]
    }

    fn sync_callouts(&mut self, callouts: &[CalloutSpec]) {
        self.callout_manager.sync_specs(callouts);
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut datasets = vec![
            ("Virtual Valuation J(v)".to_string(), self.jv_points.clone()),
            (
                "First-Price Bid b*(v)".to_string(),
                self.first_price_points.clone(),
            ),
            (
                "Second-Price Bid b(v)".to_string(),
                self.second_price_points.clone(),
            ),
            (
                "Expected Seller Revenue R(r)".to_string(),
                self.revenue_curve.clone(),
            ),
            (
                "Expected Social Welfare W(r)".to_string(),
                self.welfare_curve.clone(),
            ),
        ];
        datasets.extend(self.snapshot_manager.export_datasets());
        datasets
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.export_config(true)).ok()
    }

    fn save_recovery_state(&self) -> Option<String> {
        serde_json::to_string(&self.export_config(false)).ok()
    }

    fn try_load_recovery_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let saved_snaps = self.snapshot_manager.clone();
        let mut cfg: MyersonAuctionPresetConfig = serde_json::from_str(state)
            .map_err(|e| format!("Invalid MyersonAuctionTool recovery state: {e}"))?;
        cfg.snapshot_manager = saved_snaps;
        self.apply_config(cfg)
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let cfg: MyersonAuctionPresetConfig = serde_json::from_str(state)
            .map_err(|e| format!("Invalid MyersonAuctionTool state: {e}"))?;
        self.apply_config(cfg)
    }

    fn apply_challenge(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        self.set_auction_challenge(challenge_id)
    }

    fn active_challenge_id(&self) -> Option<&'static str> {
        self.active_challenge
    }

    fn evaluate_challenge(&self) -> Option<ChallengeEvaluation> {
        self.eval_auction_challenge()
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "myerson_auction_side_panel");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            self.show_settings_panel(ui);
        });
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.show_central_plot(ui);
        });
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "MyersonAuctionTool",
        display_name: "Myerson Optimal Auction",
        domain: "game_theory",
        tags: &["auction", "mechanism design", "myerson", "bayes nash", "reserve price", "vickrey"],
        build: || Box::new(MyersonAuctionTool::default()),
    }
}

// [cite:graph_parameters_rust]
