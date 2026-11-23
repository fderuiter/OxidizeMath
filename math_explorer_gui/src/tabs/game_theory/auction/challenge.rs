//! Inverse challenges for [`super::MyersonAuctionTool`]: welfare-constrained reserve pricing
//! and First-Price Bayes-Nash bid-shading equilibrium discovery.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::math::{AuctionDistributionKind, AuctionPlotView, AuctionRuleMode};
use super::{MyersonAuctionPresetConfig, MyersonAuctionTool};
use crate::framework::{ChallengeEvaluation, ChallengeMetadata, InteractiveTool};

/// Welfare-constrained reserve inverse challenge ID.
pub const CHALLENGE_MYERSON_WELFARE_RESERVE: &str = "myerson_welfare_constrained_reserve";

/// Symmetric First-Price Bayes-Nash bid-shading equilibrium challenge ID.
pub const CHALLENGE_FIRST_PRICE_SHADING: &str = "first_price_shading_equilibrium";

/// Required minimum expected seller revenue for `myerson_welfare_constrained_reserve`.
pub const TARGET_MIN_REVENUE: f64 = 40.0;
/// Required minimum trade probability for `myerson_welfare_constrained_reserve`.
pub const TARGET_MIN_TRADE_PROB: f64 = 0.84;
/// Fixed-point bid tolerance around $b^*(100; 50, 2) = 62.5$ for `first_price_shading_equilibrium`.
pub const SHADING_FIXED_POINT_TOL: f64 = 0.5;

impl MyersonAuctionTool {
    pub(crate) fn is_env_locked(&self) -> bool {
        self.active_challenge.is_some()
    }

    pub(crate) fn is_reserve_locked(&self) -> bool {
        self.active_challenge == Some(CHALLENGE_FIRST_PRICE_SHADING)
    }

    pub(crate) fn enforce_challenge_locks(&mut self) {
        match self.active_challenge {
            Some(CHALLENGE_MYERSON_WELFARE_RESERVE) => {
                self.distribution = AuctionDistributionKind::Uniform;
                self.uniform_min = 0.0;
                self.uniform_max = 100.0;
                self.n_bidders = 2;
                self.rule_mode = AuctionRuleMode::SecondPrice;
                self.focal_valuation = 80.0;
                self.reserve_price = if self.reserve_price.is_finite() {
                    self.reserve_price.clamp(0.0, 100.0)
                } else {
                    50.0
                };
                self.candidate_bid = if self.focal_valuation >= self.reserve_price {
                    self.focal_valuation
                } else {
                    0.0
                };
            }
            Some(CHALLENGE_FIRST_PRICE_SHADING) => {
                self.distribution = AuctionDistributionKind::Uniform;
                self.uniform_min = 0.0;
                self.uniform_max = 100.0;
                self.n_bidders = 2;
                self.reserve_price = 50.0;
                self.focal_valuation = 100.0;
                self.rule_mode = AuctionRuleMode::FirstPrice;
                self.candidate_bid = if self.candidate_bid.is_finite() {
                    self.candidate_bid.clamp(0.0, 100.0)
                } else {
                    100.0
                };
            }
            _ => {}
        }
    }

    pub(crate) fn set_auction_challenge(
        &mut self,
        challenge_id: Option<&str>,
    ) -> Result<(), String> {
        match challenge_id {
            None => {
                self.active_challenge = None;
                self.recalculate();
                Ok(())
            }
            Some(CHALLENGE_MYERSON_WELFARE_RESERVE) => {
                self.active_challenge = Some(CHALLENGE_MYERSON_WELFARE_RESERVE);
                self.plot_view = AuctionPlotView::RevenueWelfare;
                self.recalculate();
                Ok(())
            }
            Some(CHALLENGE_FIRST_PRICE_SHADING) => {
                self.active_challenge = Some(CHALLENGE_FIRST_PRICE_SHADING);
                self.plot_view = AuctionPlotView::BidFunctions;
                self.recalculate();
                Ok(())
            }
            Some(other) => Err(format!(
                "Unsupported challenge '{other}' for MyersonAuctionTool"
            )),
        }
    }

    pub(crate) fn eval_auction_challenge(&self) -> Option<ChallengeEvaluation> {
        let s = &self.summary;
        match self.active_challenge {
            Some(CHALLENGE_MYERSON_WELFARE_RESERVE) => {
                let rev_shortfall = (TARGET_MIN_REVENUE - s.expected_revenue).max(0.0);
                let prob_shortfall = (TARGET_MIN_TRADE_PROB - s.trade_probability).max(0.0);
                let passed = rev_shortfall == 0.0 && prob_shortfall == 0.0;
                let residual = rev_shortfall + 100.0 * prob_shortfall;
                let score = (100.0 - 8.0 * rev_shortfall - 350.0 * prob_shortfall).clamp(0.0, 100.0);
                Some(ChallengeEvaluation::new(
                    score,
                    passed,
                    "Rev + P_trade Deficit",
                    residual,
                    format!(
                        "Revenue = {:.2} (need ≥ {:.1}), P_trade = {:.1}% (need ≥ {:.1}%)",
                        s.expected_revenue,
                        TARGET_MIN_REVENUE,
                        s.trade_probability * 100.0,
                        TARGET_MIN_TRADE_PROB * 100.0
                    ),
                ))
            }
            Some(CHALLENGE_FIRST_PRICE_SHADING) => {
                let err = (self.candidate_bid - s.equilibrium_bid).abs();
                let passed = s.is_equilibrium_fixed_point && err <= SHADING_FIXED_POINT_TOL;
                let score = if passed {
                    100.0
                } else {
                    (100.0 * (1.0 - err / 37.5)).clamp(0.0, 99.0)
                };
                Some(ChallengeEvaluation::new(
                    score,
                    passed,
                    "|b - b*(v)|",
                    err,
                    format!(
                        "Candidate b = {:.2}, Bayes-Nash b*(100) = {:.2} (tol ±{:.1}), U(b|v) = {:.2}",
                        self.candidate_bid, s.equilibrium_bid, SHADING_FIXED_POINT_TOL, s.candidate_utility
                    ),
                ))
            }
            _ => None,
        }
    }

    pub(crate) fn sync_auction_callouts(&mut self) {
        let s = &self.summary;
        match self.plot_view {
            AuctionPlotView::VirtualValuation => {
                self.callout_manager.set_auto_callout(
                    "myerson_r_star",
                    [s.optimal_reserve, 0.0],
                    format!("Myerson r* = {:.2} (J=0)", s.optimal_reserve),
                    [20.0, -22.0],
                );
                let j_focal = self.dist_eval.virtual_valuation(self.focal_valuation);
                self.callout_manager.set_auto_callout(
                    "myerson_focal",
                    [self.focal_valuation, j_focal],
                    format!("J({:.1}) = {:.2}", self.focal_valuation, j_focal),
                    [-24.0, -24.0],
                );
            }
            AuctionPlotView::BidFunctions => {
                self.callout_manager.set_auto_callout(
                    "myerson_r_star",
                    [self.reserve_price, self.reserve_price],
                    format!("Reserve r = {:.2}", self.reserve_price),
                    [20.0, 20.0],
                );
                self.callout_manager.set_auto_callout(
                    "myerson_focal",
                    [self.focal_valuation, s.equilibrium_bid],
                    format!("b*({:.1}) = {:.2}", self.focal_valuation, s.equilibrium_bid),
                    [-26.0, -22.0],
                );
            }
            AuctionPlotView::RevenueWelfare => {
                self.callout_manager.set_auto_callout(
                    "myerson_r_star",
                    [s.optimal_reserve, s.optimal_revenue],
                    format!("Opt Rev {:.2} @ r*={:.2}", s.optimal_revenue, s.optimal_reserve),
                    [18.0, -22.0],
                );
                self.callout_manager.set_auto_callout(
                    "myerson_focal",
                    [self.reserve_price, s.expected_revenue],
                    format!(
                        "Active r={:.1}: Rev={:.2}, P_trade={:.0}%",
                        self.reserve_price,
                        s.expected_revenue,
                        s.trade_probability * 100.0
                    ),
                    [-24.0, 22.0],
                );
            }
        }
    }
}

fn generate_welfare_reserve_baseline() -> String {
    let mut tool = MyersonAuctionTool {
        distribution: AuctionDistributionKind::Uniform,
        uniform_min: 0.0,
        uniform_max: 100.0,
        n_bidders: 2,
        reserve_price: 50.0,
        focal_valuation: 80.0,
        rule_mode: AuctionRuleMode::SecondPrice,
        plot_view: AuctionPlotView::RevenueWelfare,
        active_challenge: Some(CHALLENGE_MYERSON_WELFARE_RESERVE),
        ..MyersonAuctionTool::default()
    };
    tool.recalculate();
    tool.save_state().unwrap_or_default()
}

fn clamp_welfare_reserve_state(raw_json: &str) -> Result<String, String> {
    let mut tool = MyersonAuctionTool::default();
    tool.state_bounds().validate_json(raw_json)?;
    let cfg: MyersonAuctionPresetConfig = serde_json::from_str(raw_json)
        .map_err(|e| format!("Invalid MyersonAuctionTool challenge JSON: {e}"))?;
    tool.apply_config(cfg)?;
    tool.set_auction_challenge(Some(CHALLENGE_MYERSON_WELFARE_RESERVE))?;
    tool.save_state()
        .ok_or_else(|| "Failed to serialize clamped MyersonAuctionTool state".to_string())
}

fn generate_first_price_shading_baseline() -> String {
    let mut tool = MyersonAuctionTool {
        distribution: AuctionDistributionKind::Uniform,
        uniform_min: 0.0,
        uniform_max: 100.0,
        n_bidders: 2,
        reserve_price: 50.0,
        focal_valuation: 100.0,
        candidate_bid: 100.0,
        rule_mode: AuctionRuleMode::FirstPrice,
        plot_view: AuctionPlotView::BidFunctions,
        active_challenge: Some(CHALLENGE_FIRST_PRICE_SHADING),
        ..MyersonAuctionTool::default()
    };
    tool.recalculate();
    tool.save_state().unwrap_or_default()
}

fn clamp_first_price_shading_state(raw_json: &str) -> Result<String, String> {
    let mut tool = MyersonAuctionTool::default();
    tool.state_bounds().validate_json(raw_json)?;
    let cfg: MyersonAuctionPresetConfig = serde_json::from_str(raw_json)
        .map_err(|e| format!("Invalid MyersonAuctionTool challenge JSON: {e}"))?;
    tool.apply_config(cfg)?;
    tool.set_auction_challenge(Some(CHALLENGE_FIRST_PRICE_SHADING))?;
    tool.save_state()
        .ok_or_else(|| "Failed to serialize clamped MyersonAuctionTool state".to_string())
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_MYERSON_WELFARE_RESERVE,
        title: "Welfare-Constrained Reserve Pricing",
        description: "On Uniform(0, 100) with n = 2 bidders, Myerson's unconstrained r* = 50 maximizes revenue (41.67) but blocks 25% of all trades. Tune reserve r to capture ≥ 40.0 expected revenue while guaranteeing ≥ 84.0% trade probability.",
        category: "Mechanism Design",
        icon: "⚖",
        domain: "game_theory",
        tool_name: "MyersonAuctionTool",
        locked_inputs: "Uniform(0, 100), n = 2 bidders, Second-Price (Vickrey)",
        tunable_budget: "Reserve price r ∈ [0, 100]",
        target_summary: "Expected Seller Revenue ≥ 40.0 and Trade Probability P_trade ≥ 84.0%",
        baseline_state_generator: generate_welfare_reserve_baseline,
        clamp_locked_state: clamp_welfare_reserve_state,
    }
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_FIRST_PRICE_SHADING,
        title: "First-Price Shading Equilibrium",
        description: "In a First-Price Sealed-Bid auction on Uniform(0, 100) with n = 2 bidders and reserve r = 50, truthful bidding b = 100 yields zero utility. Shade candidate bid b(100) to the symmetric Bayes-Nash fixed point b*(100) = 62.5.",
        category: "Mechanism Design",
        icon: "🎯",
        domain: "game_theory",
        tool_name: "MyersonAuctionTool",
        locked_inputs: "Uniform(0, 100), n = 2 bidders, r = 50.0, v = 100.0, First-Price Sealed-Bid",
        tunable_budget: "Candidate bid b ∈ [0, 100]",
        target_summary: "is_fixed_point holds within ±0.5 of b*(100; 50, 2) = 62.5",
        baseline_state_generator: generate_first_price_shading_baseline,
        clamp_locked_state: clamp_first_price_shading_state,
    }
}
