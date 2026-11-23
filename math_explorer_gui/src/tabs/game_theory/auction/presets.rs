//! Preset configuration, validation, `TheoryDescribable`, and `StarterPresetMetadata`
//! registrations for [`super::MyersonAuctionTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::MyersonAuctionTool;
use super::atlas::{default_auction_atlas_axes, is_default_auction_atlas_axes};
use crate::framework::TraceSnapshotManager;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported bidder valuation distributions in [`super::MyersonAuctionTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuctionDistributionKind {
    /// Continuous uniform distribution $U(v_{\min}, v_{\max})$.
    #[default]
    Uniform,
    /// Exponential distribution $\text{Exp}(\lambda)$ on $[0, \infty)$.
    Exp,
    /// Beta distribution $\text{Beta}(\alpha, \beta)$ on $[0, 1]$.
    Beta,
}

impl AuctionDistributionKind {
    /// Human-readable distribution label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Uniform => "Uniform(min, max)",
            Self::Exp => "Exponential(λ)",
            Self::Beta => "Beta(α, β)",
        }
    }
}

/// Auction payment and bidding rule mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuctionRuleMode {
    /// Second-price sealed-bid (Vickrey) auction with truthful dominant strategy $b(v) = v$.
    #[default]
    SecondPrice,
    /// First-price sealed-bid auction with symmetric Bayes-Nash shading $b^*(v; r, n)$.
    FirstPrice,
}

impl AuctionRuleMode {
    /// Human-readable auction format label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SecondPrice => "Second-Price (Vickrey)",
            Self::FirstPrice => "First-Price Sealed-Bid",
        }
    }
}

/// Active curve family rendered in the central plot of [`super::MyersonAuctionTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuctionPlotView {
    /// Virtual valuation $J(v)$ and shaded $J(v) < 0$ exclusion region.
    #[default]
    VirtualValuation,
    /// First-price equilibrium shading $b^*(v)$ vs second-price truthful $b(v) = v$.
    BidFunctions,
    /// Expected seller revenue $R(r)$ vs social welfare $W(r)$ across reserve prices.
    RevenueWelfare,
}

pub(super) const DEFAULT_MC_SEED: u64 = 0x4D59_4552_534F_4E31;

pub(super) fn default_mc_samples() -> usize {
    2_000
}

pub(super) fn default_mc_seed() -> u64 {
    DEFAULT_MC_SEED
}

/// Bounded preset configuration (`<= 4,096` bytes) for [`MyersonAuctionTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MyersonAuctionPresetConfig {
    /// Lower bound of `Uniform(min, max)` (placed first so float perturbation tests stay valid).
    pub uniform_min: f64,
    /// Upper bound of `Uniform(min, max)`.
    pub uniform_max: f64,
    /// Rate parameter $\lambda$ of `Exp(rate)`.
    pub exp_rate: f64,
    /// Shape $\alpha$ of `Beta(alpha, beta)`.
    pub beta_alpha: f64,
    /// Shape $\beta$ of `Beta(alpha, beta)`.
    pub beta_beta: f64,
    /// Active valuation distribution family.
    pub distribution: AuctionDistributionKind,
    /// Active payment & bidding rule.
    pub rule_mode: AuctionRuleMode,
    /// Active central plot view.
    #[serde(default)]
    pub plot_view: AuctionPlotView,
    /// Number of symmetric bidders $n \in [1, 20]$.
    pub n_bidders: usize,
    /// Seller reserve price $r$.
    pub reserve_price: f64,
    /// Focal bidder valuation $v$.
    pub focal_valuation: f64,
    /// Candidate submitted bid $b(v)$ tested against [`math_explorer::applied::game_theory::equilibrium::is_fixed_point`].
    pub candidate_bid: f64,
    /// Monte Carlo simulation count for [`math_explorer::applied::game_theory::mechanism_design::simulate_optimal_revenue_with_rng`].
    #[serde(default = "default_mc_samples")]
    pub mc_samples: usize,
    /// Deterministic RNG seed for Monte Carlo revenue comparison.
    #[serde(default = "default_mc_seed")]
    pub mc_seed: u64,
    /// Pinned trace snapshots (`<= 10`).
    #[serde(default, skip_serializing_if = "TraceSnapshotManager::is_empty")]
    pub snapshot_manager: TraceSnapshotManager,
    /// Whether the progressive 2D parameter atlas is visible.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub atlas_visible: bool,
    /// Selected `(x, y)` atlas parameter keys.
    #[serde(
        default = "default_auction_atlas_axes",
        skip_serializing_if = "is_default_auction_atlas_axes"
    )]
    pub atlas_axes: [String; 2],
}

impl MyersonAuctionTool {
    pub(crate) fn export_config(&self, include_snapshots: bool) -> MyersonAuctionPresetConfig {
        let (xk, yk) = self.atlas.axis_keys();
        MyersonAuctionPresetConfig {
            uniform_min: self.uniform_min,
            uniform_max: self.uniform_max,
            exp_rate: self.exp_rate,
            beta_alpha: self.beta_alpha,
            beta_beta: self.beta_beta,
            distribution: self.distribution,
            rule_mode: self.rule_mode,
            plot_view: self.plot_view,
            n_bidders: self.n_bidders,
            reserve_price: self.reserve_price,
            focal_valuation: self.focal_valuation,
            candidate_bid: self.candidate_bid,
            mc_samples: self.mc_samples,
            mc_seed: self.mc_seed,
            snapshot_manager: if include_snapshots {
                self.snapshot_manager.clone()
            } else {
                TraceSnapshotManager::new()
            },
            atlas_visible: self.atlas.is_visible(),
            atlas_axes: [xk.to_string(), yk.to_string()],
        }
    }

    pub(crate) fn apply_config(&mut self, cfg: MyersonAuctionPresetConfig) -> Result<(), String> {
        self.validate_raw_config(&cfg)?;
        self.atlas
            .set_axis_keys(&cfg.atlas_axes[0], &cfg.atlas_axes[1])?;
        self.atlas.set_visible(cfg.atlas_visible);
        self.uniform_min = cfg.uniform_min;
        self.uniform_max = cfg.uniform_max;
        self.exp_rate = cfg.exp_rate;
        self.beta_alpha = cfg.beta_alpha;
        self.beta_beta = cfg.beta_beta;
        self.distribution = cfg.distribution;
        self.rule_mode = cfg.rule_mode;
        self.plot_view = cfg.plot_view;
        self.n_bidders = cfg.n_bidders;
        self.reserve_price = cfg.reserve_price;
        self.focal_valuation = cfg.focal_valuation;
        self.candidate_bid = cfg.candidate_bid;
        self.mc_samples = cfg.mc_samples;
        self.mc_seed = cfg.mc_seed;
        self.snapshot_manager = cfg.snapshot_manager;
        self.recalculate();
        Ok(())
    }

    fn validate_raw_config(&self, cfg: &MyersonAuctionPresetConfig) -> Result<(), String> {
        let floats = [
            cfg.uniform_min,
            cfg.uniform_max,
            cfg.exp_rate,
            cfg.beta_alpha,
            cfg.beta_beta,
            cfg.reserve_price,
            cfg.focal_valuation,
            cfg.candidate_bid,
        ];
        if !floats.into_iter().all(f64::is_finite) {
            return Err("MyersonAuctionTool parameters must be finite".to_string());
        }
        if cfg.uniform_min < 0.0 || cfg.uniform_max <= cfg.uniform_min || cfg.uniform_max > 500.0 {
            return Err("Uniform bounds must satisfy 0 <= uniform_min < uniform_max <= 500".into());
        }
        if !(0.05..=5.0).contains(&cfg.exp_rate)
            || !(0.5..=8.0).contains(&cfg.beta_alpha)
            || !(0.5..=8.0).contains(&cfg.beta_beta)
            || !(1..=20).contains(&cfg.n_bidders)
            || !(100..=25_000).contains(&cfg.mc_samples)
            || cfg.snapshot_manager.len() > crate::framework::MAX_SNAPSHOTS
        {
            return Err("MyersonAuctionTool parameter out of valid bounds".to_string());
        }
        Ok(())
    }
}

impl TheoryDescribable for MyersonAuctionTool {
    fn theory_description(&self) -> String {
        format!(
            "Myerson (1981) optimal mechanism design & Revenue Equivalence: virtual valuation $J(v) = v - \\frac{{1 - F(v)}}{{f(v)}}$ sets the seller-optimal reserve $r^* = {:.3}$ at $J(r^*) = 0$, while symmetric First-Price Sealed-Bid equilibrium bids shade to $b^*(v) = v - \\frac{{\\int_r^v F(x)^{{n-1}}\\,dx}}{{F(v)^{{n-1}}}}$.",
            self.summary.optimal_reserve
        )
    }

    fn phonetic_description(&self) -> String {
        "Myerson optimal auction virtual valuation and Bayes-Nash bid shading laboratory".into()
    }

    fn theory_citation(&self) -> String {
        "Optimal Auction Design (Myerson, Mathematics of Operations Research, 1981)".into()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        let (v_min, v_max) = self.dist_eval.plot_bounds();
        let step = ((v_max - v_min) / 200.0).max(1e-3);
        [
            ("reserve_price", v_min, v_max, step),
            ("n_bidders", 1.0, 20.0, 1.0),
            ("focal_valuation", v_min, v_max, step),
            ("candidate_bid", 0.0, v_max, step),
        ]
        .into_iter()
        .map(|(k, min, max, s)| (k.to_string(), ParameterConstraint { min, max, step: s }))
        .collect()
    }
}

const PRESET_UNIFORM_RESERVE_JSON: &str = r#"{
  "metadata": { "schema_version": "1.0", "title": "Myerson Optimal Reserve & Exclusion", "domain": "game_theory", "timestamp": "2026-11-23T00:00:00Z" },
  "config": {
    "uniform_min": 0.0, "uniform_max": 100.0, "exp_rate": 0.1, "beta_alpha": 2.0, "beta_beta": 2.0,
    "distribution": "Uniform", "rule_mode": "SecondPrice", "plot_view": "VirtualValuation",
    "n_bidders": 1, "reserve_price": 50.0, "focal_valuation": 80.0, "candidate_bid": 80.0
  }
}"#;

const PRESET_FIRST_PRICE_SHADING_JSON: &str = r#"{
  "metadata": { "schema_version": "1.0", "title": "First-Price Bayes-Nash Bid Shading", "domain": "game_theory", "timestamp": "2026-11-23T00:00:00Z" },
  "config": {
    "uniform_min": 0.0, "uniform_max": 100.0, "exp_rate": 0.1, "beta_alpha": 2.0, "beta_beta": 2.0,
    "distribution": "Uniform", "rule_mode": "FirstPrice", "plot_view": "BidFunctions",
    "n_bidders": 2, "reserve_price": 50.0, "focal_valuation": 100.0, "candidate_bid": 62.5
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "myerson_uniform_optimal_reserve",
        title: "Myerson Optimal Reserve & Deadweight Loss",
        description: "J(v) = 2v - 100 on Uniform(0, 100): r* = 50 maximizes expected revenue (25.0 for n = 1) while deliberately blocking 50% of positive-value trades.",
        category: "Mechanism Design",
        icon: "🏛",
        domain: "game_theory",
        tool_name: Some("MyersonAuctionTool"),
        preset_generator: || PRESET_UNIFORM_RESERVE_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "myerson_first_price_bid_shading",
        title: "First-Price Bayes-Nash Bid Shading",
        description: "Revenue Equivalence in action: compare dominant-strategy Vickrey bidding b(v) = v against symmetric First-Price shading b*(100; 50, 2) = 62.5.",
        category: "Mechanism Design",
        icon: "🎯",
        domain: "game_theory",
        tool_name: Some("MyersonAuctionTool"),
        preset_generator: || PRESET_FIRST_PRICE_SHADING_JSON,
    }
}
