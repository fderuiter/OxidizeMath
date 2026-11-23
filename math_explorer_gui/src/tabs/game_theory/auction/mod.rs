//! Interactive Myerson Optimal Auction & Virtual Valuation Workbench (`OptimalAuctionTool`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_explorer::applied::game_theory::mechanism_design::{
    distribution::{Exp, Uniform},
    optimal_reserve_price, simulate_optimal_revenue_with_rng, ValuationDistribution,
};
use oxidize_core::rng::OxidizeRng;
use serde::{Deserialize, Serialize};

pub mod presets;
mod ui;

#[cfg(test)]
mod tests;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuctionDistKind {
    Uniform,
    Exponential,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AuctionConfig {
    /// Upper valuation bound or plot horizon (first scalar float for perturbation tests).
    pub upper_bound: f64,
    pub lower_bound: f64,
    pub exp_rate: f64,
    pub dist_kind: AuctionDistKind,
    pub n_bidders: usize,
    pub custom_reserve: f64,
    pub mc_trials: usize,
    pub seed: u64,
}

impl AuctionConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.upper_bound.is_finite() || !(0.5..=1e4).contains(&self.upper_bound) {
            return Err("upper_bound must be finite and in [0.5, 10000]".into());
        }
        if !self.lower_bound.is_finite()
            || self.lower_bound < 0.0
            || self.lower_bound + 0.1 >= self.upper_bound
        {
            return Err("lower_bound must be >= 0 and < upper_bound - 0.1".into());
        }
        if !self.exp_rate.is_finite() || !(0.02..=10.0).contains(&self.exp_rate) {
            return Err("exp_rate must be finite and in [0.02, 10.0]".into());
        }
        if !(1..=32).contains(&self.n_bidders) {
            return Err("n_bidders must be in 1..=32".into());
        }
        if !self.custom_reserve.is_finite() || !(0.0..=1e4).contains(&self.custom_reserve) {
            return Err("custom_reserve must be finite and in [0, 10000]".into());
        }
        if !(200..=20_000).contains(&self.mc_trials) {
            return Err("mc_trials must be in 200..=20000".into());
        }
        Ok(())
    }
}

pub struct OptimalAuctionTool {
    pub(crate) cfg: AuctionConfig,
    pub(crate) optimal_reserve: f64,
    pub(crate) mc_optimal_revenue: f64,
    pub(crate) no_reserve_revenue: f64,
    pub(crate) custom_reserve_revenue: f64,
    pub(crate) virtual_curve: Vec<[f64; 2]>,
    pub(crate) revenue_curve: Vec<[f64; 2]>,
    state_rev: u64,
}

impl Default for OptimalAuctionTool {
    fn default() -> Self {
        let mut tool = Self {
            cfg: presets::default_config(),
            optimal_reserve: 0.0,
            mc_optimal_revenue: 0.0,
            no_reserve_revenue: 0.0,
            custom_reserve_revenue: 0.0,
            virtual_curve: Vec::new(),
            revenue_curve: Vec::new(),
            state_rev: 1,
        };
        tool.recompute();
        tool
    }
}

/// Exact expected second-price revenue with reserve `r` via numerical quadrature:
/// $R(r) = r \cdot n (1 - F(r)) F(r)^{n-1} + \int_r^{b} (1 - F(v)^n - n(1 - F(v))F(v)^{n-1}) \, dv$.
/// Using integration by parts on order statistics:
/// $\mathbb{E}[\text{Rev}(r)] = r (1 - F(r)^n) + \int_r^{b} (1 - F(v)^n - n(1 - F(v))F(v)^{n-1}) \, dv$
/// simplifies cleanly via Myerson's lemma to $\int_r^b J(v) n F(v)^{n-1} f(v) \, dv$!
fn expected_revenue_at_reserve<D: ValuationDistribution>(
    dist: &D,
    reserve: f64,
    upper: f64,
    n_bidders: usize,
) -> f64 {
    if reserve >= upper {
        return 0.0;
    }
    let steps = 160usize;
    let h = (upper - reserve) / (steps as f64);
    let n_f = n_bidders as f64;
    let integrand = |v: f64| -> f64 {
        let f_cdf = dist.cdf(v).clamp(0.0, 1.0);
        let f_pdf = dist.pdf(v).max(0.0);
        dist.virtual_valuation(v) * n_f * f_cdf.powi((n_bidders as i32) - 1) * f_pdf
    };
    let mut sum = 0.5 * (integrand(reserve) + integrand(upper));
    for k in 1..steps {
        sum += integrand(reserve + (k as f64) * h);
    }
    (sum * h).max(0.0)
}

impl OptimalAuctionTool {
    pub(crate) fn recompute(&mut self) {
        self.cfg.custom_reserve = self.cfg.custom_reserve.clamp(0.0, self.cfg.upper_bound);
        match self.cfg.dist_kind {
            AuctionDistKind::Uniform => {
                if let Ok(dist) = Uniform::new(self.cfg.lower_bound, self.cfg.upper_bound) {
                    self.recompute_for_dist(&dist, self.cfg.lower_bound, self.cfg.upper_bound);
                }
            }
            AuctionDistKind::Exponential => {
                if let Ok(dist) = Exp::new(self.cfg.exp_rate) {
                    let horizon = self.cfg.upper_bound.max(8.0 / self.cfg.exp_rate);
                    self.recompute_for_dist(&dist, 0.0, horizon);
                }
            }
        }
        self.state_rev = self.state_rev.wrapping_add(1).max(1);
    }

    fn recompute_for_dist<D: ValuationDistribution>(&mut self, dist: &D, lo: f64, quad_hi: f64) {
        let plot_hi = self.cfg.upper_bound;
        self.optimal_reserve = optimal_reserve_price(dist, lo, quad_hi);
        let mut rng = OxidizeRng::from_seed(self.cfg.seed);
        self.mc_optimal_revenue = simulate_optimal_revenue_with_rng(
            dist,
            self.cfg.n_bidders,
            self.cfg.mc_trials,
            &mut rng,
        );
        self.no_reserve_revenue =
            expected_revenue_at_reserve(dist, lo, quad_hi, self.cfg.n_bidders);
        self.custom_reserve_revenue = expected_revenue_at_reserve(
            dist,
            self.cfg.custom_reserve.max(lo),
            quad_hi,
            self.cfg.n_bidders,
        );

        let samples = 80usize;
        self.virtual_curve.clear();
        self.revenue_curve.clear();
        for i in 0..=samples {
            let frac = (i as f64) / (samples as f64);
            let v = lo + frac * (plot_hi - lo);
            self.virtual_curve.push([v, dist.virtual_valuation(v)]);
            let rev = expected_revenue_at_reserve(dist, v, quad_hi, self.cfg.n_bidders);
            self.revenue_curve.push([v, rev]);
        }
    }
}

impl InteractiveTool for OptimalAuctionTool {
    fn name(&self) -> &'static str {
        "Myerson Optimal Auctions"
    }

    fn domain(&self) -> &'static str {
        "game_theory"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn state_revision(&self) -> u64 {
        self.state_rev
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::new(1024, 2, 160)
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("optimal_reserve", self.optimal_reserve),
            ("mc_optimal_revenue", self.mc_optimal_revenue),
            ("no_reserve_revenue", self.no_reserve_revenue),
            ("custom_reserve_revenue", self.custom_reserve_revenue),
        ]
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.cfg).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let cfg: AuctionConfig = serde_json::from_str(state)
            .map_err(|e| format!("Invalid OptimalAuctionTool state: {e}"))?;
        cfg.validate()?;
        self.cfg = cfg;
        self.recompute();
        Ok(())
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        vec![
            (
                "Virtual Valuation J(v)".to_string(),
                self.virtual_curve.clone(),
            ),
            (
                "Expected Revenue R(r)".to_string(),
                self.revenue_curve.clone(),
            ),
        ]
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let side_id = crate::framework::scoped_panel_id(parent_ui, "optimal_auction_side");
        crate::framework::PaneSidePanel::left(side_id).show_inside(parent_ui, |ui| {
            self.show_controls(ui);
        });
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.show_main_view(ui);
        });
    }
}

impl scientific_metadata::theory::TheoryDescribable for OptimalAuctionTool {
    fn theory_description(&self) -> String {
        "Designs revenue-maximizing single-item auctions via Myerson's Lemma: bidders with i.i.d. valuations $v \\sim F$ induce virtual valuations $J(v) = v - \\frac{1 - F(v)}{f(v)}$, yielding the bidder-count-invariant optimal reserve price $r^* = J^{-1}(0)$ and expected seller revenue $\\mathbb{E}[\\max(0, J(v_1), \\dots, J(v_n))]$.".into()
    }

    fn phonetic_description(&self) -> String {
        "Myerson optimal auction virtual valuation reserve price and revenue equivalence workbench"
            .into()
    }

    fn theory_citation(&self) -> String {
        "Optimal Auction Design (Myerson, Mathematics of Operations Research, 1981)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "OptimalAuctionTool",
        display_name: "Myerson Optimal Auctions",
        domain: "game_theory",
        tags: &["auction", "myerson", "mechanism-design", "reserve-price", "revenue"],
        build: || Box::new(OptimalAuctionTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "auction_uniform_myerson_reserve",
        title: "Single-Bidder Monopoly Reserve Surge",
        description: "v ~ Uniform[0, 200], n = 1: Optimal reserve r* = 100 lifts expected revenue from 0 to 50.",
        category: "Game Theory",
        icon: "🔨",
        domain: "game_theory",
        tool_name: Some("OptimalAuctionTool"),
        preset_generator: || presets::PRESET_SINGLE_BIDDER_MONOPOLY,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "auction_exponential_constant_hazard",
        title: "Constant-Hazard Exponential Auction",
        description: "v ~ Exp(λ = 0.25): Constant hazard rate yields linear J(v) = v − 4 and reserve r* = 4.0.",
        category: "Game Theory",
        icon: "📈",
        domain: "game_theory",
        tool_name: Some("OptimalAuctionTool"),
        preset_generator: || presets::PRESET_EXPONENTIAL_HAZARD,
    }
}
