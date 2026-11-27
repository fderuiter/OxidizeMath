//! Progressive 2D parameter regime cartographer for [`super::MyersonAuctionTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::math::{AuctionDistEval, AuctionRuleMode};
use super::MyersonAuctionTool;
use crate::async_sim::atlas::{
    atlas_scalar_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, AtlasTeleportEvent,
    RegimeCartographer,
};
use eframe::egui;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Supported `(x, y)` parameter axis pairs for [`AuctionCartographer`].
pub const AUCTION_ATLAS_AXIS_PAIRS: &[(&str, &str)] = &[
    ("reserve_price", "n_bidders"),
    ("reserve_price", "focal_valuation"),
];

/// Default serialized atlas axis keys for [`MyersonAuctionTool`].
#[must_use]
pub fn default_auction_atlas_axes() -> [String; 2] {
    ["reserve_price".to_string(), "n_bidders".to_string()]
}

/// Predicate omitting default `atlas_axes` during JSON serialization.
#[must_use]
pub fn is_default_auction_atlas_axes(axes: &[String; 2]) -> bool {
    axes[0] == "reserve_price" && axes[1] == "n_bidders"
}

/// Discrete mechanism-design regimes across the Myerson optimal reserve $r^*$.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AuctionRegime {
    /// Under-reserved ($r < r^*$): high trade frequency but leaves seller rent unextracted.
    UnderReserved = 0,
    /// Near Myerson optimal ($r \approx r^*$): maximizes expected virtual surplus $\max(0, J(v))$.
    MyersonOptimalBand = 1,
    /// Over-reserved exclusion ($r > r^*$): inefficiently blocks positive-virtual-valuation trades.
    OverpricedExclusion = 2,
}

impl AuctionRegime {
    /// Converts a raw `u8` code into [`Self`].
    #[inline]
    #[must_use]
    pub const fn from_u8(code: u8) -> Self {
        match code {
            0 => Self::UnderReserved,
            1 => Self::MyersonOptimalBand,
            _ => Self::OverpricedExclusion,
        }
    }

    /// Human-readable regime label.
    #[inline]
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::UnderReserved => "Under-Reserved (r < r*: Welfare-Heavy)",
            Self::MyersonOptimalBand => "Myerson Optimal Band (r ≈ r*)",
            Self::OverpricedExclusion => "Over-Reserved Exclusion (r > r*)",
        }
    }
}

fn auction_axis_spec(
    theory_params: &HashMap<String, ParameterConstraint>,
    key: &'static str,
    v_min: f64,
    v_max: f64,
) -> AtlasAxisSpec {
    let step = ((v_max - v_min) / 200.0).max(1e-3);
    match key {
        "n_bidders" => AtlasAxisSpec::from_theory(
            theory_params,
            "n_bidders",
            "Number of Bidders (n)",
            ParameterConstraint {
                min: 1.0,
                max: 20.0,
                step: 1.0,
            },
        ),
        "focal_valuation" => AtlasAxisSpec {
            key: "focal_valuation",
            label: "Bidder Valuation (v)",
            constraint: ParameterConstraint {
                min: v_min,
                max: v_max,
                step,
            },
        },
        _ => AtlasAxisSpec {
            key: "reserve_price",
            label: "Reserve Price (r)",
            constraint: ParameterConstraint {
                min: v_min,
                max: v_max,
                step,
            },
        },
    }
}

/// Zero-allocation 2D Myerson auction cartographer for [`MyersonAuctionTool`].
#[derive(Clone, Debug)]
pub struct AuctionCartographer {
    axes: AtlasAxisPair,
    dist: AuctionDistEval,
    optimal_r: f64,
    base_n_bidders: usize,
    rule: AuctionRuleMode,
    revenue_ref_scale: f64,
}

impl AuctionCartographer {
    /// Constructs an [`AuctionCartographer`] for the active valuation distribution and axis slice.
    #[must_use]
    pub fn new(
        theory_params: &HashMap<String, ParameterConstraint>,
        axis_keys: (&'static str, &'static str),
        dist: AuctionDistEval,
        base_n_bidders: usize,
        rule: AuctionRuleMode,
    ) -> Self {
        let (v_min, v_max) = dist.plot_bounds();
        let optimal_r = dist.optimal_reserve();
        let (max_rev, _, _, _) = dist.expected_revenue_and_welfare(optimal_r, 20);
        Self {
            axes: AtlasAxisPair {
                x: auction_axis_spec(theory_params, axis_keys.0, v_min, v_max),
                y: auction_axis_spec(theory_params, axis_keys.1, v_min, v_max),
            },
            dist,
            optimal_r,
            base_n_bidders: base_n_bidders.clamp(1, 20),
            rule,
            revenue_ref_scale: max_rev.max(1e-6),
        }
    }

    #[inline]
    fn classify_reserve(&self, r: f64) -> AuctionRegime {
        let band = (self.axes.x.span() * 0.03).max(1e-4);
        if (r - self.optimal_r).abs() <= band {
            AuctionRegime::MyersonOptimalBand
        } else if r < self.optimal_r {
            AuctionRegime::UnderReserved
        } else {
            AuctionRegime::OverpricedExclusion
        }
    }
}

impl RegimeCartographer for AuctionCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let r = self.axes.x.clamp(p1);
        if self.axes.y.key == "focal_valuation" {
            let v = self.axes.y.clamp(p2);
            let bid = match self.rule {
                AuctionRuleMode::SecondPrice => {
                    if v >= r {
                        v
                    } else {
                        0.0
                    }
                }
                AuctionRuleMode::FirstPrice => {
                    self.dist
                        .first_price_equilibrium_bid(v, r, self.base_n_bidders)
                }
            };
            let shade_ratio = if v > r && v.abs() > 1e-9 {
                ((v - bid) / v).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let regime = if v < r {
                AuctionRegime::OverpricedExclusion
            } else if (r - self.optimal_r).abs() <= (self.axes.x.span() * 0.03).max(1e-4) {
                AuctionRegime::MyersonOptimalBand
            } else {
                AuctionRegime::UnderReserved
            };
            return AtlasCell {
                regime: regime as u8,
                primary: bid,
                secondary: shade_ratio,
            };
        }

        let n = p2.round().clamp(1.0, 20.0) as usize;
        let (rev, welfare, eff_welfare, _trade_prob) = self.dist.expected_revenue_and_welfare(r, n);
        let efficiency = if eff_welfare > 1e-9 {
            (welfare / eff_welfare).clamp(0.0, 1.0)
        } else {
            1.0
        };
        AtlasCell {
            regime: self.classify_reserve(r) as u8,
            primary: rev,
            secondary: efficiency,
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let (_, v_max) = self.dist.plot_bounds();
        let norm = if self.axes.y.key == "focal_valuation" {
            (cell.primary / v_max.max(1e-6)).clamp(0.0, 1.0)
        } else {
            (cell.primary / self.revenue_ref_scale).clamp(0.0, 1.0)
        };
        let mapped = match AuctionRegime::from_u8(cell.regime) {
            AuctionRegime::OverpricedExclusion => 0.06 + 0.28 * norm,
            AuctionRegime::UnderReserved => 0.36 + 0.38 * norm,
            AuctionRegime::MyersonOptimalBand => 0.78 + 0.22 * norm,
        };
        atlas_scalar_color(mapped, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        AuctionRegime::from_u8(regime).label()
    }

    fn primary_metric_label(&self) -> &'static str {
        if self.axes.y.key == "focal_valuation" {
            "Equilibrium Bid b*(v)"
        } else {
            "Expected Seller Revenue"
        }
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        if self.axes.y.key == "focal_valuation" {
            format!(
                "r = {:.3}, v = {:.3} (r* = {:.3})\nRegime: {}\nb*(v) = {:.3}  |  Shading: {:.1}%",
                p1,
                p2,
                self.optimal_r,
                self.regime_label(cell.regime),
                cell.primary,
                cell.secondary * 100.0
            )
        } else {
            let n = p2.round().clamp(1.0, 20.0) as usize;
            format!(
                "r = {:.3}, n = {} (Myerson r* = {:.3})\nRegime: {}\nExpected Revenue: {:.3}  |  Welfare Efficiency: {:.1}%",
                p1,
                n,
                self.optimal_r,
                self.regime_label(cell.regime),
                cell.primary,
                cell.secondary * 100.0
            )
        }
    }

    fn overlay_contour(&self) -> Option<(&'static str, Vec<[f64; 2]>)> {
        let rx = self.axes.x.clamp(self.optimal_r);
        Some((
            "Myerson Optimal r*",
            vec![
                [rx, self.axes.y.constraint.min],
                [rx, self.axes.y.constraint.max],
            ],
        ))
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        vec![
            (
                AuctionRegime::OverpricedExclusion.label(),
                atlas_scalar_color(0.16, high_contrast),
            ),
            (
                AuctionRegime::UnderReserved.label(),
                atlas_scalar_color(0.55, high_contrast),
            ),
            (
                AuctionRegime::MyersonOptimalBand.label(),
                atlas_scalar_color(0.90, high_contrast),
            ),
        ]
    }
}

impl MyersonAuctionTool {
    /// Applies a 2D atlas teleport event, respecting any active challenge parameter locks.
    pub fn apply_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        let prev_summary = self.param_summary();
        let prev_series = self.current_trace_series();
        let (v_min, v_max) = self.dist_eval.plot_bounds();
        for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            match key {
                "reserve_price" if !self.is_reserve_locked() => {
                    self.reserve_price = val.clamp(v_min, v_max);
                }
                "n_bidders" if !self.is_env_locked() => {
                    self.n_bidders = val.round().clamp(1.0, 20.0) as usize;
                }
                "focal_valuation" if !self.is_env_locked() => {
                    self.focal_valuation = val.clamp(v_min, v_max);
                }
                _ => {}
            }
        }
        self.snapshot_manager
            .push_history_run(&prev_summary, prev_series);
        self.recalculate();
    }

    pub(super) fn render_auction_atlas_panel(&mut self, ui: &mut egui::Ui) {
        if !self.atlas.is_visible() {
            return;
        }
        let (xk, yk) = self.atlas.axis_keys();
        let get_val = |k: &str| match k {
            "reserve_price" => self.reserve_price,
            "n_bidders" => self.n_bidders as f64,
            "focal_valuation" => self.focal_valuation,
            _ => 0.0,
        };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.distribution.hash(&mut h);
        self.rule_mode.hash(&mut h);
        self.uniform_min.to_bits().hash(&mut h);
        self.uniform_max.to_bits().hash(&mut h);
        self.exp_rate.to_bits().hash(&mut h);
        self.beta_alpha.to_bits().hash(&mut h);
        self.beta_beta.to_bits().hash(&mut h);
        if yk == "focal_valuation" {
            self.n_bidders.hash(&mut h);
        }
        let theory_params = self.theory_parameters();
        let dist = self.dist_eval.clone();
        let base_n = self.n_bidders;
        let rule = self.rule_mode;
        if let Some(ev) = self.atlas.show_panel(
            ui,
            "myerson_auction",
            get_val(xk),
            get_val(yk),
            h.finish(),
            move |keys| {
                Arc::new(AuctionCartographer::new(
                    &theory_params,
                    keys,
                    dist,
                    base_n,
                    rule,
                ))
            },
        ) {
            self.apply_atlas_teleport(ev);
        }
    }
}
