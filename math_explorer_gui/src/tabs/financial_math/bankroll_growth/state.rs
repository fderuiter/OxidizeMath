//! Serializable configuration state for [`super::BankrollGrowthTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::bankroll_ensemble::{
    BankrollLabMode, VolatilityHypothesis, DEFAULT_CUSTOM_FRACTION, MAX_CUSTOM_FRACTION,
};

/// Active wager settlement model inside [`super::BankrollGrowthTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BankrollWagerModel {
    /// Two-state binary (`push_probability == 0`) or three-state push refund wager.
    #[default]
    BinaryOrPush,
    /// Five-state Asian handicap quarter-line split wager (`+b, +b/2, 0, -1/2, -1`).
    QuarterLine,
    /// Simultaneous round of `slate_size` independent wagers settled per step.
    SimultaneousSlate,
}

impl BankrollWagerModel {
    pub(super) const ALL: [Self; 3] = [
        Self::BinaryOrPush,
        Self::QuarterLine,
        Self::SimultaneousSlate,
    ];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::BinaryOrPush => "Binary / Push",
            Self::QuarterLine => "Quarter-Line",
            Self::SimultaneousSlate => "Simultaneous Slate",
        }
    }
}

pub(super) const DEFAULT_PUSH_PROBABILITY: f64 = 0.0;
pub(super) const DEFAULT_HALF_WIN_PROBABILITY: f64 = 0.12;
pub(super) const DEFAULT_HALF_LOSS_PROBABILITY: f64 = 0.0;
pub(super) const DEFAULT_SLATE_SIZE: usize = 5;
pub(super) const DEFAULT_DRAWDOWN_ALPHA: f64 = 0.70;
pub(super) const DEFAULT_DRAWDOWN_BETA: f64 = 0.10;

const fn default_custom_fraction() -> f64 {
    DEFAULT_CUSTOM_FRACTION
}
const fn default_push_prob() -> f64 {
    DEFAULT_PUSH_PROBABILITY
}
const fn default_hw_prob() -> f64 {
    DEFAULT_HALF_WIN_PROBABILITY
}
const fn default_hl_prob() -> f64 {
    DEFAULT_HALF_LOSS_PROBABILITY
}
const fn default_slate_size() -> usize {
    DEFAULT_SLATE_SIZE
}
const fn default_alpha() -> f64 {
    DEFAULT_DRAWDOWN_ALPHA
}
const fn default_beta() -> f64 {
    DEFAULT_DRAWDOWN_BETA
}

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}
fn is_default_mode(mode: &BankrollLabMode) -> bool {
    *mode == BankrollLabMode::KellyStrategies
}
fn is_default_fraction(val: &f64) -> bool {
    approx_eq(*val, DEFAULT_CUSTOM_FRACTION)
}
fn is_default_wager_model(m: &BankrollWagerModel) -> bool {
    *m == BankrollWagerModel::BinaryOrPush
}
fn is_default_push_prob(v: &f64) -> bool {
    approx_eq(*v, DEFAULT_PUSH_PROBABILITY)
}
fn is_default_hw_prob(v: &f64) -> bool {
    approx_eq(*v, DEFAULT_HALF_WIN_PROBABILITY)
}
fn is_default_hl_prob(v: &f64) -> bool {
    approx_eq(*v, DEFAULT_HALF_LOSS_PROBABILITY)
}
fn is_default_slate_size(c: &usize) -> bool {
    *c == DEFAULT_SLATE_SIZE
}
fn is_default_alpha(v: &f64) -> bool {
    approx_eq(*v, DEFAULT_DRAWDOWN_ALPHA)
}
fn is_default_beta(v: &f64) -> bool {
    approx_eq(*v, DEFAULT_DRAWDOWN_BETA)
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BankrollState {
    pub initial_bankroll: f64,
    pub probability: f64,
    pub odds: f64,
    pub num_bets: usize,
    #[serde(default, skip_serializing_if = "is_default_mode")]
    pub mode: BankrollLabMode,
    #[serde(
        default = "default_custom_fraction",
        skip_serializing_if = "is_default_fraction"
    )]
    pub custom_fraction: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hypothesis: Option<VolatilityHypothesis>,
    #[serde(default, skip_serializing_if = "is_default_wager_model")]
    pub wager_model: BankrollWagerModel,
    #[serde(
        default = "default_push_prob",
        skip_serializing_if = "is_default_push_prob"
    )]
    pub push_probability: f64,
    #[serde(
        default = "default_hw_prob",
        skip_serializing_if = "is_default_hw_prob"
    )]
    pub half_win_probability: f64,
    #[serde(
        default = "default_hl_prob",
        skip_serializing_if = "is_default_hl_prob"
    )]
    pub half_loss_probability: f64,
    #[serde(
        default = "default_slate_size",
        skip_serializing_if = "is_default_slate_size"
    )]
    pub slate_size: usize,
    #[serde(default = "default_alpha", skip_serializing_if = "is_default_alpha")]
    pub drawdown_alpha: f64,
    #[serde(default = "default_beta", skip_serializing_if = "is_default_beta")]
    pub drawdown_beta: f64,
}

impl BankrollState {
    pub(super) fn validate_bounds(&self) -> Result<(), String> {
        let scalars = [
            self.initial_bankroll,
            self.probability,
            self.odds,
            self.custom_fraction,
            self.push_probability,
            self.half_win_probability,
            self.half_loss_probability,
            self.drawdown_alpha,
            self.drawdown_beta,
        ];
        if !scalars.iter().all(|v| v.is_finite())
            || !(100.0..=10_000.0).contains(&self.initial_bankroll)
            || !(0.01..=0.99).contains(&self.probability)
            || !(0.1..=10.0).contains(&self.odds)
            || !(10..=1000).contains(&self.num_bets)
            || !(0.0..=MAX_CUSTOM_FRACTION).contains(&self.custom_fraction)
            || !(0.0..=0.80).contains(&self.push_probability)
            || !(0.0..=0.60).contains(&self.half_win_probability)
            || !(0.0..=0.60).contains(&self.half_loss_probability)
            || !(2..=32).contains(&self.slate_size)
            || !(0.05..=0.99).contains(&self.drawdown_alpha)
            || !(0.001..=0.99).contains(&self.drawdown_beta)
        {
            return Err("Bankroll simulation parameters out of bounds".into());
        }
        Ok(())
    }
}

// [cite:clinical_trials_statistics]
