//! Serializable configuration state for [`super::BankrollGrowthTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::bankroll_ensemble::{
    BankrollLabMode, VolatilityHypothesis, DEFAULT_BARRIER_FLOOR_RATIO, DEFAULT_CUSTOM_FRACTION,
    MAX_BARRIER_FLOOR_RATIO, MAX_CUSTOM_FRACTION, MIN_BARRIER_FLOOR_RATIO,
};
use math_commons::primitives::{NonNegativeFloat, PositiveFloat, UnitInterval};

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

pub(super) const DEFAULT_STAKE_MULTIPLIER: f64 = 1.0;
pub(super) const DEFAULT_PUSH_PROBABILITY: f64 = 0.0;
pub(super) const DEFAULT_HALF_WIN_PROBABILITY: f64 = 0.12;
pub(super) const DEFAULT_HALF_LOSS_PROBABILITY: f64 = 0.0;
pub(super) const DEFAULT_SLATE_SIZE: usize = 5;
pub(super) const DEFAULT_DRAWDOWN_ALPHA: f64 = 0.70;
pub(super) const DEFAULT_DRAWDOWN_BETA: f64 = 0.10;

fn default_stake_multiplier() -> NonNegativeFloat {
    NonNegativeFloat::new(DEFAULT_STAKE_MULTIPLIER).expect("valid default stake multiplier")
}
fn default_custom_fraction() -> NonNegativeFloat {
    NonNegativeFloat::new(DEFAULT_CUSTOM_FRACTION).expect("valid default custom fraction")
}
fn default_barrier_floor_ratio() -> UnitInterval {
    UnitInterval::new(DEFAULT_BARRIER_FLOOR_RATIO).expect("valid default barrier floor ratio")
}
fn default_push_prob() -> UnitInterval {
    UnitInterval::new(DEFAULT_PUSH_PROBABILITY).expect("valid default push probability")
}
fn default_hw_prob() -> UnitInterval {
    UnitInterval::new(DEFAULT_HALF_WIN_PROBABILITY).expect("valid default half-win probability")
}
fn default_hl_prob() -> UnitInterval {
    UnitInterval::new(DEFAULT_HALF_LOSS_PROBABILITY).expect("valid default half-loss probability")
}
const fn default_slate_size() -> usize {
    DEFAULT_SLATE_SIZE
}
fn default_alpha() -> UnitInterval {
    UnitInterval::new(DEFAULT_DRAWDOWN_ALPHA).expect("valid default drawdown alpha")
}
fn default_beta() -> UnitInterval {
    UnitInterval::new(DEFAULT_DRAWDOWN_BETA).expect("valid default drawdown beta")
}

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}
fn is_default_stake_multiplier(val: &NonNegativeFloat) -> bool {
    approx_eq(val.value(), DEFAULT_STAKE_MULTIPLIER)
}
fn is_default_mode(mode: &BankrollLabMode) -> bool {
    *mode == BankrollLabMode::KellyStrategies
}
fn is_default_fraction(val: &NonNegativeFloat) -> bool {
    approx_eq(val.value(), DEFAULT_CUSTOM_FRACTION)
}
fn is_default_barrier_floor_ratio(val: &UnitInterval) -> bool {
    approx_eq(val.value(), DEFAULT_BARRIER_FLOOR_RATIO)
}
fn is_default_wager_model(m: &BankrollWagerModel) -> bool {
    *m == BankrollWagerModel::BinaryOrPush
}
fn is_default_push_prob(v: &UnitInterval) -> bool {
    approx_eq(v.value(), DEFAULT_PUSH_PROBABILITY)
}
fn is_default_hw_prob(v: &UnitInterval) -> bool {
    approx_eq(v.value(), DEFAULT_HALF_WIN_PROBABILITY)
}
fn is_default_hl_prob(v: &UnitInterval) -> bool {
    approx_eq(v.value(), DEFAULT_HALF_LOSS_PROBABILITY)
}
fn is_default_slate_size(c: &usize) -> bool {
    *c == DEFAULT_SLATE_SIZE
}
fn is_default_alpha(v: &UnitInterval) -> bool {
    approx_eq(v.value(), DEFAULT_DRAWDOWN_ALPHA)
}
fn is_default_beta(v: &UnitInterval) -> bool {
    approx_eq(v.value(), DEFAULT_DRAWDOWN_BETA)
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BankrollState {
    pub initial_bankroll: PositiveFloat,
    pub probability: UnitInterval,
    pub odds: PositiveFloat,
    pub num_bets: usize,
    #[serde(
        default = "default_stake_multiplier",
        skip_serializing_if = "is_default_stake_multiplier"
    )]
    pub stake_multiplier: NonNegativeFloat,
    #[serde(default, skip_serializing_if = "is_default_mode")]
    pub mode: BankrollLabMode,
    #[serde(
        default = "default_custom_fraction",
        skip_serializing_if = "is_default_fraction"
    )]
    pub custom_fraction: NonNegativeFloat,
    #[serde(
        default = "default_barrier_floor_ratio",
        skip_serializing_if = "is_default_barrier_floor_ratio"
    )]
    pub barrier_floor_ratio: UnitInterval,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hypothesis: Option<VolatilityHypothesis>,
    #[serde(default, skip_serializing_if = "is_default_wager_model")]
    pub wager_model: BankrollWagerModel,
    #[serde(
        default = "default_push_prob",
        skip_serializing_if = "is_default_push_prob"
    )]
    pub push_probability: UnitInterval,
    #[serde(
        default = "default_hw_prob",
        skip_serializing_if = "is_default_hw_prob"
    )]
    pub half_win_probability: UnitInterval,
    #[serde(
        default = "default_hl_prob",
        skip_serializing_if = "is_default_hl_prob"
    )]
    pub half_loss_probability: UnitInterval,
    #[serde(
        default = "default_slate_size",
        skip_serializing_if = "is_default_slate_size"
    )]
    pub slate_size: usize,
    #[serde(default = "default_alpha", skip_serializing_if = "is_default_alpha")]
    pub drawdown_alpha: UnitInterval,
    #[serde(default = "default_beta", skip_serializing_if = "is_default_beta")]
    pub drawdown_beta: UnitInterval,
}

impl BankrollState {
    pub(super) fn validate_bounds(&self) -> Result<(), String> {
        if !(100.0..=10_000.0).contains(&self.initial_bankroll.value())
            || !(0.01..=0.99).contains(&self.probability.value())
            || !(0.1..=10.0).contains(&self.odds.value())
            || !(10..=1000).contains(&self.num_bets)
            || self.stake_multiplier.value() > 3.0
            || self.custom_fraction.value() > MAX_CUSTOM_FRACTION
            || !(MIN_BARRIER_FLOOR_RATIO..=MAX_BARRIER_FLOOR_RATIO)
                .contains(&self.barrier_floor_ratio.value())
            || self.push_probability.value() > 0.80
            || self.half_win_probability.value() > 0.60
            || self.half_loss_probability.value() > 0.60
            || !(2..=32).contains(&self.slate_size)
            || !(0.05..=0.99).contains(&self.drawdown_alpha.value())
            || !(0.001..=0.99).contains(&self.drawdown_beta.value())
        {
            return Err("Bankroll simulation parameters out of bounds".into());
        }
        Ok(())
    }
}

// [cite:clinical_trials_statistics]
