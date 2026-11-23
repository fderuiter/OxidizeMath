//! Serializable configuration state and defaults for [`super::BetSizeCalculatorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

/// Supported sportsbook odds representations in [`super::BetSizeCalculatorTool`].
#[derive(Debug, PartialEq, Eq, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum OddsFormat {
    /// European decimal odds (total payout per unit stake, e.g. `2.00`).
    Decimal,
    /// Moneyline American odds (e.g. `+120` or `-110`).
    American,
    /// Fractional net profit odds (e.g. `10 / 11`).
    Fractional,
}

/// Active multi-state or portfolio Kelly solver mode in [`super::BetSizeCalculatorTool`].
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum BetSizingMode {
    /// Classic two-outcome binary wager (`+b` win, `-1` loss).
    #[default]
    Binary,
    /// Three-state integer spread / total wager with push refund (`+b`, `0`, `-1`).
    PushSpread,
    /// Five-state Asian handicap quarter-line split wager (`+b`, `+b/2`, `0`, `-1/2`, `-1`).
    QuarterLine,
    /// Mutually exclusive `N`-outcome market solved via Smoczynski-Tomkins active set.
    MutuallyExclusive,
    /// Simultaneous independent `K`-game slate solved via projected FISTA.
    SimultaneousSlate,
}

impl BetSizingMode {
    pub(super) const ALL: [Self; 5] = [
        Self::Binary,
        Self::PushSpread,
        Self::QuarterLine,
        Self::MutuallyExclusive,
        Self::SimultaneousSlate,
    ];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Binary => "Binary",
            Self::PushSpread => "3-State Push",
            Self::QuarterLine => "5-State Quarter-Line",
            Self::MutuallyExclusive => "N-Way Market",
            Self::SimultaneousSlate => "K-Game Slate",
        }
    }
}

/// Configurable row for a mutually exclusive market outcome in the UI.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MarketRowConfig {
    /// True outcome probability in percent (`0.0..=100.0`).
    pub probability_pct: f64,
    /// Gross decimal payout odds (`> 1.0`).
    pub decimal_odds: f64,
}

/// Maximum number of editable outcomes in the mutually exclusive market UI table.
pub const UI_MAX_MARKET_ROWS: usize = 8;

pub(super) const DEFAULT_PUSH_PCT: f64 = 10.0;
pub(super) const DEFAULT_QL_FULL_WIN_PCT: f64 = 44.0;
pub(super) const DEFAULT_QL_HALF_WIN_PCT: f64 = 12.0;
pub(super) const DEFAULT_QL_PUSH_PCT: f64 = 0.0;
pub(super) const DEFAULT_QL_HALF_LOSS_PCT: f64 = 0.0;
pub(super) const DEFAULT_QL_FULL_LOSS_PCT: f64 = 44.0;
pub(super) const DEFAULT_MARKET_COUNT: usize = 3;
pub(super) const DEFAULT_SLATE_COUNT: usize = 10;
pub(super) const DEFAULT_SLATE_WIN_PCT: f64 = 58.0;
pub(super) const DEFAULT_SLATE_PUSH_PCT: f64 = 0.0;
pub(super) const DEFAULT_SLATE_NET_ODDS: f64 = 1.0;
pub(super) const DEFAULT_DRAWDOWN_ALPHA: f64 = 0.70;
pub(super) const DEFAULT_DRAWDOWN_BETA: f64 = 0.10;

#[must_use]
pub(super) const fn default_market_rows() -> [MarketRowConfig; UI_MAX_MARKET_ROWS] {
    [
        MarketRowConfig {
            probability_pct: 50.0,
            decimal_odds: 2.20,
        },
        MarketRowConfig {
            probability_pct: 28.0,
            decimal_odds: 4.00,
        },
        MarketRowConfig {
            probability_pct: 22.0,
            decimal_odds: 3.20,
        },
        MarketRowConfig {
            probability_pct: 10.0,
            decimal_odds: 8.00,
        },
        MarketRowConfig {
            probability_pct: 10.0,
            decimal_odds: 8.00,
        },
        MarketRowConfig {
            probability_pct: 10.0,
            decimal_odds: 8.00,
        },
        MarketRowConfig {
            probability_pct: 10.0,
            decimal_odds: 8.00,
        },
        MarketRowConfig {
            probability_pct: 10.0,
            decimal_odds: 8.00,
        },
    ]
}

const fn default_push_pct() -> f64 {
    DEFAULT_PUSH_PCT
}
const fn default_ql_fw() -> f64 {
    DEFAULT_QL_FULL_WIN_PCT
}
const fn default_ql_hw() -> f64 {
    DEFAULT_QL_HALF_WIN_PCT
}
const fn default_ql_push() -> f64 {
    DEFAULT_QL_PUSH_PCT
}
const fn default_ql_hl() -> f64 {
    DEFAULT_QL_HALF_LOSS_PCT
}
const fn default_ql_fl() -> f64 {
    DEFAULT_QL_FULL_LOSS_PCT
}
const fn default_market_count() -> usize {
    DEFAULT_MARKET_COUNT
}
const fn default_slate_count() -> usize {
    DEFAULT_SLATE_COUNT
}
const fn default_slate_win() -> f64 {
    DEFAULT_SLATE_WIN_PCT
}
const fn default_slate_push() -> f64 {
    DEFAULT_SLATE_PUSH_PCT
}
const fn default_slate_odds() -> f64 {
    DEFAULT_SLATE_NET_ODDS
}
const fn default_alpha() -> f64 {
    DEFAULT_DRAWDOWN_ALPHA
}
const fn default_beta() -> f64 {
    DEFAULT_DRAWDOWN_BETA
}

fn is_default_mode(m: &BetSizingMode) -> bool {
    *m == BetSizingMode::Binary
}
fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}
fn is_default_push(v: &f64) -> bool {
    approx_eq(*v, DEFAULT_PUSH_PCT)
}
fn is_default_ql_fw(v: &f64) -> bool {
    approx_eq(*v, DEFAULT_QL_FULL_WIN_PCT)
}
fn is_default_ql_hw(v: &f64) -> bool {
    approx_eq(*v, DEFAULT_QL_HALF_WIN_PCT)
}
fn is_default_ql_push(v: &f64) -> bool {
    approx_eq(*v, DEFAULT_QL_PUSH_PCT)
}
fn is_default_ql_hl(v: &f64) -> bool {
    approx_eq(*v, DEFAULT_QL_HALF_LOSS_PCT)
}
fn is_default_ql_fl(v: &f64) -> bool {
    approx_eq(*v, DEFAULT_QL_FULL_LOSS_PCT)
}
fn is_default_market_count(c: &usize) -> bool {
    *c == DEFAULT_MARKET_COUNT
}
fn is_default_market_rows(rows: &[MarketRowConfig; UI_MAX_MARKET_ROWS]) -> bool {
    *rows == default_market_rows()
}
fn is_default_slate_count(c: &usize) -> bool {
    *c == DEFAULT_SLATE_COUNT
}
fn is_default_slate_win(v: &f64) -> bool {
    approx_eq(*v, DEFAULT_SLATE_WIN_PCT)
}
fn is_default_slate_push(v: &f64) -> bool {
    approx_eq(*v, DEFAULT_SLATE_PUSH_PCT)
}
fn is_default_slate_odds(v: &f64) -> bool {
    approx_eq(*v, DEFAULT_SLATE_NET_ODDS)
}
fn is_default_alpha(v: &f64) -> bool {
    approx_eq(*v, DEFAULT_DRAWDOWN_ALPHA)
}
fn is_default_beta(v: &f64) -> bool {
    approx_eq(*v, DEFAULT_DRAWDOWN_BETA)
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BetSizeState {
    pub probability_input: f64,
    pub odds_format: OddsFormat,
    pub decimal_odds: f64,
    pub american_odds: f64,
    pub fractional_numerator: f64,
    pub fractional_denominator: f64,
    pub bankroll: f64,
    #[serde(default, skip_serializing_if = "is_default_mode")]
    pub mode: BetSizingMode,
    #[serde(default = "default_push_pct", skip_serializing_if = "is_default_push")]
    pub push_probability_pct: f64,
    #[serde(default = "default_ql_fw", skip_serializing_if = "is_default_ql_fw")]
    pub ql_full_win_pct: f64,
    #[serde(default = "default_ql_hw", skip_serializing_if = "is_default_ql_hw")]
    pub ql_half_win_pct: f64,
    #[serde(
        default = "default_ql_push",
        skip_serializing_if = "is_default_ql_push"
    )]
    pub ql_push_pct: f64,
    #[serde(default = "default_ql_hl", skip_serializing_if = "is_default_ql_hl")]
    pub ql_half_loss_pct: f64,
    #[serde(default = "default_ql_fl", skip_serializing_if = "is_default_ql_fl")]
    pub ql_full_loss_pct: f64,
    #[serde(
        default = "default_market_count",
        skip_serializing_if = "is_default_market_count"
    )]
    pub market_outcome_count: usize,
    #[serde(
        default = "default_market_rows",
        skip_serializing_if = "is_default_market_rows"
    )]
    pub market_rows: [MarketRowConfig; UI_MAX_MARKET_ROWS],
    #[serde(
        default = "default_slate_count",
        skip_serializing_if = "is_default_slate_count"
    )]
    pub slate_count: usize,
    #[serde(
        default = "default_slate_win",
        skip_serializing_if = "is_default_slate_win"
    )]
    pub slate_win_pct: f64,
    #[serde(
        default = "default_slate_push",
        skip_serializing_if = "is_default_slate_push"
    )]
    pub slate_push_pct: f64,
    #[serde(
        default = "default_slate_odds",
        skip_serializing_if = "is_default_slate_odds"
    )]
    pub slate_net_odds: f64,
    #[serde(default = "default_alpha", skip_serializing_if = "is_default_alpha")]
    pub drawdown_alpha: f64,
    #[serde(default = "default_beta", skip_serializing_if = "is_default_beta")]
    pub drawdown_beta: f64,
}

impl BetSizeState {
    pub(super) fn validate(&self) -> Result<(), String> {
        let scalars = [
            self.probability_input,
            self.decimal_odds,
            self.american_odds,
            self.fractional_numerator,
            self.fractional_denominator,
            self.bankroll,
            self.push_probability_pct,
            self.ql_full_win_pct,
            self.ql_half_win_pct,
            self.ql_push_pct,
            self.ql_half_loss_pct,
            self.ql_full_loss_pct,
            self.slate_win_pct,
            self.slate_push_pct,
            self.slate_net_odds,
            self.drawdown_alpha,
            self.drawdown_beta,
        ];
        if !scalars.iter().all(|v| v.is_finite())
            || !(0.0..=100.0).contains(&self.probability_input)
            || !(1.01..=1000.0).contains(&self.decimal_odds)
            || !(-10_000.0..=10_000.0).contains(&self.american_odds)
            || !(1.0..=1000.0).contains(&self.fractional_numerator)
            || !(1.0..=1000.0).contains(&self.fractional_denominator)
            || !(1.0..=1_000_000.0).contains(&self.bankroll)
            || !self.validate_extended_bounds()
        {
            return Err("Bet size parameters out of bounds".into());
        }
        Ok(())
    }

    fn validate_extended_bounds(&self) -> bool {
        let pct_fields = [
            self.push_probability_pct,
            self.ql_full_win_pct,
            self.ql_half_win_pct,
            self.ql_push_pct,
            self.ql_half_loss_pct,
            self.ql_full_loss_pct,
            self.slate_win_pct,
            self.slate_push_pct,
        ];
        let rows_valid = self.market_rows.iter().all(|r| {
            r.probability_pct.is_finite()
                && r.decimal_odds.is_finite()
                && (0.0..=100.0).contains(&r.probability_pct)
                && (1.01..=1000.0).contains(&r.decimal_odds)
        });
        pct_fields.iter().all(|p| (0.0..=100.0).contains(p))
            && rows_valid
            && (2..=UI_MAX_MARKET_ROWS).contains(&self.market_outcome_count)
            && (1..=64).contains(&self.slate_count)
            && (0.05..=20.0).contains(&self.slate_net_odds)
            && (0.05..=0.99).contains(&self.drawdown_alpha)
            && (0.001..=0.99).contains(&self.drawdown_beta)
    }
}

// [cite:clinical_trials_statistics]
