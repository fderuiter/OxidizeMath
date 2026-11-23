//! Multi-outcome bookmaker devigging and `+EV` Kelly calculator for `FinancialMathTab`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod presets;
mod ui;

use crate::framework::{InteractiveTool, ToolMetadata};
use eframe::egui;
use math_explorer::pure_math::statistics::kelly::Odds;
use math_explorer::pure_math::statistics::market_pricing::DevigMethod;
use std::collections::HashMap;

pub(super) const MIN_OUTCOMES: usize = 2;
pub(super) const MAX_GUI_OUTCOMES: usize = 8;

/// Input display format for quoted sportsbook prices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum QuoteFormat {
    /// European decimal odds (`> 1.0`, including stake).
    Decimal,
    /// US moneyline odds (`<= -100` for favorites, `>= +100` for underdogs).
    American,
}

/// Interactive multi-way fair-odds devigger (`Multiplicative`, `Additive`, `Power`, `Shin`)
/// paired with soft-book `+EV` and fractional Kelly bet sizing.
pub struct MarketDeviggingTool {
    pub(super) bankroll: f64,
    pub(super) quote_format: QuoteFormat,
    pub(super) method: DevigMethod,
    pub(super) decimal_quotes: Vec<f64>,
    pub(super) target_outcome: usize,
    pub(super) offered_decimal_odds: f64,
}

impl Default for MarketDeviggingTool {
    fn default() -> Self {
        Self {
            bankroll: 1000.0,
            quote_format: QuoteFormat::Decimal,
            method: DevigMethod::Shin,
            decimal_quotes: vec![1.35, 4.80, 8.50],
            target_outcome: 2,
            offered_decimal_odds: 9.50,
        }
    }
}

impl MarketDeviggingTool {
    pub(super) fn apply_two_way_longshot_preset(&mut self) {
        self.quote_format = QuoteFormat::American;
        self.method = DevigMethod::Shin;
        self.decimal_quotes = vec![1.20, 4.50];
        self.target_outcome = 1;
        self.offered_decimal_odds = 5.00;
    }

    pub(super) fn apply_three_way_soccer_preset(&mut self) {
        self.quote_format = QuoteFormat::Decimal;
        self.method = DevigMethod::Shin;
        self.decimal_quotes = vec![1.35, 4.80, 8.50];
        self.target_outcome = 2;
        self.offered_decimal_odds = 9.50;
    }

    pub(super) fn apply_six_way_pod_preset(&mut self) {
        self.quote_format = QuoteFormat::Decimal;
        self.method = DevigMethod::Additive;
        self.decimal_quotes = vec![1.90, 3.25, 6.50, 10.0, 21.0, 101.0];
        self.target_outcome = 5;
        self.offered_decimal_odds = 110.0;
    }

    pub(super) fn build_odds_vec(&self) -> Result<Vec<Odds>, String> {
        let mut out = Vec::with_capacity(self.decimal_quotes.len());
        for &q in &self.decimal_quotes {
            let odds = Odds::new(q).map_err(|e| format!("Invalid market quote {q:.3}: {e}"))?;
            out.push(odds);
        }
        Ok(out)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MarketDevigState {
    bankroll: f64,
    quote_format: QuoteFormat,
    method: DevigMethod,
    decimal_quotes: Vec<f64>,
    target_outcome: usize,
    offered_decimal_odds: f64,
}

impl InteractiveTool for MarketDeviggingTool {
    fn name(&self) -> &'static str {
        "Fair Odds & +EV Devigger"
    }

    fn domain(&self) -> &'static str {
        "financial_math"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn has_finite_parameters(&self) -> bool {
        self.bankroll.is_finite()
            && self.offered_decimal_odds.is_finite()
            && self.decimal_quotes.iter().all(|v| v.is_finite())
    }

    fn clamp_parameters(&mut self) -> bool {
        self.bankroll = self.bankroll.clamp(1.0, 1_000_000.0);
        self.offered_decimal_odds = self.offered_decimal_odds.clamp(1.01, 1000.0);
        if self.decimal_quotes.len() < MIN_OUTCOMES {
            self.decimal_quotes.resize(MIN_OUTCOMES, 2.0);
        } else if self.decimal_quotes.len() > MAX_GUI_OUTCOMES {
            self.decimal_quotes.truncate(MAX_GUI_OUTCOMES);
        }
        for q in &mut self.decimal_quotes {
            if !q.is_finite() {
                *q = 2.0;
            } else {
                *q = q.clamp(1.01, 1000.0);
            }
        }
        if self.target_outcome >= self.decimal_quotes.len() {
            self.target_outcome = self.decimal_quotes.len() - 1;
        }
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&MarketDevigState {
            bankroll: self.bankroll,
            quote_format: self.quote_format,
            method: self.method,
            decimal_quotes: self.decimal_quotes.clone(),
            target_outcome: self.target_outcome,
            offered_decimal_odds: self.offered_decimal_odds,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: MarketDevigState =
            serde_json::from_str(state).map_err(|e| format!("Invalid devigger state: {e}"))?;
        if !s.bankroll.is_finite()
            || !(1.0..=1_000_000.0).contains(&s.bankroll)
            || !s.offered_decimal_odds.is_finite()
            || !(1.01..=1000.0).contains(&s.offered_decimal_odds)
            || !(MIN_OUTCOMES..=MAX_GUI_OUTCOMES).contains(&s.decimal_quotes.len())
            || s.target_outcome >= s.decimal_quotes.len()
            || !s
                .decimal_quotes
                .iter()
                .all(|q| q.is_finite() && (1.01..=1000.0).contains(q))
        {
            return Err("Market devigger parameters out of bounds".into());
        }
        self.bankroll = s.bankroll;
        self.quote_format = s.quote_format;
        self.method = s.method;
        self.decimal_quotes = s.decimal_quotes;
        self.target_outcome = s.target_outcome;
        self.offered_decimal_odds = s.offered_decimal_odds;
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        ui::render_devig_tool(self, ui);
    }
}

inventory::submit! {
    ToolMetadata {
        id: "MarketDeviggingTool",
        display_name: "Fair Odds & +EV Devigger",
        domain: "financial_math",
        tags: &["kelly", "devig", "shin", "odds", "ev"],
        build: || Box::new(MarketDeviggingTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for MarketDeviggingTool {
    fn theory_description(&self) -> String {
        format!(
            "Bookmaker vig removal over $n = {}$ outcomes recovering simplex probabilities $\\sum_{{i=1}}^n p_i = 1$ via Multiplicative ($p_i = \\pi_i / S$), Additive ($p_i = \\pi_i - (S - 1)/n$), Power ($\\sum \\pi_i^k = 1$), and Shin insider-trading equilibrium ($\\sum p_i(z) = 1$) calibrated with zero-allocation Brent root-finding, paired with Kelly sizing on ${:.0}$ bankroll.",
            self.decimal_quotes.len(),
            self.bankroll
        )
    }

    fn phonetic_description(&self) -> String {
        "Multi-outcome bookmaker vig removal, Shin insider-trading calibration, and expected value Kelly sizing".into()
    }

    fn theory_citation(&self) -> String {
        "Optimal Betting Odds Against Insider Traders (Shin, 1991); Measuring the Incidence of Insider Trading in a Market for State-Contingent Claims (Shin, 1993); Adjusting Bookmaker's Odds to Allow for Overround (Clarke, Kovalchik & Ingram, 2017)".into()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "Multiplicative".into(),
            "Scales raw implied probabilities proportionally by 1 / booksum".into(),
        );
        map.insert(
            "Additive".into(),
            "Subtracts equal overround per outcome with probability simplex boundary clamping"
                .into(),
        );
        map.insert(
            "Power".into(),
            "Raises implied probabilities to a constant exponent k > 1 accounting for favorite-longshot bias".into(),
        );
        map.insert(
            "Shin".into(),
            "Models bookmaker spread against an unknown fraction z of informed insider volume"
                .into(),
        );
        map
    }
}

// [cite:clinical_trials_statistics]
