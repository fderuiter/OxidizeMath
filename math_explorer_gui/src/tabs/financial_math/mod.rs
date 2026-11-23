#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod bankroll_ensemble;
pub mod bankroll_growth;
pub mod bet_size;
pub mod correlated_risk;
pub mod market_devig;
pub mod market_execution;
pub mod parrondo;
#[cfg(test)]
mod tests;

crate::declare_simulation_tab!(
    pub struct FinancialMathTab,
    name: "Financial Math",
    domain: "Pure Math",
    slug: "financial_math",
    order: 24,
);

// [cite:clinical_trials_statistics]
