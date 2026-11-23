// @explorer_feature = "pure_math"
// @explorer_domain = "Pure Math"
// @explorer_slug = "financial_math"

pub mod bankroll_ensemble;
pub mod bankroll_growth;
pub mod bet_size;
pub mod parrondo;

crate::declare_simulation_tab!(
    pub struct FinancialMathTab,
    "Financial Math",
);

// [cite:clinical_trials_statistics]
