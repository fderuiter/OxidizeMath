// @explorer_feature = "pure_math"
// @explorer_domain = "Pure Math"
// @explorer_slug = "analysis"

pub mod chladni_wave_tank;
pub mod complex_mapping;
pub mod ode;
pub mod riemann;

crate::declare_simulation_tab!(
    pub struct AnalysisTab,
    "Analysis & Calculus",
);

// [cite:graph_parameters_rust]
