#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod chladni_wave_tank;
pub mod complex_mapping;
pub mod ode;
pub mod riemann;

crate::declare_simulation_tab!(
    pub struct AnalysisTab,
    name: "Analysis & Calculus",
    domain: "Pure Math",
    slug: "analysis",
    order: 20,
);

// [cite:graph_parameters_rust]
