#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod cera;
pub mod co2_projections;
pub mod temperature;

crate::declare_simulation_tab!(
    pub struct ClimateTab,
    name: "Climate",
    domain: "Climate",
    slug: "climate",
    order: 50,
);

// [cite:graph_parameters_rust]
