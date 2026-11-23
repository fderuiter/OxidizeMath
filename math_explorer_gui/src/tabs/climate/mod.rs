// @explorer_feature = "climate"
// @explorer_domain = "Climate"
// @explorer_slug = "climate"

pub mod cera;
pub mod co2_projections;
pub mod temperature;

crate::declare_simulation_tab!(
    pub struct ClimateTab,
    "Climate",
);

// [cite:graph_parameters_rust]
