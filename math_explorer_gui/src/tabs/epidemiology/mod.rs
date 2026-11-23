// @explorer_feature = "epidemiology"
// @explorer_domain = "Biology"
// @explorer_slug = "epidemiology"

pub mod network_propagation;
pub mod sir;

crate::declare_simulation_tab!(
    pub struct EpidemiologyTab,
    "Epidemiology",
);

// [cite:graph_parameters_rust]
