#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod network_propagation;
pub mod sir;

crate::declare_simulation_tab!(
    pub struct EpidemiologyTab,
    name: "Epidemiology",
    domain: "Biology",
    slug: "epidemiology",
    order: 42,
);

// [cite:graph_parameters_rust]
