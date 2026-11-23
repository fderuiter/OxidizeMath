#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod band_structure;
pub mod crystal_viewer;
pub mod ising;

crate::declare_simulation_tab!(
    pub struct SolidStateTab,
    name: "Solid State Physics",
    domain: "Physics",
    slug: "solid_state",
    order: 14,
);

// [cite:graph_parameters_rust]
