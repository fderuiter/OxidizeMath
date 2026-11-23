#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod lattice_boltzmann;
pub mod lattice_boltzmann_sculpt;
pub mod lbm_fairing;
pub mod lbm_fairing_ui;
pub mod potential_flow;
mod potential_flow_brush;
pub mod turbulence;

crate::declare_simulation_tab!(
    pub struct FluidDynamicsTab,
    name: "Fluid Dynamics",
    domain: "Physics",
    slug: "fluid_dynamics",
    order: 12,
);

// [cite:graph_parameters_rust]
