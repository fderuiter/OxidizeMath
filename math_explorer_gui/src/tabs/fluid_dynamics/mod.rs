// @explorer_feature = "physics"
// @explorer_domain = "Physics"
// @explorer_slug = "fluid_dynamics"
// @explorer_order = 12

pub mod lattice_boltzmann;
pub mod lattice_boltzmann_sculpt;
pub mod lbm_fairing;
pub mod lbm_fairing_ui;
pub mod potential_flow;
mod potential_flow_brush;
pub mod turbulence;

crate::declare_simulation_tab!(
    pub struct FluidDynamicsTab,
    "Fluid Dynamics",
);

// [cite:graph_parameters_rust]
