#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod attractors;
pub mod bifurcation;
pub mod fractals;
pub mod hamiltonian;

crate::declare_simulation_tab!(
    pub struct ChaosTab,
    name: "Chaos Theory",
    domain: "Physics",
    slug: "chaos",
    order: 13,
);

// [cite:graph_parameters_rust]
