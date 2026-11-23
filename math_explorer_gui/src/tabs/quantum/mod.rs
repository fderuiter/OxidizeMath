#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod clebsch;
pub mod spin_viz;
pub mod volumetric_studio;
pub mod wave_sim;

crate::declare_simulation_tab!(
    pub struct QuantumTab,
    name: "Quantum Mechanics",
    domain: "Physics",
    slug: "quantum",
    order: 10,
);

// [cite:quantum_mechanics]
