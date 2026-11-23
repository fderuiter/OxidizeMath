// @explorer_feature = "physics"
// @explorer_domain = "Physics"
// @explorer_slug = "quantum"

pub mod clebsch;
pub mod spin_viz;
pub mod volumetric_studio;
pub mod wave_sim;
pub mod wave_sim_sculpt;

crate::declare_simulation_tab!(
    pub struct QuantumTab,
    "Quantum Mechanics",
);

// [cite:quantum_mechanics]
