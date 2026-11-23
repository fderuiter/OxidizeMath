// @explorer_feature = "physics"
// @explorer_domain = "Physics"
// @explorer_slug = "chaos"
// @explorer_order = 13

pub mod attractors;
pub mod bifurcation;
pub mod fractals;
pub mod hamiltonian;

crate::declare_simulation_tab!(
    pub struct ChaosTab,
    "Chaos Theory",
);

// [cite:graph_parameters_rust]
