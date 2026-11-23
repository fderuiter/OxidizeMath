// @explorer_feature = "physics"
// @explorer_domain = "Physics"
// @explorer_slug = "solid_state"

pub mod band_structure;
pub mod crystal_viewer;
pub mod ising;

crate::declare_simulation_tab!(
    pub struct SolidStateTab,
    "Solid State Physics",
);

// [cite:graph_parameters_rust]
