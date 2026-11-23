// @explorer_feature = "physics"
// @explorer_domain = "Physics"
// @explorer_slug = "medical"

pub mod beam_profiling;
pub mod dose;
pub mod imrt;
pub mod radar_gating;

crate::declare_simulation_tab!(
    pub struct MedicalTab,
    "Medical Physics",
);

// [cite:graph_parameters_rust]
