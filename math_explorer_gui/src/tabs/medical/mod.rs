#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod beam_profiling;
pub mod dose;
pub mod imrt;
pub mod radar_gating;

crate::declare_simulation_tab!(
    pub struct MedicalTab,
    name: "Medical Physics",
    domain: "Physics",
    slug: "medical",
    order: 15,
);

// [cite:graph_parameters_rust]
