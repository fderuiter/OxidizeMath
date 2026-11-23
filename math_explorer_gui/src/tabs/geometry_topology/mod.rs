// @explorer_feature = "pure_math"
// @explorer_domain = "Pure Math"
// @explorer_slug = "geometry_topology"

pub mod curvature_heatmap;
pub mod custom_surface;
pub mod export_utils;
pub mod surface_viewer;
pub mod vietoris_rips;

crate::declare_simulation_tab!(
    pub struct GeometryTopologyTab,
    "Geometry & Topology",
);

// [cite:graph_parameters_rust]
