#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod curvature_heatmap;
pub mod custom_surface;
pub mod export_utils;
pub mod hyperbolic_geometry;
#[cfg(feature = "biology")]
pub mod surface_morphogenesis;
pub mod surface_viewer;
pub mod vietoris_rips;

crate::declare_simulation_tab!(
    pub struct GeometryTopologyTab,
    name: "Geometry & Topology",
    domain: "Pure Math",
    slug: "geometry_topology",
    order: 22,
);

// [cite:graph_parameters_rust]
