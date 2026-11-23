//! Interactive General Relativity tab exposing 3D geodesic integration, Flamm/wormhole
//! embeddings, radial effective potentials, twin clocks, and Christoffel tensor inspection.
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod inspector;
pub mod playground;
pub mod presets;

#[cfg(test)]
pub use inspector::ChristoffelInspectorTool;
#[cfg(test)]
pub use playground::{GeodesicPlaygroundTool, SpacetimePreset};

crate::declare_simulation_tab!(
    /// Explorer tab grouping curved-spacetime geodesic and tensor inspection tools.
    pub struct RelativityTab,
    name: "General Relativity",
    domain: "Physics",
    slug: "relativity",
    order: 17,
);

// [cite:general_relativity_geodesics]

#[cfg(test)]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;
