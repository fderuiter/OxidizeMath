// @explorer_feature = "physics"
// @explorer_domain = "Physics"
// @explorer_slug = "relativity"
// Interactive General Relativity tab exposing 3D geodesic integration, Flamm/wormhole
// embeddings, radial effective potentials, twin clocks, and Christoffel tensor inspection.

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
    "General Relativity",
);

// [cite:general_relativity_geodesics]

#[cfg(test)]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;
