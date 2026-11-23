#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod controls;
pub mod plots;
pub mod presets;
pub mod state;

#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

crate::declare_simulation_tab!(
    /// Top-level `Pure Math` tab registered via [`crate::tabs::TabDescriptor`].
    pub struct AlgorithmicInfoTab,
    name: "Algorithmic Information",
    domain: "Pure Math",
    slug: "algorithmic_information",
    order: 25,
);

// [cite:algorithmic_information_rust]
