// @explorer_feature = "pure_math"
// @explorer_domain = "Pure Math"
// @explorer_slug = "algorithmic_information"

pub mod controls;
pub mod plots;
pub mod presets;
pub mod state;

#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

crate::declare_simulation_tab!(
    /// Top-level `Pure Math` tab auto-discovered by `math_explorer_gui/build.rs`.
    pub struct AlgorithmicInfoTab,
    "Algorithmic Information",
);

// [cite:algorithmic_information_rust]
