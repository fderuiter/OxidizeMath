// @explorer_feature = "applied"
// @explorer_domain = "Applied Math"
// @explorer_slug = "cannibalism"
// @explorer_order = 36

/// Validated state schema, bounds, and federated presets for age-structured cannibalism.
pub mod state;
/// Theoretical equations and citations for McKendrick-von Foerster and van den Bosch models.
pub mod theory;
/// In-thread simulation runner, Lexis heatmap builder, and [`FiniteStateGuard`](crate::framework::FiniteStateGuard).
pub mod tool;
/// Side-panel controls and scientific plots (Lexis heatmap, age pyramid, Hopf cycles, 2D ODE).
pub mod views;

pub use tool::PopulationStructureTool;

crate::declare_simulation_tab!(
    /// Applied mathematics tab hosting [`PopulationStructureTool`] inside [`crate::framework::SimulationFramework`].
    pub struct PopulationStructureTab,
    "Population Structure",
);

#[cfg(test)]
mod tests;

// [cite:cannibalism]
