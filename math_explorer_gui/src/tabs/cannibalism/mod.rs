#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

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
    name: "Population Structure",
    domain: "Applied Math",
    slug: "cannibalism",
    order: 36,
);

#[cfg(test)]
mod tests;

// [cite:cannibalism]
