//! Physics tab for Stochastic Partial Differential Equations (SPDEs):
//! 1D & 2D Edwards-Wilkinson and Kardar-Parisi-Zhang interface growth (`KpzInterfaceTool`)
//! and 2D Allen-Cahn & Cahn-Hilliard spinodal decomposition (`SpinodalDecompositionTool`).
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:stochastic_pde]
// [cite:interface_growth]
// [cite:phase_field]

use crate::tabs::{ExplorerTab, TabDescriptor};
use eframe::egui;

mod kpz_canvas;
/// Interactive 1D/2D Edwards-Wilkinson and KPZ surface growth playground.
pub mod kpz_tool;
mod kpz_ui;
/// Starter presets, state schemas, and [`scientific_metadata::theory::TheoryDescribable`] descriptors.
pub mod presets;
mod spinodal_canvas;
/// Interactive 2D Model A (Allen-Cahn) and Model B (Cahn-Hilliard) spinodal decomposition studio.
pub mod spinodal_tool;
mod spinodal_ui;

#[cfg(test)]
pub use kpz_tool::{KpzDimensionMode, KpzInterfaceTool};
#[cfg(test)]
pub use spinodal_tool::SpinodalDecompositionTool;

#[cfg(test)]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

/// Explorer tab grouping stochastic interface roughening and spinodal phase-separation studios.
pub struct StochasticPdeTab {
    /// Underlying tool-switching, preset, recovery, and export framework.
    pub framework: crate::framework::SimulationFramework,
}

impl Default for StochasticPdeTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("stochastic_pde"),
        }
    }
}

impl ExplorerTab for StochasticPdeTab {
    fn name(&self) -> &'static str {
        "Stochastic PDEs"
    }

    fn academic_domain(&self) -> &'static str {
        "Physics"
    }

    fn url_slug(&self) -> &'static str {
        "stochastic_pde"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "stochastic_pde");
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.framework
            .active_tool
            .as_ref()
            .and_then(|tool| tool.focused_canvas_id())
    }

    fn state_revision(&self) -> u64 {
        self.framework.state_revision()
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        self.framework.load_state(state);
    }

    fn select_tool_by_name(&mut self, name: &str) -> bool {
        self.framework.select_tool_by_name(name)
    }

    fn framework(&self) -> Option<&crate::framework::SimulationFramework> {
        Some(&self.framework)
    }

    fn framework_mut(&mut self) -> Option<&mut crate::framework::SimulationFramework> {
        Some(&mut self.framework)
    }
}

inventory::submit! {
    TabDescriptor {
        id: "stochastic_pde",
        order: 18,
        build: || Box::new(StochasticPdeTab::default()),
    }
}
