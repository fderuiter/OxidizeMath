// @explorer_feature = "physics"
// @explorer_order = 17
// Interactive General Relativity tab exposing 3D geodesic integration, Flamm/wormhole
// embeddings, radial effective potentials, twin clocks, and Christoffel tensor inspection.

use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod inspector;
pub mod playground;
pub mod presets;

#[cfg(test)]
pub use inspector::ChristoffelInspectorTool;
#[cfg(test)]
pub use playground::{GeodesicPlaygroundTool, SpacetimePreset};

/// Explorer tab grouping curved-spacetime geodesic and tensor inspection tools.
pub struct RelativityTab {
    /// Underlying tool-switching and export simulation framework.
    pub framework: crate::framework::SimulationFramework,
}

impl Default for RelativityTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("relativity"),
        }
    }
}

impl ExplorerTab for RelativityTab {
    fn name(&self) -> &'static str {
        "General Relativity"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "relativity");
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

// [cite:general_relativity_geodesics]

#[cfg(test)]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;
