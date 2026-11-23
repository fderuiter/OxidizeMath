// @explorer_feature = "physics"
// Interactive 2D Helmholtz angular-spectrum wave diffraction and Gerchberg-Saxton
// phase-holography workbench under the Physics academic domain.

// [cite:graph_parameters_rust]

use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod presets;
pub mod workbench;

#[cfg(test)]
pub use workbench::{
    AperturePreset, FourierOpticsWorkbenchTool, OpticsDisplayMode, OpticsMode,
};

/// Explorer tab exposing Fourier diffraction, Talbot self-imaging, Zernike lens aberrations,
/// and Gerchberg-Saxton spatial light modulator phase-holography synthesis.
pub struct OpticsTab {
    /// Underlying tool-switching, preset, and figure/dataset export framework.
    pub framework: crate::framework::SimulationFramework,
}

impl Default for OpticsTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("optics"),
        }
    }
}

impl ExplorerTab for OpticsTab {
    fn name(&self) -> &'static str {
        "Optics"
    }

    fn academic_domain(&self) -> &'static str {
        "Physics"
    }

    fn url_slug(&self) -> &'static str {
        "optics"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "optics");
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

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.framework.focused_canvas_id()
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

#[cfg(test)]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;
