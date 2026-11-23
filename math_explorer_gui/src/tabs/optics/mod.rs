// @explorer_feature = "physics"
// [cite:optics]

use eframe::egui;

use crate::tabs::ExplorerTab;

/// Far-field Fraunhofer single-slit, multi-slit grating, and circular Airy diffraction tool.
pub mod diffraction_tool;
mod diffraction_ui;
/// Shared optics colormaps, scene selectors, and federated starter gallery presets.
pub mod presets;
/// Interactive 2D optical bench, GRIN metamaterial ray tracer, and caustic envelope analyzer.
pub mod ray_bench_tool;
mod ray_bench_ui;

#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

/// Interactive `Optics & Caustics` tab backed by [`crate::framework::SimulationFramework`].
pub struct OpticsTab {
    framework: crate::framework::SimulationFramework,
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
        "Optics & Caustics"
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
