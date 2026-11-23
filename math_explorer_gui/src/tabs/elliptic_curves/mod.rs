// @explorer_feature = "pure_math"
// [cite:modular_polynomials_review]
// [cite:hurwitz_kronecker]

use crate::tabs::ExplorerTab;
use eframe::egui;

mod chord_tangent_canvas;
mod chord_tangent_scene;
pub mod chord_tangent_tool;
mod chord_tangent_ui;
pub mod modular_invariants_tool;
mod modular_invariants_ui;

#[cfg(test)]
mod tests;

/// Auto-discovered Pure Math tab for short-Weierstrass elliptic curve geometry and modular invariants.
pub struct EllipticCurvesTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for EllipticCurvesTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("elliptic_curves"),
        }
    }
}

impl ExplorerTab for EllipticCurvesTab {
    fn name(&self) -> &'static str {
        "Elliptic Curves"
    }

    fn academic_domain(&self) -> &'static str {
        "Pure Math"
    }

    fn url_slug(&self) -> &'static str {
        "elliptic_curves"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "elliptic_curves");
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
