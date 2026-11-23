#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// Interactive Subatomic Physics tab exposing nuclear stability & Bateman decay chains,
// Standard Model flavor/electroweak/QCD/MSW neutrino physics, and relativistic synchrotron jets.

use crate::tabs::{ExplorerTab, TabDescriptor};
use eframe::egui;

pub mod nuclear_chart;
pub mod nuclear_tool;
pub mod nuclear_ui;
pub mod presets;
pub mod relativistic_jet_tool;
pub mod relativistic_jet_ui;
pub mod standard_model_tool;
pub mod standard_model_ui;

#[cfg(test)]
pub use nuclear_tool::NuclearStabilityTool;
#[cfg(test)]
pub use relativistic_jet_tool::RelativisticJetTool;
#[cfg(test)]
pub use standard_model_tool::StandardModelTool;

/// Explorer tab grouping Nuclear, Standard Model, and Relativistic Jet laboratories.
pub struct SubatomicPhysicsTab {
    /// Underlying tool-switching, recovery, and export simulation framework.
    pub framework: crate::framework::SimulationFramework,
}

impl Default for SubatomicPhysicsTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("subatomic"),
        }
    }
}

impl ExplorerTab for SubatomicPhysicsTab {
    fn name(&self) -> &'static str {
        "Subatomic Physics"
    }

    fn academic_domain(&self) -> &'static str {
        "Physics"
    }

    fn url_slug(&self) -> &'static str {
        "subatomic"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "subatomic");
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
        id: "subatomic",
        order: 18,
        build: || Box::new(SubatomicPhysicsTab::default()),
    }
}

// [cite:nuclear]
// [cite:standard_model]
// [cite:high_energy]

#[cfg(test)]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;
