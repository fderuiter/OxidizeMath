#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::tabs::{ExplorerTab, TabDescriptor};
use eframe::egui;

pub mod capacity_fade;
pub mod lifetime_estimator;

pub struct BatteryDegradationTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for BatteryDegradationTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("battery_degradation"),
        }
    }
}

impl ExplorerTab for BatteryDegradationTab {
    fn name(&self) -> &'static str {
        "Battery Degradation"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "battery_degradation");
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
        id: "battery_degradation",
        order: 100,
        build: || Box::new(BatteryDegradationTab::default()),
    }
}

// [cite:algorithmic_information_rust]
