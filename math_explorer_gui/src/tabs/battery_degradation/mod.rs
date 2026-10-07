// @explorer_feature = "applied"
use crate::tabs::ExplorerTab;
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

    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.framework.show(ctx, "battery_degradation");
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        self.framework.load_state(state);
    }
}

// [cite:algorithmic_information_rust]
