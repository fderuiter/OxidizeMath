// @explorer_feature = "physics"
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod band_structure;
pub mod crystal_viewer;
pub mod ising;

pub struct SolidStateTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for SolidStateTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("solid_state"),
        }
    }
}

impl ExplorerTab for SolidStateTab {
    fn name(&self) -> &'static str {
        "Solid State Physics"
    }

    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.framework.show(ctx, "solid_state");
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        self.framework.load_state(state);
    }
}

// [cite:graph_parameters_rust]
