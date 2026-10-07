// @explorer_feature = "physics"
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod beam_profiling;
pub mod dose;


pub struct MedicalTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for MedicalTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("medical"),
        }
    }
}

impl ExplorerTab for MedicalTab {
    fn name(&self) -> &'static str {
        "Medical Physics"
    }

    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.framework.show(ctx, "medical");
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        self.framework.load_state(state);
    }
}

// [cite:graph_parameters_rust]
