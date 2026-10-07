// @explorer_feature = "applied"
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod randomization;
pub mod sample_size;
pub mod survival;


pub struct ClinicalTrialsTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for ClinicalTrialsTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("clinical_trials"),
        }
    }
}

impl ExplorerTab for ClinicalTrialsTab {
    fn name(&self) -> &'static str {
        "Clinical Trials"
    }

    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.framework.show(ctx, "clinical_trials");
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        self.framework.load_state(state);
    }
}

// [cite:clinical_trials_statistics]
