// @explorer_feature = "applied"
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod pharmacokinetics;
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

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "clinical_trials");
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

// [cite:clinical_trials_statistics]
