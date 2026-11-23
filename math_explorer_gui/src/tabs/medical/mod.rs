// @explorer_feature = "physics"
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod beam_profiling;
pub mod dose;
pub mod imrt;
pub mod radar_gating;

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

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "medical");
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

// [cite:graph_parameters_rust]
