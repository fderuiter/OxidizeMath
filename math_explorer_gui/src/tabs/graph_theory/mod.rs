// @explorer_feature = "pure_math"
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod algorithm_visualizer;
pub mod graph_editor;
pub mod network_metrics;

pub struct GraphTheoryTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for GraphTheoryTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("graph_theory"),
        }
    }
}

impl ExplorerTab for GraphTheoryTab {
    fn name(&self) -> &'static str {
        "Graph Theory"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "graph_theory");
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

    fn can_undo(&self) -> bool {
        self.framework.can_undo()
    }

    fn can_redo(&self) -> bool {
        self.framework.can_redo()
    }

    fn try_undo(&mut self) -> bool {
        self.framework.try_undo()
    }

    fn try_redo(&mut self) -> bool {
        self.framework.try_redo()
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.framework.focused_canvas_id()
    }

    #[cfg(test)]
    fn framework(&self) -> Option<&crate::framework::SimulationFramework> {
        Some(&self.framework)
    }

    #[cfg(test)]
    fn framework_mut(&mut self) -> Option<&mut crate::framework::SimulationFramework> {
        Some(&mut self.framework)
    }
}

// [cite:graph_parameters_rust]
