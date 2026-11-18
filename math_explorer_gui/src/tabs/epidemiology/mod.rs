// @explorer_feature = "epidemiology"
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod network_propagation;
pub mod sir;


pub struct EpidemiologyTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for EpidemiologyTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("epidemiology"),
        }
    }
}

impl ExplorerTab for EpidemiologyTab {
    fn name(&self) -> &'static str {
        "Epidemiology"
    }

    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.framework.show(ctx, "epidemiology");
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
