#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:biology]

use crate::tabs::{ExplorerTab, TabDescriptor};
use eframe::egui;

pub mod dynamics;
pub mod evolution;

/// Static-inventory Biology tab hosting [`dynamics::BiochemicalDynamicsTool`] and
/// [`evolution::EvolutionaryBiologyTool`].
pub struct BiochemicalTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for BiochemicalTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("biochemical"),
        }
    }
}

impl ExplorerTab for BiochemicalTab {
    fn name(&self) -> &'static str {
        "Biochemical Dynamics"
    }

    fn academic_domain(&self) -> &'static str {
        "Biology"
    }

    fn url_slug(&self) -> &'static str {
        "biochemical"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "biochemical");
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.framework
            .active_tool
            .as_ref()
            .and_then(|tool| tool.focused_canvas_id())
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
        id: "biochemical",
        order: 43,
        build: || Box::new(BiochemicalTab::default()),
    }
}
