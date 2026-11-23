#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::tabs::{ExplorerTab, TabDescriptor};
use eframe::egui;

pub mod activation_functions;
pub mod attention_maps;
pub mod gaussian_splatting;
pub mod grid_world;
pub mod loss_landscape;
pub mod q_table_inspector;
pub mod reward_plots;
pub mod tokenization;
pub mod training_monitor;

pub struct AiTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for AiTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("ai"),
        }
    }
}

impl ExplorerTab for AiTab {
    fn name(&self) -> &'static str {
        "Artificial Intelligence"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "ai");
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
        id: "ai",
        order: 100,
        build: || Box::new(AiTab::default()),
    }
}

// [cite:graph_parameters_rust]
