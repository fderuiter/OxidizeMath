// @explorer_feature = "ai"
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod activation_functions;
pub mod attention_maps;
pub mod grid_world;
pub mod loss_landscape;
pub mod q_table_inspector;
pub mod reward_plots;
pub mod tokenization;
pub mod training_monitor;
pub mod gaussian_splatting;

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

    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.framework.show(ctx, "ai");
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        self.framework.load_state(state);
    }
}

// [cite:graph_parameters_rust]
