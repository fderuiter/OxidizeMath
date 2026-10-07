// @explorer_feature = "applied"
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod replicator;

pub struct GameTheoryTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for GameTheoryTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("game_theory"),
        }
    }
}

impl ExplorerTab for GameTheoryTab {
    fn name(&self) -> &'static str {
        "Game Theory"
    }

    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.framework.show(ctx, "game_theory");
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        self.framework.load_state(state);
    }
}

// [cite:graph_parameters_rust]
