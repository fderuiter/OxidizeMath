// @explorer_feature = "physics"
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod attractors;
pub mod bifurcation;
pub mod fractals;

pub struct ChaosTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for ChaosTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("chaos"),
        }
    }
}

impl ExplorerTab for ChaosTab {
    fn name(&self) -> &'static str {
        "Chaos Theory"
    }

    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.framework.show(ctx, "chaos");
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        self.framework.load_state(state);
    }
}

// [cite:graph_parameters_rust]
