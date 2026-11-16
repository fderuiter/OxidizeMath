// @explorer_feature = "physics"
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod clebsch;
pub mod spin_viz;
pub mod wave_sim;


pub struct QuantumTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for QuantumTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("quantum"),
        }
    }
}

impl ExplorerTab for QuantumTab {
    fn name(&self) -> &'static str {
        "Quantum Mechanics"
    }

    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.framework.show(ctx, "quantum");
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        self.framework.load_state(state);
    }
}

// [cite:quantum_mechanics]
