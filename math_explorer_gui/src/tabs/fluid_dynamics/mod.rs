// @explorer_feature = "physics"
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod lattice_boltzmann;
pub mod potential_flow;
pub mod turbulence;


pub struct FluidDynamicsTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for FluidDynamicsTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("fluid_dynamics"),
        }
    }
}

impl ExplorerTab for FluidDynamicsTab {
    fn name(&self) -> &'static str {
        "Fluid Dynamics"
    }

    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.framework.show(ctx, "fluid_dynamics");
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        self.framework.load_state(state);
    }
}

// [cite:graph_parameters_rust]
