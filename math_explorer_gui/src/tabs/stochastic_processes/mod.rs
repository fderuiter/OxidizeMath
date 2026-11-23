// @explorer_feature = "pure_math"
// Dedicated `Stochastic Processes` tab (`Pure Math`) hosting three interactive
// `SimulationFramework` workbenches for DTMCs & transition tensors, CTMCs & M/M/c queues,
// and HMMs & Ornstein-Uhlenbeck calibration.

use crate::tabs::ExplorerTab;
use eframe::egui;

/// Continuous-time Markov chains, Poisson thinning, and $M/M/c$ birth-death queueing workbench.
pub mod ctmc_queue_workbench;
/// Discrete-time Markov chains and non-stationary `TransitionTensor` workbench.
pub mod dtmc_workbench;
/// Log-space Hidden Markov Model (Viterbi + Baum-Welch EM) and exact Ornstein-Uhlenbeck workbench.
pub mod hmm_ou_workbench;

#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

/// Top-level `Pure Math` tab auto-discovered by `math_explorer_gui/build.rs`.
pub struct StochasticProcessesTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for StochasticProcessesTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("stochastic_processes"),
        }
    }
}

impl ExplorerTab for StochasticProcessesTab {
    fn name(&self) -> &'static str {
        "Stochastic Processes"
    }

    fn academic_domain(&self) -> &'static str {
        "Pure Math"
    }

    fn url_slug(&self) -> &'static str {
        "stochastic_processes"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "stochastic_processes");
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

// [cite:clinical_trials_statistics]
