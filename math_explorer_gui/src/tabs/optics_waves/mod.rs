//! Interactive Physics tab for 2D FDTD wave propagation, Hamiltonian ray caustics, and optical sensing.
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};

pub mod overlays;
pub mod presets;
pub mod runner;
pub mod state;
pub mod tool;
pub mod ui;

#[cfg(test)]
pub use tool::OpticsWaveTool;

crate::declare_simulation_tab!(
    /// Static-inventory Physics tab for 2D FDTD wave propagation, ray caustics, and optical sensing.
    pub struct OpticsWaveTab,
    name: "Optics & Waves",
    domain: "Physics",
    slug: "optics_waves",
    order: 18,
);

impl FiniteStateGuard for OpticsWaveTab {
    fn is_finite_state(&self) -> bool {
        self.framework
            .active_tool
            .as_ref()
            .is_none_or(|t| t.has_finite_parameters())
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        None
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }

    fn auto_scale_step_size(&mut self) {
        if let Some(tool) = self.framework.active_tool.as_mut() {
            let _ = tool.clamp_parameters();
        }
    }

    fn clamp_parameters(&mut self) {
        if let Some(tool) = self.framework.active_tool.as_mut() {
            let _ = tool.clamp_parameters();
        }
    }

    fn reset_parameters(&mut self) {
        self.framework = crate::framework::SimulationFramework::new("optics_waves");
    }
}

#[cfg(test)]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

// [cite:optics]
