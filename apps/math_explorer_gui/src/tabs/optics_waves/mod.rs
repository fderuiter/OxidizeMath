//! Interactive Physics tab for 2D FDTD wave propagation, Hamiltonian ray caustics, and optical sensing.
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::tabs::{ExplorerTab, TabDescriptor};
use eframe::egui;

mod export;
pub mod overlays;
pub mod presets;
pub mod runner;
pub mod state;
pub mod tool;
pub mod ui;

#[cfg(test)]
pub use tool::OpticsWaveTool;

/// Static-inventory Physics tab for 2D FDTD wave propagation, ray caustics, and optical sensing.
pub struct OpticsWaveTab {
    /// Underlying simulation framework managing presets, exports, URL sharing, and recovery.
    pub framework: crate::framework::SimulationFramework,
}

impl Default for OpticsWaveTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("optics_waves"),
        }
    }
}

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

impl ExplorerTab for OpticsWaveTab {
    fn name(&self) -> &'static str {
        "Optics & Waves"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "optics_waves");
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
        id: "optics_waves",
        order: 18,
        build: || Box::new(OpticsWaveTab::default()),
    }
}

#[cfg(test)]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

// [cite:optics]
