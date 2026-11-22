// @explorer_feature = "physics"
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::tabs::ExplorerTab;
use eframe::egui;
use std::sync::Arc;

pub mod overlays;
pub mod presets;
pub mod runner;
pub mod state;
pub mod tool;
pub mod ui;

pub use runner::OpticsSharedParams;
pub use tool::OpticsWaveTool;

/// Auto-discovered Physics tab for 2D FDTD wave propagation, ray caustics, and optical sensing.
pub struct OpticsWaveTab {
    /// Underlying simulation framework managing presets, exports, URL sharing, and recovery.
    pub framework: crate::framework::SimulationFramework,
    pub(crate) params: Arc<std::sync::RwLock<OpticsSharedParams>>,
}

impl Default for OpticsWaveTab {
    fn default() -> Self {
        let mut framework = crate::framework::SimulationFramework::new("optics_waves");
        let tool = OpticsWaveTool::default();
        let params = Arc::clone(&tool.params);
        let name = crate::framework::InteractiveTool::name(&tool).to_string();
        let baseline = crate::framework::InteractiveTool::save_state(&tool);
        framework.active_tool = Some(Box::new(tool));
        if let Some(b) = baseline {
            framework.saved_tool_states.insert(name, b.clone());
            framework.baseline_tool_state = Some(b);
        }
        framework.cache_valid_snapshot();
        Self { framework, params }
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
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.cfl_scale = (p.cfl_scale * 0.5).clamp(0.10, 1.00);
        p.force_non_finite = false;
    }

    fn clamp_parameters(&mut self) {
        if let Some(tool) = self.framework.active_tool.as_mut() {
            let _ = tool.clamp_parameters();
        }
    }

    fn reset_parameters(&mut self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        *p = OpticsSharedParams::default();
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

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.framework
            .active_tool
            .as_ref()
            .and_then(|t| t.focused_canvas_id())
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

#[cfg(test)]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

// [cite:optics]
