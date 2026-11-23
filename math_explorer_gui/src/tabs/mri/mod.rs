// @explorer_feature = "physics"
use crate::async_sim::SimWorkerState;
use crate::tabs::ExplorerTab;
use eframe::egui;
use serde::{Deserialize, Serialize};

pub mod kspace_bloch_view;
pub mod kspace_phantom;
pub mod kspace_runner;
pub mod kspace_tool;
pub mod kspace_types;
pub mod kspace_ui;
pub mod kspace_worker;
pub mod pulse;
pub mod runner;
mod state;
pub mod theory;
pub mod tool;
pub mod ui;

#[cfg(test)]
pub use kspace_tool::MriKSpaceTool;
pub use runner::sanitize_mri_time_scale;
#[cfg(test)]
pub use runner::{MriRunner, MriSharedParams, MriSnapshot, MAX_MRI_TIME_SCALE, MIN_MRI_TIME_SCALE};
#[cfg(test)]
pub use tool::MriTool;
pub use tool::MriToolState;

/// Legacy flat `MriState` schema saved by pre-framework `.mws` workspaces.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyMriState {
    /// Optional worker execution state.
    #[serde(default)]
    pub state: Option<SimWorkerState>,
    /// Legacy boolean pause flag (`paused: false` => `Running`, `paused: true` => `Ready`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused: Option<bool>,
    /// Previously persisted simulation speed multiplier.
    pub time_scale: f64,
}

/// Physics tab hosting [`tool::MriTool`] inside [`crate::framework::SimulationFramework`].
pub struct MriTab {
    /// Underlying simulation framework managing exports, presets, theory portal, and state guard.
    pub framework: crate::framework::SimulationFramework,
}

impl Default for MriTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("mri"),
        }
    }
}

impl MriTab {
    fn try_load_legacy_state(&mut self, state: &str) -> bool {
        let Ok(legacy) = serde_json::from_str::<LegacyMriState>(state) else {
            return false;
        };
        let _ = self.framework.select_tool_by_name("MriTool");
        let Some(tool) = self.framework.active_tool.as_mut() else {
            return true;
        };
        let Some(current_json) = tool.save_state() else {
            return true;
        };
        let Ok(mut current_state) = serde_json::from_str::<MriToolState>(&current_json) else {
            return true;
        };
        current_state.time_scale = sanitize_mri_time_scale(legacy.time_scale);
        current_state.state = Some(legacy.state.unwrap_or_else(|| {
            if legacy.paused.unwrap_or(false) {
                SimWorkerState::Ready
            } else {
                SimWorkerState::Running
            }
        }));
        current_state.paused = None;
        if let Ok(updated_json) = serde_json::to_string(&current_state) {
            if tool.try_load_state(&updated_json).is_ok() {
                if let Some(saved) = tool.save_state() {
                    self.framework
                        .saved_tool_states
                        .insert(tool.name().to_string(), saved);
                }
                self.framework.cache_valid_snapshot();
            }
        }
        true
    }
}

impl ExplorerTab for MriTab {
    fn name(&self) -> &'static str {
        "MRI Physics"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "mri");
    }

    fn state_revision(&self) -> u64 {
        self.framework.state_revision()
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        if !self.try_load_legacy_state(state) {
            self.framework.load_state(state);
        }
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

impl tool::MriTool {
    pub(super) fn build_diagnostic_report(
        &self,
    ) -> Option<crate::framework::recovery::DiagnosticReport> {
        use crate::framework::InteractiveTool as _;
        if let Some(report) = self.worker.diagnostic_report() {
            return Some(report.clone());
        }
        if !self.has_finite_parameters() {
            let t = if self.current_time.is_finite() {
                self.current_time
            } else {
                0.0
            };
            let var = if !self.simulator.t1.is_finite() {
                "t1"
            } else if !self.simulator.t2.is_finite() {
                "t2"
            } else if !self.simulator.b_field.x.is_finite()
                || !self.simulator.b_field.y.is_finite()
                || !self.simulator.b_field.z.is_finite()
            {
                "b_field"
            } else if !self.time_scale.is_finite() {
                "time_scale"
            } else if self.pulse_config.validate().is_err() {
                "pulse_config"
            } else {
                "magnetization"
            };
            return Some(crate::framework::recovery::DiagnosticReport::new(
                t,
                self.history.len(),
                var,
                "Bloch magnetization vector or MRI parameter became non-finite.",
            ));
        }
        None
    }

    pub(super) fn clamp_mri_parameters(&mut self) {
        self.simulator.t1 = if self.simulator.t1.is_finite() {
            self.simulator.t1.clamp(0.1, 5.0)
        } else {
            tool::DEFAULT_MRI_T1
        };
        self.simulator.t2 = if self.simulator.t2.is_finite() {
            self.simulator.t2.clamp(0.01, 2.0)
        } else {
            tool::DEFAULT_MRI_T2
        };
        let clamp_b = |v: f64, fallback: f64| {
            if v.is_finite() {
                v.clamp(-50.0, 50.0)
            } else {
                fallback
            }
        };
        self.simulator.b_field = nalgebra::Vector3::new(
            clamp_b(self.simulator.b_field.x, 0.0),
            clamp_b(self.simulator.b_field.y, 0.0),
            clamp_b(self.simulator.b_field.z, tool::DEFAULT_MRI_BZ),
        );
        self.time_scale = sanitize_mri_time_scale(self.time_scale);
        if self.pulse_config.validate().is_err() {
            self.pulse_config = pulse::RfPulseConfig::default();
        }
        self.reset();
    }
}

#[cfg(test)]
mod kspace_tests;
#[cfg(test)]
mod tests;

// [cite:graph_parameters_rust]
