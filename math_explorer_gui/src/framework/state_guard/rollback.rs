#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::recovery::*;
use crate::framework::SimulationFramework;

impl SimulationFramework {
    fn try_restore_snapshot_or_clamped(&mut self, state_str: &str) -> Option<(&'static str, bool)> {
        let base_ref = self.baseline_tool_state.as_deref();
        let tool = self.active_tool.as_mut()?;
        if tool.try_load_recovery_state(state_str).is_ok() && tool.has_finite_parameters() {
            return Some((
                "Restored exact pre-divergence parameter configuration and plot data",
                false,
            ));
        }
        let constraints = tool.theory().theory_parameters();
        let clamped = crate::framework::clamp::clamp_state_json(state_str, base_ref, &constraints)?;
        if tool.try_load_recovery_state(&clamped).is_ok() && tool.has_finite_parameters() {
            return Some((
                "Snapshot validation failed; restored clamped parameter configuration",
                true,
            ));
        }
        None
    }

    fn restore_verified_recovery_state(
        &mut self,
        snapshot_state: Option<&str>,
    ) -> (&'static str, bool) {
        if self.active_tool.is_none() {
            return (
                "Restored exact pre-divergence parameter configuration and plot data",
                false,
            );
        }
        if let Some(state_str) = snapshot_state {
            if let Some(outcome) = self.try_restore_snapshot_or_clamped(state_str) {
                return outcome;
            }
        } else if self
            .active_tool
            .as_ref()
            .is_some_and(|t| t.has_finite_parameters())
        {
            return (
                "Restored exact pre-divergence parameter configuration and plot data",
                false,
            );
        }

        let base_clone = self.baseline_tool_state.clone();
        if let (Some(tool), Some(ref base)) = (&mut self.active_tool, &base_clone) {
            if tool.try_load_recovery_state(base).is_ok() && tool.has_finite_parameters() {
                return (
                    "Snapshot validation failed; restored baseline parameter configuration",
                    true,
                );
            }
        }
        if let Some(tool) = &mut self.active_tool {
            if tool.clamp_parameters() && tool.has_finite_parameters() {
                return (
                    "Snapshot validation failed; clamped active tool parameters",
                    true,
                );
            }
            if let Some(&meta) = self
                .selected_tool_index
                .and_then(|idx| self.available_tools.get(idx))
            {
                let fresh = (meta.build)();
                let factory = fresh.save_state();
                self.factory_tool_state = factory.clone();
                self.baseline_tool_state = self.preset_tool_state.clone().or(factory);
                *tool = fresh;
            }
        }
        (
            "Snapshot validation failed; rebuilt default baseline tool",
            true,
        )
    }

    /// Executes the Step-Back action: restores pre-divergence parameters without clobbering pinned baselines.
    pub fn rollback_step_back(&mut self) -> bool {
        let Some(snapshot) = self.last_valid_snapshot.clone() else {
            return false;
        };
        let (details, used_fallback) =
            self.restore_verified_recovery_state(snapshot.tool_state.as_deref());
        self.bump_plot_sync_generation();
        self.bump_framework_revision();
        let plots =
            if !used_fallback && inspect_datasets_for_divergence(&snapshot.plot_datasets).is_ok() {
                snapshot.plot_datasets
            } else {
                Vec::new()
            };
        self.restore_snapshot_plots(plots);
        if used_fallback {
            self.refresh_saved_and_snapshot_after_reset();
        }

        emit_recovery_telemetry("SimulationFramework", "Step Back", details);
        true
    }

    /// Executes the Auto-Scale Step Size action: halves `dt` on the recovery snapshot or via domain hook while preserving pinned baselines.
    pub fn rollback_auto_scale_step_size(&mut self) -> bool {
        let Some(snapshot) = self.last_valid_snapshot.clone() else {
            if let Some(tool) = &mut self.active_tool {
                if tool.auto_scale_step_size() && tool.has_finite_parameters() {
                    self.bump_framework_revision();
                    let valid_plots = self.current_valid_plot_datasets.clone();
                    self.restore_snapshot_plots(valid_plots);
                    self.refresh_saved_and_snapshot_after_reset();
                    emit_recovery_telemetry(
                        "SimulationFramework",
                        "Auto-scale Step Size (dt)",
                        "Halved step size via domain hook",
                    );
                    return true;
                }
            }
            return false;
        };
        let scaled = snapshot
            .tool_state
            .as_deref()
            .and_then(auto_scale_state_json);
        let mut applied_scaled = false;
        if let (Some(tool), Some(ref halved_state)) = (&mut self.active_tool, &scaled) {
            if tool.try_load_recovery_state(halved_state).is_ok() && tool.has_finite_parameters() {
                applied_scaled = true;
            }
        }
        if !applied_scaled {
            if let Some(tool) = &mut self.active_tool {
                if let Some(ref state_str) = snapshot.tool_state {
                    tool.load_recovery_state(state_str);
                }
                if tool.auto_scale_step_size() && tool.has_finite_parameters() {
                    applied_scaled = true;
                }
            }
        }
        let mut used_fallback = false;
        if !applied_scaled {
            let (_, fb) = self.restore_verified_recovery_state(snapshot.tool_state.as_deref());
            used_fallback = fb;
        }
        self.bump_plot_sync_generation();
        self.bump_framework_revision();
        let plots =
            if !used_fallback && inspect_datasets_for_divergence(&snapshot.plot_datasets).is_ok() {
                snapshot.plot_datasets
            } else {
                Vec::new()
            };
        self.restore_snapshot_plots(plots);
        if used_fallback {
            self.refresh_saved_and_snapshot_after_reset();
        }

        emit_recovery_telemetry(
            "SimulationFramework",
            "Auto-scale Step Size (dt)",
            "Halved step size and restarted simulation from snapshot",
        );
        true
    }

    pub(super) fn restore_snapshot_plots(&mut self, plot_datasets: Vec<(String, Vec<[f64; 2]>)>) {
        egui_plot::clear_last_plot_datasets();
        egui_plot::append_last_plot_datasets(&plot_datasets);
        self.sync_state.datasets_owner_tool = self
            .active_tool
            .as_ref()
            .filter(|_| !plot_datasets.is_empty())
            .map(|t| t.name());
        self.current_valid_plot_datasets = plot_datasets;
        self.is_diverged = false;
        self.diagnostic_report = None;
        self.resync_active_hud_callouts();
    }

    /// Coordinates parameter clamping on the active tool via domain hooks or JSON bounds fallback.
    pub fn clamp_parameters(&mut self) {
        self.bump_framework_revision();
        let Some(snapshot) = self.last_valid_snapshot.clone() else {
            self.clamp_without_snapshot();
            return;
        };

        let mut telemetry_details = "Fallback to baseline parameters: zero snapshot fields changed";
        if let Some(tool) = &mut self.active_tool {
            let live_diverged = (!tool.has_finite_parameters())
                .then(|| tool.save_recovery_state())
                .flatten();
            if let Some(ref state_str) = snapshot.tool_state {
                tool.load_recovery_state(state_str);
            }
            if tool.clamp_parameters() {
                telemetry_details =
                    "Clamped instability parameters via domain hook and restored plot datasets";
            } else {
                let constraints = tool.theory().theory_parameters();
                let base_ref = self.baseline_tool_state.as_deref();
                let clamped = snapshot
                    .tool_state
                    .as_deref()
                    .and_then(|s| {
                        crate::framework::clamp::clamp_state_json(s, base_ref, &constraints)
                    })
                    .or_else(|| {
                        live_diverged.as_deref().and_then(|s| {
                            crate::framework::clamp::clamp_state_json(s, base_ref, &constraints)
                        })
                    });
                if let Some(clamped_json) = clamped {
                    tool.load_recovery_state(&clamped_json);
                    telemetry_details =
                        "Clamped snapshot JSON parameters to stable bounds and restored plot datasets";
                } else if let Some(base) = base_ref {
                    tool.load_recovery_state(base);
                }
            }
        }

        self.restore_snapshot_plots(snapshot.plot_datasets);
        emit_recovery_telemetry(
            "SimulationFramework",
            "Clamp Instability Parameters",
            telemetry_details,
        );
    }

    fn clamp_without_snapshot(&mut self) {
        if let (Some(tool), Some(ref base)) = (&mut self.active_tool, &self.baseline_tool_state) {
            tool.load_recovery_state(base);
        } else if let Some(tool) = &mut self.active_tool {
            let _ = tool.clamp_parameters();
        }
        let valid_plots = self.current_valid_plot_datasets.clone();
        self.restore_snapshot_plots(valid_plots);
        emit_recovery_telemetry(
            "SimulationFramework",
            "Clamp Instability Parameters",
            "Fallback to baseline parameters: missing last_valid_snapshot",
        );
    }

    /// Dispatches a StateGuardAction remedy.
    pub fn handle_state_guard_action(&mut self, action: StateGuardAction) {
        match action {
            StateGuardAction::StepBack => {
                self.rollback_step_back();
            }
            StateGuardAction::AutoScaleStepSize => {
                self.rollback_auto_scale_step_size();
            }
            StateGuardAction::ClampParameters => {
                self.clamp_parameters();
            }
            StateGuardAction::ResetParameters => {
                self.reset_parameters();
            }
        }
    }
}
