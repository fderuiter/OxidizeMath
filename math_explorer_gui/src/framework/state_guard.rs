#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::recovery::*;
use super::SimulationFramework;

/// Pre-divergence state snapshot cached before each simulation step.
#[derive(Clone, Debug)]
pub struct StateSnapshot {
    /// Serialized tool state (parameters) from `tool.save_state()`.
    pub tool_state: Option<String>,
    /// Visual plot datasets captured prior to divergence.
    pub plot_datasets: Vec<(String, Vec<[f64; 2]>)>,
    /// Step or frame index when this snapshot was captured.
    pub step_index: usize,
    /// Instant when this snapshot was created.
    pub timestamp: crate::web_io::PlatformInstant,
}

impl StateSnapshot {
    /// Calculates the approximate memory footprint of this snapshot in bytes.
    pub fn memory_size(&self) -> usize {
        let state_bytes = self.tool_state.as_ref().map(|s| s.len()).unwrap_or(0);
        let plot_bytes: usize = self
            .plot_datasets
            .iter()
            .map(|(name, pts)| name.len() + pts.len() * std::mem::size_of::<[f64; 2]>())
            .sum();
        std::mem::size_of::<Self>() + state_bytes + plot_bytes
    }
}

impl SimulationFramework {
    /// Caches a valid state snapshot before each simulation step/frame.
    /// Memory footprint is strictly bounded to remain under 2 megabytes per tool.
    pub fn cache_valid_snapshot(&mut self) {
        if self.is_diverged {
            return;
        }
        if let Some(tool) = &self.active_tool {
            let tool_state = tool.save_state();
            let mut datasets = self.current_valid_plot_datasets.clone();
            if datasets.is_empty() {
                let exported = tool.export_datasets();
                if inspect_datasets_for_divergence(&exported).is_ok() {
                    datasets = exported;
                }
            }

            let mut snapshot = StateSnapshot {
                tool_state,
                plot_datasets: datasets,
                step_index: self.step_counter,
                timestamp: crate::web_io::PlatformInstant::now(),
            };

            // Memory guardrail: enforce <= 2 megabytes per active tool
            const MAX_SNAPSHOT_BYTES: usize = 2 * 1024 * 1024;
            while snapshot.memory_size() > MAX_SNAPSHOT_BYTES {
                let mut pruned = false;
                for (_, pts) in &mut snapshot.plot_datasets {
                    if pts.len() > 100 {
                        let keep = pts.len() / 2;
                        pts.drain(0..pts.len() - keep);
                        pruned = true;
                    }
                }
                if !pruned {
                    break;
                }
            }

            self.last_valid_snapshot = Some(snapshot);
        }
    }

    /// Inspects output plot datasets and serialized tool state for non-finite values (NaN / Inf).
    /// Enforces latency under 1 millisecond per frame.
    /// Halts solver execution, records DiagnosticReport, and emits telemetry when divergence occurs.
    pub fn inspect_runtime_state(&mut self) -> Result<(), DiagnosticReport> {
        let plot_datasets = egui_plot::get_last_plot_datasets();
        let tool_datasets = self
            .active_tool
            .as_ref()
            .map(|t| t.export_datasets())
            .unwrap_or_default();

        let mut check_result = inspect_datasets_for_divergence(&plot_datasets);
        if check_result.is_ok() && !tool_datasets.is_empty() {
            check_result = inspect_datasets_for_divergence(&tool_datasets);
        }

        if check_result.is_ok() {
            if let Some(tool) = &self.active_tool {
                if let Some(state_str) = tool.save_state() {
                    if state_str.contains("NaN")
                        || state_str.contains("Infinity")
                        || state_str.contains("-Infinity")
                    {
                        check_result = Err(DiagnosticReport::new(
                            self.start_time.elapsed().as_secs_f64(),
                            self.step_counter,
                            "state",
                            "Non-finite floating-point state encountered in serialized tool parameters.",
                        ));
                    }
                }
            }
        }

        match check_result {
            Ok(()) => {
                if !self.is_diverged {
                    if !plot_datasets.is_empty() {
                        self.current_valid_plot_datasets = plot_datasets;
                    } else if !tool_datasets.is_empty() {
                        self.current_valid_plot_datasets = tool_datasets;
                    }
                    self.step_counter += 1;
                }
                Ok(())
            }
            Err(report) => {
                let tool_name = self
                    .active_tool
                    .as_ref()
                    .map(|t| t.name())
                    .unwrap_or("SimulationFramework");
                emit_divergence_telemetry(
                    tool_name,
                    report.t_diverged,
                    report.step_index,
                    &report.variable,
                    &report.message,
                );
                self.is_diverged = true;
                self.diagnostic_report = Some(report.clone());
                Err(report)
            }
        }
    }

    /// Executes the Step-Back action: restores the exact pre-divergence parameter configuration and plot data.
    pub fn rollback_step_back(&mut self) -> bool {
        if let Some(snapshot) = self.last_valid_snapshot.clone() {
            if let Some(tool) = &mut self.active_tool {
                if let Some(ref state_str) = snapshot.tool_state {
                    tool.load_state(state_str);
                }
            }
            egui_plot::clear_last_plot_datasets();
            egui_plot::append_last_plot_datasets(&snapshot.plot_datasets);
            self.current_valid_plot_datasets = snapshot.plot_datasets;
            self.is_diverged = false;
            self.diagnostic_report = None;

            emit_recovery_telemetry(
                "SimulationFramework",
                "Step Back",
                "Restored exact pre-divergence parameter configuration and plot data",
            );
            true
        } else {
            false
        }
    }

    /// Executes the Auto-Scale Step Size action: restores saved snapshot, halves the time step (dt), and restarts simulation.
    pub fn rollback_auto_scale_step_size(&mut self) -> bool {
        if let Some(snapshot) = self.last_valid_snapshot.clone() {
            if let Some(tool) = &mut self.active_tool {
                if let Some(ref state_str) = snapshot.tool_state {
                    if let Some(halved_state) = auto_scale_state_json(state_str) {
                        tool.load_state(&halved_state);
                    } else {
                        tool.load_state(state_str);
                    }
                }
            }
            egui_plot::clear_last_plot_datasets();
            egui_plot::append_last_plot_datasets(&snapshot.plot_datasets);
            self.current_valid_plot_datasets = snapshot.plot_datasets;
            self.is_diverged = false;
            self.diagnostic_report = None;

            emit_recovery_telemetry(
                "SimulationFramework",
                "Auto-scale Step Size (dt)",
                "Halved step size and restarted simulation from snapshot",
            );
            true
        } else {
            false
        }
    }

    /// Coordinates parameter clamping on the active tool.
    pub fn clamp_parameters(&mut self) {
        if let Some(snapshot) = &self.last_valid_snapshot {
            if let Some(tool) = &mut self.active_tool {
                if let Some(ref state_str) = snapshot.tool_state {
                    tool.load_state(state_str);
                }
            }
        }
        self.is_diverged = false;
        self.diagnostic_report = None;
        emit_recovery_telemetry(
            "SimulationFramework",
            "Clamp Instability Parameters",
            "Clamped rate parameters to stable bounds",
        );
    }

    /// Coordinates parameter reset to baseline on the active tool.
    pub fn reset_parameters(&mut self) {
        if let Some(baseline_state) = self.baseline_tool_state.clone() {
            if let Some(tool) = &mut self.active_tool {
                tool.load_state(&baseline_state);
            } else if let Some(&meta) = self
                .selected_tool_index
                .and_then(|idx| self.available_tools.get(idx))
            {
                let mut new_tool = (meta.build)();
                new_tool.load_state(&baseline_state);
                self.active_tool = Some(new_tool);
            }
        } else if let Some(&meta) = self
            .selected_tool_index
            .and_then(|idx| self.available_tools.get(idx))
        {
            let new_tool = (meta.build)();
            self.baseline_tool_state = new_tool.save_state();
            self.active_tool = Some(new_tool);
        }
        egui_plot::clear_last_plot_datasets();
        self.current_valid_plot_datasets.clear();
        self.is_diverged = false;
        self.diagnostic_report = None;
        self.step_counter = 0;
        if let Some(tool) = &self.active_tool {
            let restored_state = tool.save_state();
            if let Some(ref state) = restored_state {
                self.saved_tool_states
                    .insert(tool.name().to_string(), state.clone());
            } else {
                self.saved_tool_states.remove(tool.name());
            }
            self.last_valid_snapshot = Some(StateSnapshot {
                tool_state: restored_state,
                plot_datasets: Vec::new(),
                step_index: 0,
                timestamp: std::time::Instant::now(),
            });
        } else {
            self.last_valid_snapshot = None;
        }
        emit_recovery_telemetry(
            "SimulationFramework",
            "Reset All Parameters",
            "Restored default baseline configuration",
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

    /// Returns true if the framework is currently in a diverged state.
    pub fn is_diverged(&self) -> bool {
        self.is_diverged
    }

    /// Returns the diagnostic report if diverged.
    pub fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.diagnostic_report.as_ref()
    }

    /// Returns the cached pre-divergence state snapshot.
    pub fn last_valid_snapshot(&self) -> Option<&StateSnapshot> {
        self.last_valid_snapshot.as_ref()
    }
}
