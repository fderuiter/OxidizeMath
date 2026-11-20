#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::recovery::*;
use super::SimulationFramework;

#[path = "state_guard/rollback.rs"]
mod rollback;

/// Maximum permitted memory footprint (2 MiB) for a cached recovery snapshot.
pub const MAX_SNAPSHOT_BYTES: usize = 2 * 1024 * 1024;

/// Pre-divergence state snapshot cached before each simulation step.
#[derive(Clone, Debug)]
pub struct StateSnapshot {
    /// Serialized parameter-only recovery state from `tool.save_recovery_state()`.
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
        let state_bytes = self.tool_state.as_ref().map_or(0, String::len);
        let plot_bytes: usize = self
            .plot_datasets
            .iter()
            .map(|(name, pts)| name.len() + pts.len() * std::mem::size_of::<[f64; 2]>())
            .sum();
        std::mem::size_of::<Self>() + state_bytes + plot_bytes
    }

    pub(super) fn enforce_memory_budget(&mut self) {
        while self.memory_size() > MAX_SNAPSHOT_BYTES {
            let mut pruned = false;
            for (_, pts) in &mut self.plot_datasets {
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
        if self.memory_size() > MAX_SNAPSHOT_BYTES {
            self.plot_datasets.clear();
        }
        if self.memory_size() > MAX_SNAPSHOT_BYTES {
            self.tool_state = None;
        }
    }
}

impl SimulationFramework {
    /// Caches a valid parameter-only state snapshot before each simulation step/frame.
    /// Memory footprint is strictly bounded to remain at or below 2 megabytes per tool.
    pub fn cache_valid_snapshot(&mut self) {
        if self.is_diverged {
            return;
        }
        let Some(tool) = &self.active_tool else {
            return;
        };
        if !tool.has_finite_parameters() {
            if let Some(snap) = &mut self.last_valid_snapshot {
                if !self.current_valid_plot_datasets.is_empty()
                    && inspect_datasets_for_divergence(&self.current_valid_plot_datasets).is_ok()
                {
                    snap.plot_datasets = self.current_valid_plot_datasets.clone();
                    snap.enforce_memory_budget();
                }
            }
            return;
        }
        let name = tool.name();
        let rev = tool.state_revision();
        let reuse_cached_state = rev > 0
            && self
                .rev_cache
                .last_snapshot_rev
                .is_some_and(|(n, r)| n == name && r == rev)
            && self.last_valid_snapshot.is_some();

        let tool_state = if reuse_cached_state {
            self.last_valid_snapshot
                .as_ref()
                .and_then(|s| s.tool_state.clone())
        } else {
            let saved = tool.save_recovery_state();
            if rev > 0 {
                self.rev_cache.last_snapshot_rev = Some((name, rev));
            }
            saved
        };

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
        snapshot.enforce_memory_budget();
        self.last_valid_snapshot = Some(snapshot);
    }

    /// Inspects active tool parameters via `has_finite_parameters` and output plot datasets for `NaN`/`Inf`.
    /// Enforces latency under 1 millisecond per frame without calling `InteractiveTool::save_state`.
    pub fn inspect_runtime_state(&mut self) -> Result<(), DiagnosticReport> {
        let plot_datasets = egui_plot::get_last_plot_datasets();
        let mut check_result = inspect_datasets_for_divergence(&plot_datasets);
        let mut tool_datasets = Vec::new();

        if check_result.is_ok() {
            if let Some(tool) = &self.active_tool {
                if !tool.has_finite_parameters() {
                    check_result = Err(DiagnosticReport::new(
                        self.start_time.elapsed().as_secs_f64(),
                        self.step_counter,
                        "state",
                        "Non-finite floating-point state encountered in active tool parameters.",
                    ));
                } else if plot_datasets.is_empty() && tool.state_revision() == 0 {
                    tool_datasets = tool.export_datasets();
                    if !tool_datasets.is_empty() {
                        check_result = inspect_datasets_for_divergence(&tool_datasets);
                    }
                }
            }
        }

        match check_result {
            Ok(()) => {
                if !self.is_diverged {
                    let active_name = self.active_tool.as_ref().map(|t| t.name());
                    if !plot_datasets.is_empty() {
                        self.current_valid_plot_datasets = plot_datasets;
                        self.sync_state.datasets_owner_tool = active_name;
                    } else if !tool_datasets.is_empty() {
                        self.current_valid_plot_datasets = tool_datasets;
                        self.sync_state.datasets_owner_tool = active_name;
                    }
                    self.step_counter += 1;
                }
                Ok(())
            }
            Err(report) => {
                let tool_name = self
                    .active_tool
                    .as_ref()
                    .map_or("SimulationFramework", |t| t.name());
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
