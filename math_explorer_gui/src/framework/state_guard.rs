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
    pub plot_datasets: Vec<(String, egui_plot::SharedPoints)>,
    /// Step or frame index when this snapshot was captured.
    pub step_index: usize,
    /// Instant when this snapshot was created.
    pub timestamp: crate::web_io::PlatformInstant,
}

pub(super) fn shared_datasets_ptr_eq(
    a: &[(String, egui_plot::SharedPoints)],
    b: &[(String, egui_plot::SharedPoints)],
) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|((na, pa), (nb, pb))| na == nb && pa.ptr_eq(pb))
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
        if self.memory_size() > MAX_SNAPSHOT_BYTES {
            let mut target_lens: Vec<usize> = self
                .plot_datasets
                .iter()
                .map(|(_, pts)| pts.len())
                .collect();
            let state_bytes = self.tool_state.as_ref().map_or(0, String::len);
            let name_bytes: usize = self.plot_datasets.iter().map(|(n, _)| n.len()).sum();
            let fixed = std::mem::size_of::<Self>() + state_bytes + name_bytes;
            while fixed
                + target_lens
                    .iter()
                    .map(|len| len * std::mem::size_of::<[f64; 2]>())
                    .sum::<usize>()
                > MAX_SNAPSHOT_BYTES
            {
                let mut pruned = false;
                for len in &mut target_lens {
                    if *len > 100 {
                        *len /= 2;
                        pruned = true;
                    }
                }
                if !pruned {
                    break;
                }
            }
            for ((_, pts), keep) in self.plot_datasets.iter_mut().zip(target_lens) {
                if keep < pts.len() {
                    *pts = pts.tail(keep);
                }
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
    fn refresh_snapshot_datasets_only(&mut self) {
        let Some(snap) = &mut self.last_valid_snapshot else {
            return;
        };
        if !self.current_valid_plot_datasets.is_empty()
            && !shared_datasets_ptr_eq(
                &self.rev_cache.last_snapshot_source_datasets,
                &self.current_valid_plot_datasets,
            )
            && inspect_shared_datasets_for_divergence(&self.current_valid_plot_datasets, &[])
                .is_ok()
        {
            self.rev_cache.last_snapshot_source_datasets = self.current_valid_plot_datasets.clone();
            snap.plot_datasets = self.current_valid_plot_datasets.clone();
            snap.enforce_memory_budget();
        }
    }

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
            self.refresh_snapshot_datasets_only();
            return;
        }
        let (name, rev) = (tool.name(), tool.state_revision());
        let reuse_cached_state = rev > 0
            && self
                .rev_cache
                .last_snapshot_rev
                .is_some_and(|(n, r)| n == name && r == rev)
            && self.last_valid_snapshot.is_some();

        if reuse_cached_state
            && !self.current_valid_plot_datasets.is_empty()
            && shared_datasets_ptr_eq(
                &self.rev_cache.last_snapshot_source_datasets,
                &self.current_valid_plot_datasets,
            )
        {
            return;
        }

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
            let exported: Vec<(String, egui_plot::SharedPoints)> = tool
                .export_datasets()
                .into_iter()
                .map(|(n, pts)| (n, egui_plot::SharedPoints::from(pts)))
                .collect();
            if inspect_shared_datasets_for_divergence(&exported, &[]).is_ok() {
                datasets = exported;
            }
        }

        self.rev_cache.last_snapshot_source_datasets = datasets.clone();
        let mut snapshot = StateSnapshot {
            tool_state,
            plot_datasets: datasets,
            step_index: self.step_counter,
            timestamp: crate::web_io::PlatformInstant::now(),
        };
        snapshot.enforce_memory_budget();
        self.last_valid_snapshot = Some(snapshot);
    }

    fn commit_valid_runtime_datasets(
        &mut self,
        plot_datasets: Vec<(String, egui_plot::SharedPoints)>,
        tool_datasets: Vec<(String, egui_plot::SharedPoints)>,
    ) {
        if self.is_diverged {
            return;
        }
        let active_name = self.active_tool.as_ref().map(|t| t.name());
        if !plot_datasets.is_empty() {
            if !shared_datasets_ptr_eq(&plot_datasets, &self.current_valid_plot_datasets) {
                self.current_valid_plot_datasets = plot_datasets;
            }
            self.sync_state.datasets_owner_tool = active_name;
        } else if !tool_datasets.is_empty() {
            self.current_valid_plot_datasets = tool_datasets;
            self.sync_state.datasets_owner_tool = active_name;
        }
        self.step_counter += 1;
    }

    /// Inspects active tool parameters via `has_finite_parameters` and output plot datasets for `NaN`/`Inf`.
    /// Enforces latency under 1 millisecond per frame without calling `InteractiveTool::save_state`.
    pub fn inspect_runtime_state(&mut self) -> Result<(), DiagnosticReport> {
        let plot_datasets = egui_plot::get_last_plot_shared_datasets();
        let mut check_result = inspect_shared_datasets_for_divergence(
            &plot_datasets,
            &self.current_valid_plot_datasets,
        );
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
                    tool_datasets = tool
                        .export_datasets()
                        .into_iter()
                        .map(|(n, pts)| (n, egui_plot::SharedPoints::from(pts)))
                        .collect();
                    if !tool_datasets.is_empty() {
                        check_result = inspect_shared_datasets_for_divergence(
                            &tool_datasets,
                            &self.current_valid_plot_datasets,
                        );
                    }
                }
            }
        }

        match check_result {
            Ok(()) => {
                self.commit_valid_runtime_datasets(plot_datasets, tool_datasets);
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
