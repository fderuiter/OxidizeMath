#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[path = "trace_overlay/ui.rs"]
mod ui;

/// Maximum pinned trace snapshots permitted per tool instance.
pub const MAX_SNAPSHOTS: usize = 10;

/// Maximum automated history runs retained in FIFO buffer per tool instance.
pub const MAX_HISTORY_RUNS: usize = 5;

/// Default color palette for trace snapshots.
pub const SNAPSHOT_PALETTE: &[[u8; 3]] = &[
    [255, 127, 14],  // Orange
    [148, 103, 189], // Purple
    [227, 119, 194], // Pink
    [44, 160, 44],   // Green
    [23, 190, 207],  // Cyan
    [188, 189, 34],  // Olive
    [140, 86, 75],   // Brown
    [255, 187, 120], // Light Orange
    [197, 176, 213], // Soft Purple
    [196, 156, 148], // Soft Brown
];

pub(super) fn resolve_trace_color(
    rgb: [u8; 3],
    index: usize,
    visuals: &egui::Visuals,
) -> egui::Color32 {
    let raw = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
    if crate::app::contrast_ratio(raw, visuals.extreme_bg_color) >= 3.0 {
        raw
    } else {
        crate::app::SemanticPalette::from_visuals(visuals).series_color(index)
    }
}

/// Shared, reference-counted coordinate slice enabling O(1) snapshot cloning.
pub type TracePoints = egui_plot::SharedPoints;

mod shared_points_serde {
    use super::TracePoints;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<S: Serializer>(
        points: &TracePoints,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        points.as_slice().serialize(serializer)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<TracePoints, D::Error> {
        Vec::<[f64; 2]>::deserialize(deserializer).map(TracePoints::from)
    }
}

/// A single named dataset series within a trace snapshot.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceSeries {
    pub name: String,
    #[serde(with = "shared_points_serde")]
    pub points: TracePoints,
    #[serde(skip)]
    display_cache: std::sync::OnceLock<Arc<Vec<egui_plot::PlotPoint>>>,
}

impl PartialEq for TraceSeries {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.points == other.points
    }
}

impl TraceSeries {
    pub fn new(name: impl Into<String>, points: impl Into<TracePoints>) -> Self {
        Self {
            name: name.into(),
            points: points.into(),
            display_cache: std::sync::OnceLock::new(),
        }
    }

    /// Returns a min-max decimated display buffer capped at [`crate::framework::MAX_DISPLAY_POLYLINE_POINTS`].
    pub fn display_points(&self) -> Arc<Vec<egui_plot::PlotPoint>> {
        Arc::clone(self.display_cache.get_or_init(|| {
            crate::framework::decimate_pairs_minmax(
                &self.points,
                crate::framework::MAX_DISPLAY_POLYLINE_POINTS,
            )
        }))
    }
}

/// A captured snapshot containing simulation curves and metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceSnapshot {
    pub id: usize,
    pub label: String,
    pub param_summary: String,
    pub series: Vec<TraceSeries>,
    pub visible: bool,
    pub color_rgb: [u8; 3],
}

impl TraceSnapshot {
    pub fn display_label(&self) -> String {
        if self.param_summary.is_empty() {
            self.label.clone()
        } else if self.label.is_empty() {
            self.param_summary.clone()
        } else {
            format!("{}: {}", self.label, self.param_summary)
        }
    }
}

/// Manages pinned trace snapshots and auto history runs for interactive tools.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceSnapshotManager {
    pub(super) snapshots: Vec<TraceSnapshot>,
    #[serde(default)]
    pub(super) history_runs: Vec<TraceSnapshot>,
    next_id: usize,
    #[serde(default)]
    history_counter: usize,
}

impl Default for TraceSnapshotManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TraceSnapshotManager {
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            history_runs: Vec::new(),
            next_id: 1,
            history_counter: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty() && self.history_runs.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.snapshots.len() >= MAX_SNAPSHOTS
    }

    pub fn snapshots(&self) -> &[TraceSnapshot] {
        &self.snapshots
    }

    pub fn snapshots_mut(&mut self) -> &mut [TraceSnapshot] {
        &mut self.snapshots
    }

    pub fn add_snapshot(
        &mut self,
        label: impl Into<String>,
        param_summary: impl Into<String>,
        series: Vec<TraceSeries>,
    ) -> bool {
        if self.is_full() {
            return false;
        }

        let id = self.next_id;
        self.next_id += 1;
        let palette_idx = self.snapshots.len() % SNAPSHOT_PALETTE.len();
        let color_rgb = SNAPSHOT_PALETTE[palette_idx];

        self.snapshots.push(TraceSnapshot {
            id,
            label: label.into(),
            param_summary: param_summary.into(),
            series,
            visible: true,
            color_rgb,
        });
        true
    }

    pub fn toggle_visibility(&mut self, index: usize) {
        if let Some(snapshot) = self.snapshots.get_mut(index) {
            snapshot.visible = !snapshot.visible;
        }
    }

    pub fn remove_snapshot(&mut self, index: usize) {
        if index < self.snapshots.len() {
            self.snapshots.remove(index);
        }
    }

    pub fn clear(&mut self) {
        self.snapshots.clear();
    }

    pub fn push_history_run(&mut self, param_summary: impl Into<String>, series: Vec<TraceSeries>) {
        if series.is_empty() || series.iter().all(|s| s.points.is_empty()) {
            return;
        }
        if self.history_runs.len() >= MAX_HISTORY_RUNS {
            self.history_runs.remove(0);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.history_counter += 1;
        let color_rgb = SNAPSHOT_PALETTE[self.history_counter % SNAPSHOT_PALETTE.len()];
        self.history_runs.push(TraceSnapshot {
            id,
            label: format!("Run {}", self.history_counter),
            param_summary: param_summary.into(),
            series,
            visible: true,
            color_rgb,
        });
    }

    pub fn push_history(&mut self, param_summary: impl Into<String>, series: Vec<TraceSeries>) {
        self.push_history_run(param_summary, series);
    }

    pub fn history_runs(&self) -> &[TraceSnapshot] {
        &self.history_runs
    }

    pub fn history_runs_mut(&mut self) -> &mut [TraceSnapshot] {
        &mut self.history_runs
    }

    pub fn history_len(&self) -> usize {
        self.history_runs.len()
    }

    pub fn clear_history(&mut self) {
        self.history_runs.clear();
    }

    pub fn toggle_history_visibility(&mut self, index: usize) {
        if let Some(run) = self.history_runs.get_mut(index) {
            run.visible = !run.visible;
        }
    }

    pub fn remove_history_run(&mut self, index: usize) {
        if index < self.history_runs.len() {
            self.history_runs.remove(index);
        }
    }

    pub fn pin_history_run(&mut self, index: usize) -> bool {
        if self.is_full() || index >= self.history_runs.len() {
            return false;
        }
        let run = self.history_runs[index].clone();
        let label = format!("Baseline {}", self.snapshots.len() + 1);
        self.add_snapshot(label, run.param_summary, run.series)
    }

    pub fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut out = Vec::new();
        for run in &self.history_runs {
            if !run.visible {
                continue;
            }
            for series in &run.series {
                let name = if run.series.len() == 1 {
                    format!("{} ({})", series.name, run.display_label())
                } else {
                    format!("{} - {}", run.display_label(), series.name)
                };
                out.push((name, series.points.to_vec()));
            }
        }
        for snapshot in &self.snapshots {
            if !snapshot.visible {
                continue;
            }
            for series in &snapshot.series {
                let name = if snapshot.series.len() == 1 {
                    format!("{} ({})", series.name, snapshot.display_label())
                } else {
                    format!(
                        "{} - {}",
                        snapshot.display_label(),
                        snapshot.name_or_series(series)
                    )
                };
                out.push((name, series.points.to_vec()));
            }
        }
        out
    }
}

impl TraceSnapshot {
    fn name_or_series<'a>(&self, series: &'a TraceSeries) -> &'a str {
        &series.name
    }
}

#[cfg(test)]
#[path = "trace_overlay_tests.rs"]
mod tests;
