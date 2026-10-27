#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;
use egui_plot::{render_math_label, Line, LineStyle, PlotPoints, PlotUi};
use serde::{Deserialize, Serialize};

/// Maximum pinned trace snapshots permitted per tool instance.
pub const MAX_SNAPSHOTS: usize = 10;

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

/// A single named dataset series within a trace snapshot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceSeries {
    /// Name of the series (e.g. "y(t)" or "Infected").
    pub name: String,
    /// Vector of 2D coordinates `[x, y]` captured at snapshot time.
    pub points: Vec<[f64; 2]>,
}

impl TraceSeries {
    /// Create a new trace series.
    pub fn new(name: impl Into<String>, points: Vec<[f64; 2]>) -> Self {
        Self {
            name: name.into(),
            points,
        }
    }
}

/// A captured snapshot containing pinned simulation curves and metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceSnapshot {
    /// Unique identifier for this snapshot instance.
    pub id: usize,
    /// Title / label of the snapshot (e.g. "Baseline 1").
    pub label: String,
    /// Parameter summary string recorded at time of capture (e.g. "k=1.0, dt=0.05").
    pub param_summary: String,
    /// Captured simulation point buffers.
    pub series: Vec<TraceSeries>,
    /// Whether this snapshot is currently rendered on the plot.
    pub visible: bool,
    /// RGB stroke color assigned to this snapshot.
    pub color_rgb: [u8; 3],
}

impl TraceSnapshot {
    /// Returns the primary display title for this snapshot.
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

/// Manages pinned trace snapshots for interactive simulation tools.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceSnapshotManager {
    snapshots: Vec<TraceSnapshot>,
    next_id: usize,
}

impl Default for TraceSnapshotManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TraceSnapshotManager {
    /// Create a new empty trace snapshot manager.
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            next_id: 1,
        }
    }

    /// Number of pinned snapshots stored.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Whether any pinned snapshots are currently stored.
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Returns true if maximum capacity (10 pinned snapshots) has been reached.
    pub fn is_full(&self) -> bool {
        self.snapshots.len() >= MAX_SNAPSHOTS
    }

    /// Access slice of pinned snapshots.
    pub fn snapshots(&self) -> &[TraceSnapshot] {
        &self.snapshots
    }

    /// Access mutable slice of pinned snapshots.
    pub fn snapshots_mut(&mut self) -> &mut [TraceSnapshot] {
        &mut self.snapshots
    }

    /// Add a new pinned snapshot. Returns `true` on success, or `false` if max capacity was reached.
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

        let palette_idx = (self.snapshots.len()) % SNAPSHOT_PALETTE.len();
        let color_rgb = SNAPSHOT_PALETTE[palette_idx];

        let snapshot = TraceSnapshot {
            id,
            label: label.into(),
            param_summary: param_summary.into(),
            series,
            visible: true,
            color_rgb,
        };

        self.snapshots.push(snapshot);
        true
    }

    /// Toggle visibility of snapshot at `index`.
    pub fn toggle_visibility(&mut self, index: usize) {
        if let Some(snapshot) = self.snapshots.get_mut(index) {
            snapshot.visible = !snapshot.visible;
        }
    }

    /// Remove a snapshot by index.
    pub fn remove_snapshot(&mut self, index: usize) {
        if index < self.snapshots.len() {
            self.snapshots.remove(index);
        }
    }

    /// Remove all pinned snapshots.
    pub fn clear(&mut self) {
        self.snapshots.clear();
    }

    /// Render visible trace snapshots onto an active `egui_plot::PlotUi`.
    pub fn render_on_plot(&self, plot_ui: &mut PlotUi) {
        for snapshot in &self.snapshots {
            if !snapshot.visible {
                continue;
            }

            let color = egui::Color32::from_rgb(
                snapshot.color_rgb[0],
                snapshot.color_rgb[1],
                snapshot.color_rgb[2],
            );

            for series in &snapshot.series {
                if series.points.is_empty() {
                    continue;
                }

                let line_name = if snapshot.series.len() == 1 {
                    format!("{} ({})", series.name, snapshot.display_label())
                } else {
                    format!("{} - {}", snapshot.display_label(), series.name)
                };

                let line = Line::new(line_name, PlotPoints::new(series.points.clone()))
                    .color(color)
                    .style(LineStyle::Dashed { length: 6.0 })
                    .width(1.5_f32);

                plot_ui.line(line);
            }
        }
    }

    /// Renders UI controls for pinning, toggling, deleting, and clearing snapshots.
    ///
    /// The `pin_callback` parameter is invoked when the user clicks "Pin Snapshot".
    /// It should return `Some((label, param_summary, series_vector))` to add a new snapshot.
    #[allow(clippy::too_many_lines)]
    pub fn render_ui<F>(&mut self, ui: &mut egui::Ui, pin_callback: F) -> bool
    where
        F: FnOnce() -> Option<(String, String, Vec<TraceSeries>)>,
    {
        let mut changed = false;

        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.heading("📌 Pinned Trace Overlay");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if !self.is_empty()
                            && ui
                                .button("Clear All")
                                .on_hover_text("Remove all pinned snapshot curves")
                                .clicked()
                        {
                            self.clear();
                            changed = true;
                        }
                    });
                });

                ui.separator();

                ui.horizontal(|ui| {
                    let full = self.is_full();
                    let btn_text = if full {
                        "Pin Snapshot (Max 10)"
                    } else {
                        "📍 Pin Current Snapshot"
                    };

                    let pin_btn = ui.add_enabled(!full, egui::Button::new(btn_text));
                    let pin_btn = if full {
                        pin_btn.on_hover_text("Maximum limit of 10 pinned snapshots reached.")
                    } else {
                        pin_btn.on_hover_text(
                            "Save current trajectory curve as a pinned baseline reference",
                        )
                    };

                    if pin_btn.clicked() && !full {
                        if let Some((label, param_summary, series)) = pin_callback() {
                            if self.add_snapshot(label, param_summary, series) {
                                changed = true;
                            }
                        }
                    }

                    ui.label(format!("{}/10 pinned", self.snapshots.len()));
                });

                if !self.snapshots.is_empty() {
                    ui.add_space(4.0);
                    let mut to_remove = None;

                    for (idx, snapshot) in self.snapshots.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            let eye_symbol = if snapshot.visible { "👁" } else { "🙈" };
                            if ui
                                .button(eye_symbol)
                                .on_hover_text(if snapshot.visible {
                                    "Hide snapshot curve"
                                } else {
                                    "Show snapshot curve"
                                })
                                .clicked()
                            {
                                snapshot.visible = !snapshot.visible;
                                changed = true;
                            }

                            let color = egui::Color32::from_rgb(
                                snapshot.color_rgb[0],
                                snapshot.color_rgb[1],
                                snapshot.color_rgb[2],
                            );

                            ui.painter().circle_filled(
                                ui.cursor().min + egui::vec2(6.0, 8.0),
                                5.0,
                                color,
                            );
                            ui.add_space(12.0);

                            render_math_label(
                                ui,
                                &snapshot.display_label(),
                                egui::TextStyle::Body,
                                ui.visuals().text_color(),
                            );

                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .button("❌")
                                        .on_hover_text("Delete this snapshot")
                                        .clicked()
                                    {
                                        to_remove = Some(idx);
                                    }
                                },
                            );
                        });
                    }

                    if let Some(idx) = to_remove {
                        self.remove_snapshot(idx);
                        changed = true;
                    }
                }
            });
        });

        changed
    }

    /// Returns all visible snapshot datasets formatted as `(name, points)`.
    pub fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut out = Vec::new();
        for snapshot in &self.snapshots {
            if !snapshot.visible {
                continue;
            }
            for series in &snapshot.series {
                let name = if snapshot.series.len() == 1 {
                    format!("{} ({})", series.name, snapshot.display_label())
                } else {
                    format!("{} - {}", snapshot.display_label(), series.name)
                };
                out.push((name, series.points.clone()));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_snapshot_and_capacity_cap() {
        let mut manager = TraceSnapshotManager::new();
        assert_eq!(manager.len(), 0);
        assert!(manager.is_empty());
        assert!(!manager.is_full());

        for i in 1..=MAX_SNAPSHOTS {
            let success = manager.add_snapshot(
                format!("Baseline {}", i),
                format!("k={:.1}", i as f64),
                vec![TraceSeries::new(
                    "y(t)",
                    vec![[0.0, i as f64], [1.0, i as f64 * 2.0]],
                )],
            );
            assert!(success, "Should successfully add snapshot {}", i);
            assert_eq!(manager.len(), i);
        }

        assert!(manager.is_full());
        assert_eq!(manager.len(), MAX_SNAPSHOTS);

        // 11th attempt must fail due to MAX_SNAPSHOTS cap
        let eleventh = manager.add_snapshot(
            "Baseline 11",
            "k=11.0",
            vec![TraceSeries::new("y(t)", vec![[0.0, 11.0]])],
        );
        assert!(
            !eleventh,
            "11th snapshot addition should fail due to MAX_SNAPSHOTS limit"
        );
        assert_eq!(manager.len(), MAX_SNAPSHOTS);
    }

    #[test]
    fn test_toggle_visibility_remove_and_clear() {
        let mut manager = TraceSnapshotManager::new();
        manager.add_snapshot(
            "Baseline 1",
            "k=1.0",
            vec![TraceSeries::new("y(t)", vec![[0.0, 1.0]])],
        );
        manager.add_snapshot(
            "Baseline 2",
            "k=2.0",
            vec![TraceSeries::new("y(t)", vec![[0.0, 2.0]])],
        );

        assert_eq!(manager.len(), 2);
        assert!(manager.snapshots()[0].visible);

        manager.toggle_visibility(0);
        assert!(!manager.snapshots()[0].visible);

        manager.remove_snapshot(0);
        assert_eq!(manager.len(), 1);
        assert_eq!(manager.snapshots()[0].label, "Baseline 2");

        manager.clear();
        assert_eq!(manager.len(), 0);
        assert!(manager.is_empty());
    }

    #[test]
    fn test_export_datasets() {
        let mut manager = TraceSnapshotManager::new();
        manager.add_snapshot(
            "Baseline 1",
            "k=1.0",
            vec![TraceSeries::new("y(t)", vec![[0.0, 1.0], [1.0, 2.0]])],
        );

        let datasets = manager.export_datasets();
        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].0, "y(t) (Baseline 1: k=1.0)");
        assert_eq!(datasets[0].1, vec![[0.0, 1.0], [1.0, 2.0]]);

        // Hide snapshot and verify export excludes hidden trace
        manager.toggle_visibility(0);
        assert!(manager.export_datasets().is_empty());
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut manager = TraceSnapshotManager::new();
        manager.add_snapshot(
            "Baseline 1",
            "k=1.0, dt=0.05",
            vec![TraceSeries::new("y(t)", vec![[0.0, 1.0], [1.0, 2.0]])],
        );

        let json = serde_json::to_string(&manager).expect("Serialization failed");
        let restored: TraceSnapshotManager =
            serde_json::from_str(&json).expect("Deserialization failed");

        assert_eq!(restored.len(), 1);
        assert_eq!(restored.snapshots()[0].label, "Baseline 1");
        assert_eq!(restored.snapshots()[0].param_summary, "k=1.0, dt=0.05");
        assert_eq!(
            restored.snapshots()[0].series[0].points,
            vec![[0.0, 1.0], [1.0, 2.0]]
        );
    }
}
