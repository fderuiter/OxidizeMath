#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;
use egui_plot::{render_math_label, Line, LineStyle, PlotPoints, PlotUi};
use serde::{Deserialize, Serialize};

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

/// A single named dataset series within a trace snapshot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceSeries {
    pub name: String,
    pub points: Vec<[f64; 2]>,
}

impl TraceSeries {
    pub fn new(name: impl Into<String>, points: Vec<[f64; 2]>) -> Self {
        Self {
            name: name.into(),
            points,
        }
    }
}

/// A captured snapshot containing simulation curves and metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
pub struct TraceSnapshotManager {
    snapshots: Vec<TraceSnapshot>,
    #[serde(default)]
    history_runs: Vec<TraceSnapshot>,
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

    pub fn render_on_plot(&self, plot_ui: &mut PlotUi) {
        for run in &self.history_runs {
            if !run.visible {
                continue;
            }
            let color = egui::Color32::from_rgba_unmultiplied(
                run.color_rgb[0],
                run.color_rgb[1],
                run.color_rgb[2],
                160,
            );
            for series in &run.series {
                if series.points.is_empty() {
                    continue;
                }
                let line_name = if run.series.len() == 1 {
                    format!("{} ({})", series.name, run.display_label())
                } else {
                    format!("{} - {}", run.display_label(), series.name)
                };
                plot_ui.line(
                    Line::new(line_name, PlotPoints::new(series.points.clone()))
                        .color(color)
                        .style(LineStyle::Dashed { length: 5.0 })
                        .width(1.2_f32),
                );
            }
        }

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
                plot_ui.line(
                    Line::new(line_name, PlotPoints::new(series.points.clone()))
                        .color(color)
                        .style(LineStyle::Dashed { length: 6.0 })
                        .width(1.5_f32),
                );
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    pub fn render_ui<F>(&mut self, ui: &mut egui::Ui, pin_callback: F) -> bool
    where
        F: FnOnce() -> Option<(String, String, Vec<TraceSeries>)>,
    {
        let mut changed = false;

        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.heading("📌 Trace Overlay & Run History");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if !self.is_empty()
                            && ui
                                .button("Clear All")
                                .on_hover_text("Remove all history ghost runs and pinned baselines")
                                .clicked()
                        {
                            self.clear();
                            self.clear_history();
                            changed = true;
                        }
                    });
                });

                ui.separator();

                ui.horizontal(|ui| {
                    let full = self.is_full();
                    let btn_text = if full {
                        "Pin Baseline (Max 10)"
                    } else {
                        "📍 Pin Current Snapshot"
                    };

                    let pin_btn = ui.add_enabled(!full, egui::Button::new(btn_text));
                    let pin_btn = if full {
                        pin_btn.on_hover_text("Maximum limit of 10 pinned baselines reached.")
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

                if !self.history_runs.is_empty() {
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new("🕒 Auto History Runs (max 5 ghost traces)")
                            .small()
                            .strong(),
                    );
                    let mut history_to_remove = None;
                    let mut history_to_pin = None;
                    let is_full = self.is_full();

                    for (idx, run) in self.history_runs.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            let eye_symbol = if run.visible { "👁" } else { "🙈" };
                            if ui
                                .button(eye_symbol)
                                .on_hover_text("Toggle ghost trace visibility")
                                .clicked()
                            {
                                run.visible = !run.visible;
                                changed = true;
                            }
                            let color = egui::Color32::from_rgb(
                                run.color_rgb[0],
                                run.color_rgb[1],
                                run.color_rgb[2],
                            );
                            ui.painter().circle_filled(
                                ui.cursor().min + egui::vec2(6.0, 8.0),
                                4.0,
                                color,
                            );
                            ui.add_space(10.0);
                            ui.label(egui::RichText::new(run.display_label()).small());

                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .button("❌")
                                        .on_hover_text("Delete this history run")
                                        .clicked()
                                    {
                                        history_to_remove = Some(idx);
                                    }
                                    if !is_full
                                        && ui
                                            .button("📍 Pin")
                                            .on_hover_text("Pin this history run as baseline")
                                            .clicked()
                                    {
                                        history_to_pin = Some(idx);
                                    }
                                },
                            );
                        });
                    }

                    if let Some(idx) = history_to_pin {
                        self.pin_history_run(idx);
                        changed = true;
                    }
                    if let Some(idx) = history_to_remove {
                        self.remove_history_run(idx);
                        changed = true;
                    }
                }

                if !self.snapshots.is_empty() {
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new("📌 Pinned Baselines").small().strong());
                    let mut to_remove = None;

                    for (idx, snapshot) in self.snapshots.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            let eye_symbol = if snapshot.visible { "👁" } else { "🙈" };
                            if ui
                                .button(eye_symbol)
                                .on_hover_text("Toggle baseline visibility")
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
                out.push((name, series.points.clone()));
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
                    format!("{} - {}", snapshot.display_label(), series.name)
                };
                out.push((name, series.points.clone()));
            }
        }
        out
    }
}

#[cfg(test)]
#[path = "trace_overlay_tests.rs"]
mod tests;
