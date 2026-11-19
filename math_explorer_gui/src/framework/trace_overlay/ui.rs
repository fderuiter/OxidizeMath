#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{resolve_trace_color, TraceSeries, TraceSnapshot, TraceSnapshotManager};
use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use egui_plot::{render_math_label, Line, LineStyle, PlotPoints, PlotUi};

impl TraceSnapshotManager {
    pub fn render_on_plot(&self, plot_ui: &mut PlotUi) {
        let visuals = plot_ui.ctx().style().visuals.clone();
        for (idx, run) in self.history_runs.iter().enumerate() {
            if !run.visible {
                continue;
            }
            let base = resolve_trace_color(run.color_rgb, idx, &visuals);
            let color = egui::Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), 160);
            Self::plot_snapshot_series(plot_ui, run, color, 5.0, 1.2_f32);
        }

        for (idx, snapshot) in self.snapshots.iter().enumerate() {
            if !snapshot.visible {
                continue;
            }
            let color = resolve_trace_color(snapshot.color_rgb, idx, &visuals);
            Self::plot_snapshot_series(plot_ui, snapshot, color, 6.0, 1.5_f32);
        }
    }

    fn plot_snapshot_series(
        plot_ui: &mut PlotUi,
        snapshot: &TraceSnapshot,
        color: egui::Color32,
        dash_len: f32,
        width: f32,
    ) {
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
                Line::new(
                    line_name,
                    PlotPoints::from_iter(series.points.iter().copied()),
                )
                .color(color)
                .style(LineStyle::Dashed { length: dash_len })
                .width(width),
            );
        }
    }

    pub fn render_ui<F>(&mut self, ui: &mut egui::Ui, pin_callback: F) -> bool
    where
        F: FnOnce() -> Option<(String, String, Vec<TraceSeries>)>,
    {
        let mut changed = false;
        ui.group(|ui| {
            ui.vertical(|ui| {
                changed |= self.render_header_bar(ui);
                ui.separator();
                changed |= self.render_pin_toolbar(ui, pin_callback);
                if !self.history_runs.is_empty() {
                    changed |= self.render_history_section(ui);
                }
                if !self.snapshots.is_empty() {
                    changed |= self.render_pinned_section(ui);
                }
            });
        });
        changed
    }

    fn render_header_bar(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.heading("📌 Trace Overlay & Run History");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if !self.is_empty()
                    && ui
                        .button("Clear All")
                        .accessible_hover_text("Remove all history ghost runs and pinned baselines")
                        .clicked()
                {
                    self.clear();
                    self.clear_history();
                    changed = true;
                }
            });
        });
        changed
    }

    fn render_pin_toolbar<F>(&mut self, ui: &mut egui::Ui, pin_callback: F) -> bool
    where
        F: FnOnce() -> Option<(String, String, Vec<TraceSeries>)>,
    {
        let mut changed = false;
        ui.horizontal(|ui| {
            let full = self.is_full();
            let btn_text = if full {
                "Pin Baseline (Max 10)"
            } else {
                "📍 Pin Current Snapshot"
            };
            let pin_hover = if full {
                "Maximum limit of 10 pinned baselines reached."
            } else {
                "Save current trajectory curve as a pinned baseline reference"
            };
            let pin_btn = ui
                .add_enabled(!full, egui::Button::new(btn_text))
                .accessible_hover_text(pin_hover);

            if pin_btn.clicked() && !full {
                if let Some((label, param_summary, series)) = pin_callback() {
                    if self.add_snapshot(label, param_summary, series) {
                        changed = true;
                    }
                }
            }
            ui.label(format!("{}/10 pinned", self.snapshots.len()));
        });
        changed
    }

    fn render_history_section(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
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
                    .accessible_hover_text("Toggle ghost trace visibility")
                    .clicked()
                {
                    run.visible = !run.visible;
                    changed = true;
                }
                let color = resolve_trace_color(run.color_rgb, idx, ui.visuals());
                ui.painter()
                    .circle_filled(ui.cursor().min + egui::vec2(6.0, 8.0), 4.0, color);
                ui.add_space(10.0);
                ui.label(egui::RichText::new(run.display_label()).small());

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .button("❌")
                        .accessible_hover_text("Delete this history run")
                        .clicked()
                    {
                        history_to_remove = Some(idx);
                    }
                    if !is_full
                        && ui
                            .button("📍 Pin")
                            .accessible_hover_text("Pin this history run as baseline")
                            .clicked()
                    {
                        history_to_pin = Some(idx);
                    }
                });
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
        changed
    }

    fn render_pinned_section(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.add_space(4.0);
        ui.label(egui::RichText::new("📌 Pinned Baselines").small().strong());
        let mut to_remove = None;

        for (idx, snapshot) in self.snapshots.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                let eye_symbol = if snapshot.visible { "👁" } else { "🙈" };
                if ui
                    .button(eye_symbol)
                    .accessible_hover_text("Toggle baseline visibility")
                    .clicked()
                {
                    snapshot.visible = !snapshot.visible;
                    changed = true;
                }
                let color = resolve_trace_color(snapshot.color_rgb, idx, ui.visuals());
                ui.painter()
                    .circle_filled(ui.cursor().min + egui::vec2(6.0, 8.0), 5.0, color);
                ui.add_space(12.0);
                render_math_label(
                    ui,
                    &snapshot.display_label(),
                    egui::TextStyle::Body,
                    ui.visuals().text_color(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .button("❌")
                        .accessible_hover_text("Delete this snapshot")
                        .clicked()
                    {
                        to_remove = Some(idx);
                    }
                });
            });
        }

        if let Some(idx) = to_remove {
            self.remove_snapshot(idx);
            changed = true;
        }
        changed
    }
}
