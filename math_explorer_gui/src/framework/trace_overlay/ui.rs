#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    resolve_trace_color, KpiComparisonRow, ScenarioKpi, TraceSeries, TraceSnapshot,
    TraceSnapshotManager,
};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
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
                Line::new(line_name, PlotPoints::from_shared(series.display_points()))
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
        self.render_ui_with_kpis(ui, &[], || {
            pin_callback().map(|(label, summary, series)| (label, summary, series, Vec::new()))
        })
    }

    pub fn render_ui_with_kpis<F>(
        &mut self,
        ui: &mut egui::Ui,
        current_kpis: &[ScenarioKpi],
        pin_callback: F,
    ) -> bool
    where
        F: FnOnce() -> Option<(String, String, Vec<TraceSeries>, Vec<ScenarioKpi>)>,
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
                let kpi_rows = self.kpi_comparison_rows(current_kpis);
                if !kpi_rows.is_empty() {
                    self.render_kpi_comparison_section(ui, current_kpis, &kpi_rows);
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
        F: FnOnce() -> Option<(String, String, Vec<TraceSeries>, Vec<ScenarioKpi>)>,
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
                if let Some((label, param_summary, series, kpis)) = pin_callback() {
                    if self.add_snapshot_with_kpis(label, param_summary, series, kpis) {
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

    fn render_kpi_comparison_section(
        &self,
        ui: &mut egui::Ui,
        current_kpis: &[ScenarioKpi],
        rows: &[KpiComparisonRow],
    ) {
        ui.add_space(6.0);
        ui.separator();
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("📊 Scenario KPI Comparison")
                    .small()
                    .strong(),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button("⬇ Export KPI CSV")
                    .accessible_hover_text("Export scenario KPI comparison table to CSV")
                    .clicked()
                {
                    let csv = self.export_kpi_csv(current_kpis);
                    ui.ctx().copy_text(csv.clone());
                    #[cfg(not(test))]
                    {
                        let (tx, _rx) = std::sync::mpsc::channel();
                        crate::web_io::spawn_save_as_file_dialog(
                            tx,
                            crate::web_io::SaveAsDialogConfig {
                                filter_name: "CSV Spreadsheet (*.csv)",
                                extensions: &["csv"],
                                fallback_filename: "scenario_kpis.csv",
                                write_error_prefix: "Failed to export KPI CSV",
                            },
                            "scenario_kpis.csv",
                            csv,
                            |path, _| path,
                        );
                    }
                }
            });
        });

        let palette = SemanticPalette::from_visuals(ui.visuals());
        egui::Grid::new(ui.id().with("scenario_kpi_grid"))
            .striped(true)
            .spacing([10.0, 4.0])
            .show(ui, |ui| {
                for col in ["Scenario", "KPI", "Value", "Δ", "Δ%"] {
                    ui.label(egui::RichText::new(col).small().strong());
                }
                ui.end_row();
                for row in rows {
                    render_kpi_grid_row(ui, row, &palette);
                }
            });
    }
}

fn render_kpi_grid_row(ui: &mut egui::Ui, row: &KpiComparisonRow, palette: &SemanticPalette) {
    ui.label(egui::RichText::new(&row.scenario_label).small());
    let kpi_lbl = if row.unit.is_empty() {
        row.kpi_name.clone()
    } else {
        format!("{} ({})", row.kpi_name, row.unit)
    };
    ui.label(egui::RichText::new(kpi_lbl).small());
    ui.monospace(format!("{:.4}", row.value));
    if row.is_reference {
        ui.label(egui::RichText::new("ref").small().color(palette.muted_text));
        ui.label(egui::RichText::new("—").small().color(palette.muted_text));
    } else {
        let delta_color = if row.abs_delta > 1e-12 {
            palette.success
        } else if row.abs_delta < -1e-12 {
            palette.warning
        } else {
            palette.muted_text
        };
        ui.label(
            egui::RichText::new(format!("{:+.4}", row.abs_delta))
                .monospace()
                .small()
                .color(delta_color),
        );
        let pct_txt = row
            .pct_delta
            .map_or_else(|| "N/A".to_string(), |p| format!("{p:+.2}%"));
        ui.label(
            egui::RichText::new(pct_txt)
                .monospace()
                .small()
                .color(delta_color),
        );
    }
    ui.end_row();
}
