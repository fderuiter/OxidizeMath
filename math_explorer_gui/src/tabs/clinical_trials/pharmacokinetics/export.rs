//! Markdown and CSV dosing regimen summary cards and dataset export helpers.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{EntityMode, PharmacokineticsTool};
use crate::web_io::{sanitize_download_filename, spawn_save_as_file_dialog, SaveAsDialogConfig};
use egui_plot::PlotPoint;
use std::fmt::Write as _;
use std::sync::mpsc::{channel, Receiver, TryRecvError};

/// Non-blocking status and channel state for Markdown/CSV dosing card saves and clipboard copies.
#[derive(Default)]
pub(crate) struct CardExportState {
    save_rx: Option<Receiver<Result<String, String>>>,
    pub(crate) status_info: Option<String>,
    pub(crate) status_error: Option<String>,
}

impl CardExportState {
    pub(crate) fn record_info(&mut self, msg: String) {
        self.status_error = None;
        crate::accessibility::announce_status(&msg);
        self.status_info = Some(msg);
    }

    pub(crate) fn record_error(&mut self, err: String) {
        self.status_info = None;
        crate::accessibility::announce_status_with_priority(&err, "assertive");
        self.status_error = Some(err);
    }

    pub(crate) fn poll(&mut self) {
        let Some(rx) = &self.save_rx else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(filename)) => {
                self.save_rx = None;
                self.record_info(format!("Saved dosing summary to {filename}"));
            }
            Ok(Err(err)) => {
                self.save_rx = None;
                self.record_error(err);
            }
            Err(TryRecvError::Disconnected) => {
                self.save_rx = None;
            }
            Err(TryRecvError::Empty) => {}
        }
    }

    pub(crate) fn trigger_save_markdown(&mut self, content: String) {
        let (tx, rx) = channel();
        self.save_rx = Some(rx);
        spawn_save_as_file_dialog(
            tx,
            SaveAsDialogConfig {
                filter_name: "Markdown Regimen Card (*.md)",
                extensions: &["md"],
                fallback_filename: "pk_regimen_card.md",
                write_error_prefix: "Failed to save Markdown regimen card",
            },
            "pk_regimen_card.md",
            content,
            |path, fallback| sanitize_download_filename(&path.to_string_lossy(), fallback),
        );
    }

    pub(crate) fn trigger_save_csv(&mut self, content: String) {
        let (tx, rx) = channel();
        self.save_rx = Some(rx);
        spawn_save_as_file_dialog(
            tx,
            SaveAsDialogConfig {
                filter_name: "CSV Regimen Summary (*.csv)",
                extensions: &["csv"],
                fallback_filename: "pk_regimen_summary.csv",
                write_error_prefix: "Failed to save CSV regimen summary",
            },
            "pk_regimen_summary.csv",
            content,
            |path, fallback| sanitize_download_filename(&path.to_string_lossy(), fallback),
        );
    }
}

/// Formats a standalone Markdown dosing schedule and PK metric reference card.
#[must_use]
pub fn format_markdown_card(tool: &PharmacokineticsTool) -> String {
    let mut out = String::with_capacity(1024);
    let _ = writeln!(out, "# Pharmacokinetics (ADME) Dosing Regimen Card\n");
    let _ = writeln!(
        out,
        "- **Formulation**: {} — {}",
        tool.entity_mode.label(),
        tool.release_mode.label()
    );
    let _ = writeln!(
        out,
        "- **Parameters**: F = {:.2}, D = {:.1} mg, V_d = {:.1} L, k_a = {:.4} 1/h, k_e = {:.4} 1/h",
        tool.bioavailability, tool.dose_mg, tool.volume_l, tool.effective_ka, tool.ke
    );
    let _ = writeln!(
        out,
        "- **Regimen**: N = {} doses, tau = {:.1} h, Horizon = {:.1} h, Therapeutic Window = [{:.2}, {:.2}] mg/L\n",
        tool.n_doses, tool.interval_hours, tool.horizon_hours, tool.mec, tool.mtc
    );
    append_markdown_metrics(&mut out, tool);
    append_markdown_schedule(&mut out, tool);
    out
}

fn append_markdown_metrics(out: &mut String, tool: &PharmacokineticsTool) {
    let m = &tool.metrics;
    let _ = writeln!(out, "## Summary PK Metrics\n");
    let _ = writeln!(out, "| Metric | Value |");
    let _ = writeln!(out, "| :--- | :--- |");
    let _ = writeln!(out, "| Peak Concentration (C_max) | {:.3} mg/L |", m.c_max);
    let _ = writeln!(out, "| Trough Concentration (C_min) | {:.3} mg/L |", m.c_min);
    let _ = writeln!(out, "| Elimination Half-Life (T_1/2) | {:.2} h |", m.half_life_h);
    let _ = writeln!(out, "| Single-Dose Peak Time (T_max) | {:.2} h |", m.t_max_h);
    let _ = writeln!(out, "| Systemic Clearance (CL) | {:.2} L/h |", m.clearance_l_h);
    let _ = writeln!(out, "| Trapezoidal AUC | {:.2} mg*h/L |", m.auc_trapezoidal);
    let _ = writeln!(out, "| Accumulation Ratio (R_ac) | {:.2}x |", m.accumulation_ratio);
    let _ = writeln!(
        out,
        "| Time in Therapeutic Window | {:.1}% |\n",
        m.therapeutic_window_pct
    );
}

fn append_markdown_schedule(out: &mut String, tool: &PharmacokineticsTool) {
    let _ = writeln!(out, "## Dosing Schedule\n");
    let _ = writeln!(out, "| Slot | Time (h) | Administered Dose (mg) | Status |");
    let _ = writeln!(out, "| :--- | :--- | :--- | :--- |");
    for slot in 0..tool.n_doses {
        let time_h = (slot as f64) * tool.interval_hours;
        let missed = tool.missed_slots.binary_search(&slot).is_ok();
        let (dose, status) = if missed {
            (0.0, "Missed")
        } else if slot == 0 && tool.use_loading_dose {
            (tool.loading_dose_mg, "Loading")
        } else {
            (tool.dose_mg, "Maintenance")
        };
        let _ = writeln!(out, "| {} | {:.1} | {:.1} | {} |", slot + 1, time_h, dose, status);
    }
}

/// Formats a CSV dosing schedule and PK summary card.
#[must_use]
pub fn format_csv_card(tool: &PharmacokineticsTool) -> String {
    let m = &tool.metrics;
    let mut out = String::with_capacity(1024);
    let _ = writeln!(out, "section,key,time_h,value,unit,status");
    let _ = writeln!(out, "metric,C_max,,{:.4},mg/L,computed", m.c_max);
    let _ = writeln!(out, "metric,C_min,,{:.4},mg/L,computed", m.c_min);
    let _ = writeln!(out, "metric,half_life,,{:.4},h,computed", m.half_life_h);
    let _ = writeln!(out, "metric,t_max,,{:.4},h,computed", m.t_max_h);
    let _ = writeln!(out, "metric,clearance,,{:.4},L/h,computed", m.clearance_l_h);
    let _ = writeln!(out, "metric,AUC,,{:.4},mg*h/L,computed", m.auc_trapezoidal);
    let _ = writeln!(
        out,
        "metric,accumulation_ratio,,{:.4},ratio,computed",
        m.accumulation_ratio
    );
    let _ = writeln!(
        out,
        "metric,window_time_pct,,{:.2},%,computed",
        m.therapeutic_window_pct
    );
    for slot in 0..tool.n_doses {
        let time_h = (slot as f64) * tool.interval_hours;
        let missed = tool.missed_slots.binary_search(&slot).is_ok();
        let (dose, status) = if missed {
            (0.0, "missed")
        } else if slot == 0 && tool.use_loading_dose {
            (tool.loading_dose_mg, "loading")
        } else {
            (tool.dose_mg, "maintenance")
        };
        let _ = writeln!(
            out,
            "schedule,dose_{},{:.2},{:.2},mg,{}",
            slot + 1,
            time_h,
            dose,
            status
        );
    }
    out
}

#[inline]
fn points_to_pairs(points: &[PlotPoint]) -> Vec<[f64; 2]> {
    points.iter().map(|p| [p.x, p.y]).collect()
}

/// Collects curve datasets (always in physical `mg/L` linear units) for CSV/PNG/SVG/GIF exports.
pub(crate) fn collect_export_datasets(tool: &PharmacokineticsTool) -> Vec<(String, Vec<[f64; 2]>)> {
    let mut datasets = Vec::new();
    if !tool.total_points.is_empty() {
        datasets.push((
            "Total Plasma Concentration (mg/L)".to_string(),
            points_to_pairs(&tool.total_points),
        ));
    }
    if tool.entity_mode == EntityMode::Enantiomer {
        if !tool.d_points.is_empty() {
            datasets.push((
                "d-Isomer Concentration (mg/L)".to_string(),
                points_to_pairs(&tool.d_points),
            ));
        }
        if !tool.l_points.is_empty() {
            datasets.push((
                "l-Isomer Concentration (mg/L)".to_string(),
                points_to_pairs(&tool.l_points),
            ));
        }
    }
    datasets.extend(tool.snapshot_manager.export_datasets());
    datasets
}

/// Builds `TraceSeries` slices from the active linear plasma curve buffers.
pub(crate) fn build_active_trace_series(
    tool: &PharmacokineticsTool,
) -> Vec<crate::framework::TraceSeries> {
    let mut series = Vec::new();
    if !tool.total_points.is_empty() {
        series.push(crate::framework::TraceSeries::new(
            "Total C(t)",
            points_to_pairs(&tool.total_points),
        ));
    }
    if tool.entity_mode == EntityMode::Enantiomer {
        if !tool.d_points.is_empty() {
            series.push(crate::framework::TraceSeries::new(
                "d-Isomer",
                points_to_pairs(&tool.d_points),
            ));
        }
        if !tool.l_points.is_empty() {
            series.push(crate::framework::TraceSeries::new(
                "l-Isomer",
                points_to_pairs(&tool.l_points),
            ));
        }
    }
    series
}

// [cite:clinical_trials_statistics]
