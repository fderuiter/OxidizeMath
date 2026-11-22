//! Offscreen SVG/PNG plot primitives, dataset exports, and `web_io` CSV/JSON takeaway artifacts.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::sync::mpsc::Sender;

use egui_plot::ExportPrimitive;
use math_explorer::applied::pharmacokinetics::ExposureSummary;
use serde::Serialize;

use crate::app::{SemanticPalette, ThemeMode};

use super::ComputedCurves;
use super::state::{PkModelKind, RegimenState};

const DISCLAIMER: &str =
    "Educational ADME mathematical simulation only; not clinical prescribing advice.";

#[derive(Serialize)]
struct ExposureReportPayload<'a> {
    disclaimer: &'static str,
    model: &'static str,
    parameters: ReportParameters,
    therapeutic_window_mg_l: ReportWindow,
    exposure_metrics: ReportMetrics,
    schedule: Vec<ReportEvent>,
    notes: &'a str,
}

#[derive(Serialize)]
struct ReportParameters {
    maintenance_dose_mg: f64,
    tau_h: f64,
    horizon_h: f64,
    bioavailability_f: f64,
    ka_per_h: f64,
    ke_per_h: f64,
    volume_l: f64,
    xr_lag_h: f64,
    xr_f1: f64,
    chiral_f_d: f64,
    chiral_l_ke_per_h: f64,
}

#[derive(Serialize)]
struct ReportWindow {
    mec_mg_l: f64,
    mtc_mg_l: f64,
}

#[derive(Serialize)]
struct ReportMetrics {
    c_max_mg_l: f64,
    t_peak_h: f64,
    c_min_mg_l: f64,
    auc_0_tau_mg_h_l: f64,
    auc_total_mg_h_l: f64,
    accumulation_ratio_theoretical: f64,
    accumulation_ratio_observed: f64,
    time_in_therapeutic_range_pct: f64,
    time_in_therapeutic_range_h: f64,
    toxic_pct: f64,
    toxic_hours: f64,
    subtherapeutic_pct: f64,
    subtherapeutic_hours: f64,
}

#[derive(Serialize)]
struct ReportEvent {
    index: usize,
    time_h: f64,
    dose_mg: f64,
    formulation: &'static str,
}

/// Formats the scheduled [`RegimenState`] events and sampled pre-dose concentrations as CSV.
pub fn format_schedule_csv(state: &RegimenState, curves: &ComputedCurves) -> String {
    let mut csv = String::from(
        "event_index,time_h,dose_mg,formulation,lag_time_h,f1,f2,c_at_dose_mg_l\n",
    );
    let f1 = state.xr_f1;
    let f2 = (1.0 - f1).max(0.0);
    for (idx, ev) in state.events.iter().enumerate() {
        let use_xr = ev.xr || state.model == PkModelKind::BiphasicXr;
        let (form_str, lag, p1, p2) = if use_xr {
            ("XR", state.xr_lag_h, f1, f2)
        } else {
            ("IR", 0.0, 1.0, 0.0)
        };
        let c_at_t = sample_nearest(&curves.times, &curves.total, ev.t);
        use std::fmt::Write;
        let _ = writeln!(
            &mut csv,
            "{},{:.3},{:.3},{},{:.3},{:.3},{:.3},{:.6}",
            idx + 1,
            ev.t,
            ev.d,
            form_str,
            lag,
            p1,
            p2,
            c_at_t
        );
    }
    csv
}

fn sample_nearest(times: &[f64], values: &[f64], target_t: f64) -> f64 {
    if times.is_empty() || times.len() != values.len() {
        return 0.0;
    }
    let mut best_i = 0usize;
    let mut best_diff = (times[0] - target_t).abs();
    for (i, &t) in times.iter().enumerate().skip(1) {
        let diff = (t - target_t).abs();
        if diff < best_diff {
            best_diff = diff;
            best_i = i;
        }
    }
    values[best_i]
}

/// Formats a structured JSON exposure and therapeutic-window summary report.
pub fn format_exposure_report_json(
    state: &RegimenState,
    summary: &ExposureSummary,
) -> Result<String, String> {
    let schedule = state
        .events
        .iter()
        .enumerate()
        .map(|(idx, ev)| ReportEvent {
            index: idx + 1,
            time_h: ev.t,
            dose_mg: ev.d,
            formulation: if ev.xr || state.model == PkModelKind::BiphasicXr {
                "ExtendedRelease"
            } else {
                "ImmediateRelease"
            },
        })
        .collect();

    let payload = ExposureReportPayload {
        disclaimer: DISCLAIMER,
        model: state.model.label(),
        parameters: ReportParameters {
            maintenance_dose_mg: state.maintenance_dose_mg,
            tau_h: state.tau_h,
            horizon_h: state.horizon_h,
            bioavailability_f: state.f,
            ka_per_h: state.ka,
            ke_per_h: state.ke,
            volume_l: state.v,
            xr_lag_h: state.xr_lag_h,
            xr_f1: state.xr_f1,
            chiral_f_d: state.chiral_f_d,
            chiral_l_ke_per_h: state.chiral_l_ke,
        },
        therapeutic_window_mg_l: ReportWindow {
            mec_mg_l: state.mec,
            mtc_mg_l: state.mtc,
        },
        exposure_metrics: ReportMetrics {
            c_max_mg_l: summary.c_max,
            t_peak_h: summary.t_peak_h,
            c_min_mg_l: summary.c_min,
            auc_0_tau_mg_h_l: summary.auc_0_tau,
            auc_total_mg_h_l: summary.auc_total,
            accumulation_ratio_theoretical: summary.accumulation_ratio,
            accumulation_ratio_observed: summary.observed_accumulation_ratio,
            time_in_therapeutic_range_pct: summary.time_in_range_pct,
            time_in_therapeutic_range_h: summary.time_in_range_h,
            toxic_pct: summary.toxic_pct,
            toxic_hours: summary.toxic_hours,
            subtherapeutic_pct: summary.subtherapeutic_pct,
            subtherapeutic_hours: summary.subtherapeutic_hours,
        },
        schedule,
        notes: "AUC_0_tau and C_min are evaluated across the terminal tau_h dosing interval.",
    };
    serde_json::to_string_pretty(&payload).map_err(|e| format!("Report serialization error: {e}"))
}

/// Spawns a cross-platform `web_io` save dialog / browser download for the CSV dosing schedule.
pub fn spawn_csv_schedule_download(
    tx: Sender<Result<String, String>>,
    state: &RegimenState,
    curves: &ComputedCurves,
) {
    let csv = format_schedule_csv(state, curves);
    let config = crate::web_io::SaveAsDialogConfig {
        filter_name: "CSV Dosing Schedule (*.csv)",
        extensions: &["csv"],
        fallback_filename: "pk_dosing_schedule.csv",
        write_error_prefix: "Failed to save CSV dosing schedule",
    };
    crate::web_io::spawn_save_as_file_dialog(
        tx,
        config,
        "pk_dosing_schedule.csv",
        csv,
        |path, fallback| {
            format!(
                "Saved CSV schedule ({})",
                path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(fallback)
            )
        },
    );
}

/// Spawns a cross-platform `web_io` save dialog / browser download for the JSON exposure report.
pub fn spawn_json_report_download(
    tx: Sender<Result<String, String>>,
    state: &RegimenState,
    summary: &ExposureSummary,
) {
    match format_exposure_report_json(state, summary) {
        Ok(report_json) => {
            let config = crate::web_io::SaveAsDialogConfig {
                filter_name: "JSON Exposure Report (*.json)",
                extensions: &["json"],
                fallback_filename: "pk_exposure_report.json",
                write_error_prefix: "Failed to save JSON exposure report",
            };
            crate::web_io::spawn_save_as_file_dialog(
                tx,
                config,
                "pk_exposure_report.json",
                report_json,
                |path, fallback| {
                    format!(
                        "Saved JSON exposure report ({})",
                        path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(fallback)
                    )
                },
            );
        }
        Err(err) => {
            let _ = tx.send(Err(err));
        }
    }
}

/// Builds named `[time_h, conc_mg_l]` datasets for framework CSV/JSON export.
pub fn build_export_datasets(
    state: &RegimenState,
    curves: &ComputedCurves,
) -> Vec<(String, Vec<[f64; 2]>)> {
    let mut out = Vec::new();
    let total_pts: Vec<[f64; 2]> = curves
        .times
        .iter()
        .copied()
        .zip(curves.total.iter().copied())
        .map(|(t, c)| [t, c])
        .collect();
    out.push(("Total Plasma C(t) [mg/L]".to_string(), total_pts));

    if state.show_baseline && curves.baseline.len() == curves.times.len() {
        let base_pts: Vec<[f64; 2]> = curves
            .times
            .iter()
            .copied()
            .zip(curves.baseline.iter().copied())
            .map(|(t, c)| [t, c])
            .collect();
        out.push(("Uniform Maintenance Baseline [mg/L]".to_string(), base_pts));
    }

    if state.model == PkModelKind::ChiralEnantiomer && curves.d_isomer.len() == curves.times.len() {
        let d_pts: Vec<[f64; 2]> = curves
            .times
            .iter()
            .copied()
            .zip(curves.d_isomer.iter().copied())
            .map(|(t, c)| [t, c])
            .collect();
        let l_pts: Vec<[f64; 2]> = curves
            .times
            .iter()
            .copied()
            .zip(curves.l_isomer.iter().copied())
            .map(|(t, c)| [t, c])
            .collect();
        out.push(("d-Enantiomer [mg/L]".to_string(), d_pts));
        out.push(("l-Enantiomer [mg/L]".to_string(), l_pts));
    }
    out
}

/// Builds offscreen [`ExportPrimitive`]s including therapeutic window polygons and plasma curves.
pub fn build_export_primitives(
    state: &RegimenState,
    curves: &ComputedCurves,
    theme_mode: ThemeMode,
) -> Vec<ExportPrimitive> {
    let mut prims = Vec::new();
    let h = state.horizon_h;
    let palette = theme_mode.palette();
    let mec_color = palette.success;
    let mtc_color = palette.error;
    let win_fill = mec_color.gamma_multiply(0.18_f32);

    prims.push(ExportPrimitive::Polyline {
        name: "Therapeutic Window [MEC, MTC]".to_string(),
        points: vec![
            [0.0, state.mec],
            [h, state.mec],
            [h, state.mtc],
            [0.0, state.mtc],
        ],
        color: mec_color,
        stroke_width: 1.0_f32,
        closed: true,
        fill: Some(win_fill),
    });

    let y_top = curves
        .summary
        .map(|s| s.c_max.max(state.mtc * 1.15))
        .unwrap_or(state.mtc * 1.15);
    if y_top > state.mtc {
        prims.push(ExportPrimitive::Polyline {
            name: "Supratherapeutic Zone (> MTC)".to_string(),
            points: vec![
                [0.0, state.mtc],
                [h, state.mtc],
                [h, y_top],
                [0.0, y_top],
            ],
            color: mtc_color,
            stroke_width: 1.0_f32,
            closed: true,
            fill: Some(mtc_color.gamma_multiply(0.14_f32)),
        });
    }

    prims.push(ExportPrimitive::Polyline {
        name: "MEC Threshold".to_string(),
        points: vec![[0.0, state.mec], [h, state.mec]],
        color: mec_color,
        stroke_width: 1.5_f32,
        closed: false,
        fill: None,
    });
    prims.push(ExportPrimitive::Polyline {
        name: "MTC Threshold".to_string(),
        points: vec![[0.0, state.mtc], [h, state.mtc]],
        color: mtc_color,
        stroke_width: 1.5_f32,
        closed: false,
        fill: None,
    });

    append_curve_primitives(&mut prims, state, curves, &palette);
    prims
}

fn append_curve_primitives(
    prims: &mut Vec<ExportPrimitive>,
    state: &RegimenState,
    curves: &ComputedCurves,
    palette: &SemanticPalette,
) {
    let series = [
        palette.series_color(0),
        palette.muted_text,
        palette.series_color(2),
        palette.series_color(3),
    ];
    for (idx, (name, points)) in build_export_datasets(state, curves).into_iter().enumerate() {
        prims.push(ExportPrimitive::Polyline {
            name,
            points,
            color: series[idx % series.len()],
            stroke_width: if idx == 0 { 2.6_f32 } else { 1.8_f32 },
            closed: false,
            fill: None,
        });
    }
}
