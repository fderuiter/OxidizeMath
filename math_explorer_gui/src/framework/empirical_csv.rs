//! Inbound empirical CSV state, fit diagnostics, dialog handlers, and residual CSV export
//! utilities shared across nonlinear calibration tools in [`crate::framework::SimulationFramework`].

use crate::accessibility::{announce_live_assertive_status, announce_live_status};
use crate::web_io::{
    spawn_open_file_dialog, spawn_save_as_file_dialog, OpenDialogConfig, SaveAsDialogConfig,
};
use eframe::egui;
#[cfg(feature = "pure_math")]
use math_explorer::pure_math::analysis::optimization::CalibrationSummary;
use std::sync::mpsc::{channel, Receiver, TryRecvError};

#[path = "empirical_csv_parser.rs"]
mod parser;
#[path = "empirical_csv_ui.rs"]
mod ui;

pub use parser::{
    EmpiricalCsvTable, EmpiricalTripleSeries, MAX_EMPIRICAL_CSV_BYTES, MAX_EMPIRICAL_CSV_ROWS,
};
pub use ui::EmpiricalPanelOptions;

const EMPIRICAL_OPEN_CONFIG: OpenDialogConfig = OpenDialogConfig {
    filter_name: "Empirical CSV / TSV Dataset",
    extensions: &["csv", "tsv", "txt"],
    accept: ".csv,.tsv,.txt,text/csv,text/tab-separated-values,text/plain",
    read_error_prefix: "Failed to read CSV file",
};

const RESIDUAL_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "Calibration Residuals CSV",
    extensions: &["csv"],
    fallback_filename: "calibration_residuals.csv",
    write_error_prefix: "Failed to save residuals CSV",
};

/// Single row of an outbound calibration residual report (`x,y_observed,y_fitted,residual`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidualRow {
    /// Independent variable $x_i$ (e.g., time, cycles, or substrate concentration).
    pub x: f64,
    /// Empirical observation $y_{\text{observed}, i}$.
    pub y_observed: f64,
    /// Calibrated model prediction $y_{\text{fitted}, i}$.
    pub y_fitted: f64,
    /// Residual $r_i = y_{\text{observed}, i} - y_{\text{fitted}, i}$.
    pub residual: f64,
}

/// Formats residual rows into RFC 4180 CSV text with `x,y_observed,y_fitted,residual` header.
pub fn format_residuals_csv(rows: &[ResidualRow]) -> Result<String, String> {
    let mut out = String::with_capacity(32 + rows.len() * 44);
    out.push_str("x,y_observed,y_fitted,residual\n");
    for (idx, row) in rows.iter().enumerate() {
        if !row.x.is_finite()
            || !row.y_observed.is_finite()
            || !row.y_fitted.is_finite()
            || !row.residual.is_finite()
        {
            return Err(format!(
                "Non-finite residual value encountered at row {idx}"
            ));
        }
        out.push_str(&format!(
            "{:.4},{:.4},{:.4},{:.4}\n",
            row.x, row.y_observed, row.y_fitted, row.residual
        ));
    }
    Ok(out)
}

/// Named parameter estimate with standard error and two-sided 95% confidence interval half-width.
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterEstimate {
    /// Display name of the calibrated parameter.
    pub name: String,
    /// Fitted scalar value.
    pub value: f64,
    /// Asymptotic standard error.
    pub std_error: f64,
    /// Half-width of the 95% confidence interval.
    pub ci_95_half_width: f64,
}

/// Bookmarkable fit diagnostics and residual table produced by a calibration run.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationReport {
    /// Fitted parameter estimates with 95% confidence intervals.
    pub parameters: Vec<ParameterEstimate>,
    /// Optional domain-derived metrics (e.g., $R_0$ 95% CI or $N_{70}$ cycle life).
    pub derived_metrics: Vec<(String, String)>,
    /// Root-mean-square error.
    pub rmse: f64,
    /// Coefficient of determination $R^2$.
    pub r_squared: f64,
    /// Gaussian Akaike Information Criterion.
    pub aic: f64,
    /// Levenberg-Marquardt outer iterations executed.
    pub iterations: usize,
    /// Wall-clock calibration time in milliseconds.
    pub elapsed_ms: f64,
    /// Residual table rows.
    pub residuals: Vec<ResidualRow>,
}

impl CalibrationReport {
    /// Builds a validated finite [`CalibrationReport`] from [`CalibrationSummary`] and observations.
    #[cfg(feature = "pure_math")]
    pub fn from_summary(
        param_names: &[&str],
        summary: &CalibrationSummary,
        x_obs: &[f64],
        y_obs: &[f64],
        elapsed_ms: f64,
    ) -> Result<Self, String> {
        if !summary.rmse.is_finite() || !summary.r_squared.is_finite() || !summary.aic.is_finite() {
            return Err("Calibration produced non-finite goodness-of-fit statistics".to_string());
        }
        if x_obs.len() != y_obs.len() || y_obs.len() != summary.residuals.len() {
            return Err("Residual length mismatch after calibration".to_string());
        }
        let mut parameters = Vec::with_capacity(summary.parameters.len());
        for (idx, &val) in summary.parameters.iter().enumerate() {
            let se = summary.standard_errors.get(idx).copied().unwrap_or(0.0);
            let ci = summary.ci_95_half_widths.get(idx).copied().unwrap_or(0.0);
            if !val.is_finite() || !se.is_finite() || !ci.is_finite() {
                return Err("Calibration produced non-finite parameter estimates".to_string());
            }
            let name = param_names.get(idx).copied().unwrap_or("param").to_string();
            parameters.push(ParameterEstimate {
                name,
                value: val,
                std_error: se,
                ci_95_half_width: ci,
            });
        }
        let mut residuals = Vec::with_capacity(x_obs.len());
        for i in 0..x_obs.len() {
            let r = summary.residuals[i];
            let y_fit = y_obs[i] - r;
            if !x_obs[i].is_finite()
                || !y_obs[i].is_finite()
                || !y_fit.is_finite()
                || !r.is_finite()
            {
                return Err("Calibration produced non-finite fitted values".to_string());
            }
            residuals.push(ResidualRow {
                x: x_obs[i],
                y_observed: y_obs[i],
                y_fitted: y_fit,
                residual: r,
            });
        }
        Ok(Self {
            parameters,
            derived_metrics: Vec::new(),
            rmse: summary.rmse,
            r_squared: summary.r_squared,
            aic: summary.aic,
            iterations: summary.iterations,
            elapsed_ms: elapsed_ms.max(0.0),
            residuals,
        })
    }

    /// Formats this report's residual table as CSV text.
    pub fn residuals_csv(&self) -> Result<String, String> {
        format_residuals_csv(&self.residuals)
    }
}

/// Per-tool transient empirical CSV ingestion, column-selection, and dialog state.
#[derive(Default)]
pub struct EmpiricalCsvState {
    /// Currently loaded validated CSV table, if any.
    pub table: Option<EmpiricalCsvTable>,
    /// Selected zero-based column index for the independent variable $X$.
    pub x_col: usize,
    /// Selected zero-based column index for the observed response $Y$.
    pub y_col: usize,
    /// Optional third column index (e.g., per-row Depth of Discharge in battery cycling logs).
    pub aux_col: Option<usize>,
    /// Latest completed calibration diagnostics report.
    pub report: Option<CalibrationReport>,
    /// Inline status or error banner text.
    pub status_message: Option<String>,
    /// Whether `status_message` represents an error.
    pub is_error: bool,
    open_rx: Option<Receiver<Result<(String, String), String>>>,
    save_rx: Option<Receiver<Result<String, String>>>,
}

impl EmpiricalCsvState {
    /// Replaces the active table and resets column indices (`x_col = 0`, `y_col = 1`).
    pub fn set_table(&mut self, table: EmpiricalCsvTable) -> String {
        let summary = format!(
            "Loaded '{}' ({} rows, {} columns)",
            table.name,
            table.row_count(),
            table.column_count()
        );
        self.x_col = 0;
        self.y_col = usize::from(table.column_count() > 1);
        self.aux_col = None;
        self.report = None;
        self.table = Some(table);
        self.status_message = Some(summary.clone());
        self.is_error = false;
        summary
    }

    /// Clears the loaded CSV dataset and fit report while leaving model parameters intact.
    pub fn clear(&mut self) {
        self.table = None;
        self.report = None;
        self.aux_col = None;
        self.status_message = None;
        self.is_error = false;
    }

    /// Records a successful calibration report and announces polite status via `TwinSlotLiveRegion`.
    pub fn record_fit_success(
        &mut self,
        ctx: Option<&egui::Context>,
        report: CalibrationReport,
        announcement: impl Into<String>,
    ) {
        let msg = announcement.into();
        self.report = Some(report);
        self.status_message = Some(msg.clone());
        self.is_error = false;
        if let Some(c) = ctx {
            announce_live_status(c, msg);
        } else {
            crate::accessibility::announce_status(&msg);
        }
    }

    /// Records an ingestion or calibration error without mutating tool parameters,
    /// triggering an assertive `TwinSlotLiveRegion` screen-reader alert.
    pub fn record_error(&mut self, ctx: Option<&egui::Context>, error: impl Into<String>) {
        let msg = error.into();
        self.status_message = Some(msg.clone());
        self.is_error = true;
        if let Some(c) = ctx {
            announce_live_assertive_status(c, msg);
        } else {
            crate::accessibility::announce_status_with_priority(&msg, "assertive");
        }
    }

    /// Spawns a cross-platform file picker for `.csv`, `.tsv`, and `.txt` datasets.
    pub fn trigger_open_dialog(&mut self) {
        let (tx, rx) = channel();
        spawn_open_file_dialog(tx, EMPIRICAL_OPEN_CONFIG, |path| {
            path.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("dataset.csv")
                .to_string()
        });
        self.open_rx = Some(rx);
    }

    /// Spawns a cross-platform save dialog / browser download for `Download Residuals CSV`.
    pub fn trigger_save_residuals_dialog(
        &mut self,
        ctx: Option<&egui::Context>,
        default_filename: &str,
    ) -> Result<(), String> {
        let Some(ref report) = self.report else {
            return Err("No calibrated residuals available to export".to_string());
        };
        let csv_body = report.residuals_csv()?;
        let (tx, rx) = channel();
        spawn_save_as_file_dialog(
            tx,
            RESIDUAL_SAVE_CONFIG,
            default_filename,
            csv_body,
            |path, fallback| {
                path.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or(fallback)
                    .to_string()
            },
        );
        self.save_rx = Some(rx);
        if let Some(c) = ctx {
            announce_live_status(c, format!("Exporting residuals CSV {default_filename}"));
        }
        Ok(())
    }

    pub(crate) fn poll_dialogs(&mut self, ctx: &egui::Context) {
        if let Some(ref rx) = self.open_rx {
            match rx.try_recv() {
                Ok(Ok((name, text))) => {
                    self.open_rx = None;
                    match EmpiricalCsvTable::parse_str(name, &text) {
                        Ok(table) => {
                            let msg = self.set_table(table);
                            announce_live_status(ctx, msg);
                        }
                        Err(err) => self.record_error(Some(ctx), err),
                    }
                    ctx.request_repaint();
                }
                Ok(Err(err)) => {
                    self.open_rx = None;
                    self.record_error(Some(ctx), err);
                    ctx.request_repaint();
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    self.open_rx = None;
                }
            }
        }
        if let Some(ref rx) = self.save_rx {
            match rx.try_recv() {
                Ok(Ok(saved_name)) => {
                    self.save_rx = None;
                    let msg = format!("Saved residuals CSV '{saved_name}'");
                    self.status_message = Some(msg.clone());
                    self.is_error = false;
                    announce_live_status(ctx, msg);
                    ctx.request_repaint();
                }
                Ok(Err(err)) => {
                    self.save_rx = None;
                    self.record_error(Some(ctx), err);
                    ctx.request_repaint();
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    self.save_rx = None;
                }
            }
        }
    }
}

/// Processes `egui::RawInput::dropped_files` for tools supporting empirical CSV calibration.
pub fn handle_dropped_empirical_files(
    ctx: &egui::Context,
    tool: &mut dyn crate::framework::InteractiveTool,
) {
    if !tool.supports_empirical_csv() {
        return;
    }
    let dropped = ctx.input(|i| i.raw.dropped_files.clone());
    if dropped.is_empty() {
        return;
    }
    for file in dropped {
        let label = if !file.name.trim().is_empty() {
            file.name.clone()
        } else {
            file.path
                .as_ref()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .unwrap_or("dropped.csv")
                .to_string()
        };
        let parsed = read_dropped_file_bounded(&file)
            .and_then(|bytes| EmpiricalCsvTable::parse_bytes(label, &bytes));
        match parsed {
            Ok(table) => match tool.ingest_empirical_dataset(table) {
                Ok(msg) => announce_live_status(ctx, msg),
                Err(err) => announce_live_assertive_status(ctx, err),
            },
            Err(err) => {
                tool.notify_empirical_csv_error(err.clone());
                announce_live_assertive_status(ctx, err);
            }
        }
        ctx.request_repaint();
    }
}

fn read_dropped_file_bounded(file: &egui::DroppedFile) -> Result<Vec<u8>, String> {
    if let Some(ref bytes) = file.bytes {
        if bytes.len() > MAX_EMPIRICAL_CSV_BYTES {
            return Err(format!(
                "Dropped CSV ({} bytes) exceeds 2 MiB safety limit",
                bytes.len()
            ));
        }
        return Ok(bytes.to_vec());
    }
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(ref path) = file.path {
        use std::io::Read;
        let handle =
            std::fs::File::open(path).map_err(|e| format!("Failed to open dropped CSV: {e}"))?;
        let mut buf = Vec::new();
        handle
            .take((MAX_EMPIRICAL_CSV_BYTES + 1) as u64)
            .read_to_end(&mut buf)
            .map_err(|e| format!("Failed to read dropped CSV: {e}"))?;
        if buf.len() > MAX_EMPIRICAL_CSV_BYTES {
            return Err("Dropped CSV exceeds 2 MiB safety limit".to_string());
        }
        return Ok(buf);
    }
    Err("Dropped file contained no readable bytes".to_string())
}

#[cfg(test)]
#[path = "empirical_csv_tests.rs"]
mod tests;
