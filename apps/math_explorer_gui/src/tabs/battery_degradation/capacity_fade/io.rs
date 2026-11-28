//! Bounded lab `.csv` cycling-log parser and Levenberg-Marquardt calibration driver
//! for [`super::CapacityFadeTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:battery_degradation]

use super::CapacityFadeTool;
use crate::web_io::strip_utf8_bom;
use math_explorer::engineering::battery_degradation::DepthOfDischarge;

/// Maximum finite `(cycles, capacity)` rows imported from a lab `.csv` file.
pub const MAX_CALIBRATION_CSV_ROWS: usize = 2_000;

/// Maximum raw UTF-8 byte size (`8 MiB`) accepted for client-side cycling `.csv` files.
pub const MAX_CALIBRATION_CSV_BYTES: usize = 8 * 1_048_576;

/// Summary of the most recent Levenberg-Marquardt CSV calibration run.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibratedLabSummary {
    /// Source file label.
    pub source_name: String,
    /// Fitted $N_{70}$ cycles at calibration test conditions.
    pub fitted_n70_cycles: f64,
    /// Fitted stretched exponent $z \in [0.1, 2.0]$.
    pub fitted_exponent_z: f64,
    /// Root-mean-square state-of-health residual.
    pub rmse: f64,
    /// Coefficient of determination $R^2$.
    pub r_squared: f64,
    /// Count of finite `(cycles, capacity)` observations used.
    pub sample_count: usize,
}

fn parse_csv_pair(line: &str) -> Result<Option<[f64; 2]>, String> {
    let mut cols = line.split([',', ';', '\t']).map(str::trim);
    let Some(c0) = cols.next() else {
        return Ok(None);
    };
    let Some(c1) = cols.next() else {
        return Err(format!(
            "Expected two columns (cycles, capacity) in '{line}'"
        ));
    };
    let x = c0
        .parse::<f64>()
        .map_err(|_| format!("Invalid cycle number '{c0}'"))?;
    let y = c1
        .trim_end_matches('%')
        .trim()
        .parse::<f64>()
        .map_err(|_| format!("Invalid capacity value '{c1}'"))?;
    if !x.is_finite() || !y.is_finite() {
        return Ok(None);
    }
    if x < 0.0 {
        return Err("Cycle counts in CSV must be non-negative".into());
    }
    Ok(Some([x, y]))
}

fn looks_like_header(line: &str) -> bool {
    line.chars()
        .any(|ch| ch.is_ascii_alphabetic() && ch != 'e' && ch != 'E')
}

/// Parses up to [`MAX_CALIBRATION_CSV_ROWS`] finite `(cycles, soh_fraction)` rows after stripping any UTF-8 BOM.
pub fn parse_calibration_csv(raw: &str) -> Result<Vec<[f64; 2]>, String> {
    if raw.len() > MAX_CALIBRATION_CSV_BYTES {
        return Err("Cycling CSV exceeds 8 MiB client-side safety limit".into());
    }
    let clean = strip_utf8_bom(raw);
    let mut rows = Vec::new();
    let mut checked_header = false;

    for raw_line in clean.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if !checked_header {
            checked_header = true;
            if looks_like_header(line) {
                continue;
            }
        }
        if let Some(pair) = parse_csv_pair(line)? {
            rows.push(pair);
            if rows.len() >= MAX_CALIBRATION_CSV_ROWS {
                break;
            }
        }
    }

    normalize_and_validate_soh_rows(rows)
}

fn normalize_and_validate_soh_rows(mut rows: Vec<[f64; 2]>) -> Result<Vec<[f64; 2]>, String> {
    if rows.len() < 3 {
        return Err(format!(
            "Calibration requires at least 3 finite (cycles, SOH) rows; found {}",
            rows.len()
        ));
    }
    let all_percent_scale = rows.iter().all(|&[_, y]| (1.2001..=120.0).contains(&y));
    if all_percent_scale {
        for row in &mut rows {
            row[1] *= 0.01;
        }
    }
    for &[_, y] in &rows {
        if !(0.0..=1.2).contains(&y) {
            return Err(format!(
                "SOH value {y:.4} lies outside valid fraction [0.0, 1.2] (or [0, 120]%)"
            ));
        }
    }
    Ok(rows)
}

impl CapacityFadeTool {
    /// Parses `csv_text` and runs Levenberg-Marquardt fitting; retains all prior parameters on failure.
    pub fn import_calibration_csv(
        &mut self,
        source_name: &str,
        csv_text: &str,
    ) -> Result<CalibratedLabSummary, String> {
        let samples = parse_calibration_csv(csv_text)?;
        if self.dod <= 0.0 {
            return Err("Set Depth of Discharge > 0% before calibrating cycling logs".into());
        }
        let dod = DepthOfDischarge::new_clamped(self.dod);
        let fit = self
            .active_model()
            .calibrate_from_cycles(&samples, dod)
            .map_err(|e| format!("Levenberg-Marquardt calibration failed: {e}"))?;

        self.calibrated_alpha = Some(fit.model.alpha());
        self.exponent_z = fit.fitted_exponent_z;
        let max_csv_cyc = samples.iter().fold(0.0_f64, |acc, &[c, _]| acc.max(c));
        self.cycles_to_simulate = self
            .cycles_to_simulate
            .max((max_csv_cyc * 1.15).clamp(100.0, 10_000.0));
        self.lab_samples = samples;

        let summary = CalibratedLabSummary {
            source_name: source_name.to_string(),
            fitted_n70_cycles: fit.fitted_n70_cycles,
            fitted_exponent_z: fit.fitted_exponent_z,
            rmse: fit.rmse,
            r_squared: fit.r_squared,
            sample_count: fit.sample_count,
        };
        self.last_fit = Some(summary.clone());
        Ok(summary)
    }

    pub(crate) fn poll_dialog_channels(&mut self) {
        if let Some(open_res) = self.dialogs.poll_open() {
            match open_res {
                Ok((name, text)) => match self.import_calibration_csv(&name, &text) {
                    Ok(summary) => {
                        self.io_status = Some((
                            false,
                            format!(
                                "Fitted {} rows from {} (RMSE {:.4}, z={:.3})",
                                summary.sample_count, name, summary.rmse, summary.fitted_exponent_z
                            ),
                        ));
                    }
                    Err(err) => self.io_status = Some((true, err)),
                },
                Err(err) => self.io_status = Some((true, err)),
            }
        }
        if let Some(save_res) = self.dialogs.poll_save() {
            match save_res {
                Ok(msg) => self.io_status = Some((false, msg)),
                Err(err) => self.io_status = Some((true, err)),
            }
        }
    }
}
