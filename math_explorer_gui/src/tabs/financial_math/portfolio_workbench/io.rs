#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::engine::{calibrate_ou_from_returns, ImportedDatasetSummary};
use super::state::{AssetConfig, MAX_WORKBENCH_ASSETS};
use super::PortfolioCopulaWorkbenchTool;
use crate::accessibility::{announce_live_assertive_status, announce_live_status};
use crate::web_io::{spawn_open_file_dialog, strip_utf8_bom, OpenDialogConfig};
use eframe::egui;
use math_explorer::nalgebra::DMatrix;
use math_explorer::pure_math::statistics::copula::CorrelationMatrix;
use std::sync::mpsc::channel;

pub const MAX_IMPORT_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_IMPORT_ROWS: usize = 10_000;
const ANNUALIZATION_PERIODS: f64 = 252.0;

const OPEN_RETURNS_CONFIG: OpenDialogConfig = OpenDialogConfig {
    filter_name: "Return Series (*.csv, *.json)",
    extensions: &["csv", "json"],
    accept: ".csv,.json,text/csv,application/json",
    read_error_prefix: "Failed to read return series file",
};

#[derive(Debug, Clone)]
pub struct CalibratedReturnsData {
    pub assets: Vec<AssetConfig>,
    pub corr_upper: Vec<f64>,
    pub mean_rho: f64,
    pub summary: ImportedDatasetSummary,
}

#[derive(serde::Deserialize)]
struct JsonReturnEnvelope {
    #[serde(default)]
    columns: Vec<String>,
    returns: Vec<Vec<f64>>,
}

pub fn parse_return_series_payload(
    filename: &str,
    raw: &str,
    dt: f64,
) -> Result<CalibratedReturnsData, String> {
    if raw.len() > MAX_IMPORT_BYTES {
        return Err(format!(
            "Return series file exceeds 2 MiB limit ({} bytes > {MAX_IMPORT_BYTES} bytes)",
            raw.len()
        ));
    }
    let clean = strip_utf8_bom(raw).trim();
    if clean.is_empty() {
        return Err("Return series file is empty".into());
    }
    let is_json = filename.to_ascii_lowercase().ends_with(".json")
        || clean.starts_with('{')
        || clean.starts_with('[');
    let (headers, rows) = if is_json {
        parse_json_matrix(clean)?
    } else {
        parse_csv_matrix(clean)?
    };
    calibrate_from_matrix(filename, headers, &rows, dt)
}

fn parse_json_matrix(clean: &str) -> Result<(Vec<String>, Vec<Vec<f64>>), String> {
    if clean.starts_with('{') {
        let env: JsonReturnEnvelope =
            serde_json::from_str(clean).map_err(|e| format!("Invalid JSON return object: {e}"))?;
        validate_matrix_dims(env.columns, env.returns)
    } else {
        let rows: Vec<Vec<f64>> =
            serde_json::from_str(clean).map_err(|e| format!("Invalid JSON return array: {e}"))?;
        validate_matrix_dims(Vec::new(), rows)
    }
}

fn parse_csv_matrix(clean: &str) -> Result<(Vec<String>, Vec<Vec<f64>>), String> {
    let mut lines = clean
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'));
    let first_line = lines
        .next()
        .ok_or_else(|| String::from("CSV contains no rows"))?;
    let first_cells: Vec<&str> = first_line.split(',').map(str::trim).collect();
    let first_all_num = first_cells.iter().all(|c| c.parse::<f64>().is_ok());

    let mut headers = Vec::new();
    let mut rows = Vec::new();
    let mut skip_first_col = false;

    if first_all_num {
        rows.push(parse_numeric_cells(&first_cells)?);
    } else {
        headers = first_cells.iter().map(|s| (*s).to_string()).collect();
    }

    for (line_idx, line) in lines.enumerate() {
        if rows.len() >= MAX_IMPORT_ROWS {
            return Err(format!("Return series exceeds {MAX_IMPORT_ROWS} row limit"));
        }
        let cells: Vec<&str> = line.split(',').map(str::trim).collect();
        if rows.is_empty() && cells.len() >= 2 && cells[0].parse::<f64>().is_err() {
            skip_first_col = true;
            if headers.len() == cells.len() {
                headers.remove(0);
            }
        }
        let slice = if skip_first_col {
            &cells[1..]
        } else {
            &cells[..]
        };
        let parsed =
            parse_numeric_cells(slice).map_err(|e| format!("Row {}: {e}", line_idx + 2))?;
        rows.push(parsed);
    }
    validate_matrix_dims(headers, rows)
}

fn parse_numeric_cells(cells: &[&str]) -> Result<Vec<f64>, String> {
    if cells.is_empty() {
        return Err("Empty row".into());
    }
    let mut vals = Vec::with_capacity(cells.len());
    for cell in cells {
        let v: f64 = cell
            .parse()
            .map_err(|_| format!("Non-numeric return value '{cell}'"))?;
        if !v.is_finite() {
            return Err("Return series contains non-finite values".into());
        }
        vals.push(v);
    }
    Ok(vals)
}

fn validate_matrix_dims(
    mut headers: Vec<String>,
    rows: Vec<Vec<f64>>,
) -> Result<(Vec<String>, Vec<Vec<f64>>), String> {
    if rows.len() < 3 {
        return Err(format!(
            "At least 3 return observations are required (got {})",
            rows.len()
        ));
    }
    if rows.len() > MAX_IMPORT_ROWS {
        return Err(format!(
            "Return series exceeds {MAX_IMPORT_ROWS} row limit (got {})",
            rows.len()
        ));
    }
    let cols = rows[0].len();
    if !(1..=MAX_WORKBENCH_ASSETS).contains(&cols) {
        return Err(format!(
            "Return series must have 1..={MAX_WORKBENCH_ASSETS} asset columns (got {cols})"
        ));
    }
    if rows
        .iter()
        .any(|r| r.len() != cols || r.iter().any(|v| !v.is_finite()))
    {
        return Err("All return rows must have consistent length and finite numbers".into());
    }
    if headers.len() != cols {
        headers = (0..cols).map(|i| format!("Asset {}", i + 1)).collect();
    }
    Ok((headers, rows))
}

fn sample_means_and_covariance(rows: &[Vec<f64>], cols: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let t_len = rows.len() as f64;
    let mut means = vec![0.0; cols];
    for r in rows {
        for (c, &v) in r.iter().enumerate() {
            means[c] += v / t_len;
        }
    }
    let mut cov = vec![vec![0.0; cols]; cols];
    for r in rows {
        for i in 0..cols {
            for j in i..cols {
                cov[i][j] += (r[i] - means[i]) * (r[j] - means[j]) / (t_len - 1.0);
            }
        }
    }
    (means, cov)
}

fn calibrate_from_matrix(
    filename: &str,
    headers: Vec<String>,
    rows: &[Vec<f64>],
    dt: f64,
) -> Result<CalibratedReturnsData, String> {
    let t_len = rows.len() as f64;
    let cols = headers.len();
    let (means, cov) = sample_means_and_covariance(rows, cols);

    let mut assets = Vec::with_capacity(cols);
    for c in 0..cols {
        let pos_rate = rows.iter().filter(|r| r[c] > 0.0).count() as f64 / t_len;
        let ann_mu = (means[c] * ANNUALIZATION_PERIODS).clamp(-0.90, 2.40);
        let ann_sig = (cov[c][c].max(1e-8).sqrt() * ANNUALIZATION_PERIODS.sqrt()).clamp(0.02, 2.40);
        assets.push(AssetConfig {
            label: headers[c].chars().take(28).collect(),
            mu: ann_mu,
            sigma: ann_sig,
            marginal_u: pos_rate.clamp(0.05, 0.95),
        });
    }
    let (corr_upper, mean_rho) = build_spd_sample_correlation(&cov, cols);
    let eq_losses: Vec<f64> = rows
        .iter()
        .map(|r| -(r.iter().sum::<f64>() / (cols as f64)))
        .collect();
    let primary_series: Vec<f64> = rows.iter().map(|r| r[0]).collect();
    let ou_calibrated = calibrate_ou_from_returns(&primary_series, dt);

    Ok(CalibratedReturnsData {
        assets,
        corr_upper,
        mean_rho,
        summary: ImportedDatasetSummary {
            source_name: filename.to_string(),
            row_count: rows.len(),
            column_names: headers,
            equal_weight_losses: eq_losses,
            ou_calibrated,
        },
    })
}

fn build_spd_sample_correlation(cov: &[Vec<f64>], cols: usize) -> (Vec<f64>, f64) {
    if cols <= 1 {
        return (Vec::new(), 0.0);
    }
    let mut raw_mat = DMatrix::identity(cols, cols);
    let mut sum_rho = 0.0;
    let mut pairs = 0_usize;
    for i in 0..cols {
        for j in (i + 1)..cols {
            let denom = (cov[i][i].max(1e-12) * cov[j][j].max(1e-12)).sqrt();
            let r_ij = (cov[i][j] / denom).clamp(-0.98, 0.98);
            raw_mat[(i, j)] = r_ij;
            raw_mat[(j, i)] = r_ij;
            sum_rho += r_ij;
            pairs += 1;
        }
    }
    let mut valid_mat = raw_mat.clone();
    for shrink in [0.0, 0.01, 0.05, 0.15, 0.35] {
        let mut candidate = &raw_mat * (1.0 - shrink);
        for d in 0..cols {
            candidate[(d, d)] = 1.0;
        }
        if CorrelationMatrix::new(candidate.clone()).is_ok() {
            valid_mat = candidate;
            break;
        }
    }
    let mut upper = Vec::with_capacity(pairs);
    for i in 0..cols {
        for j in (i + 1)..cols {
            upper.push(valid_mat[(i, j)]);
        }
    }
    let mean_rho = (sum_rho / (pairs as f64)).clamp(-0.90, 0.95);
    (upper, mean_rho)
}

impl PortfolioCopulaWorkbenchTool {
    pub fn import_return_series_str(&mut self, filename: &str, raw: &str) -> Result<(), String> {
        let cal = parse_return_series_payload(filename, raw, self.state.ou_dt)?;
        self.state.assets = cal.assets;
        self.state.corr_upper = cal.corr_upper;
        self.state.rho = cal.mean_rho.clamp(
            super::state::PortfolioWorkbenchState::min_equicorrelation_for(
                self.state.assets.len().max(self.state.copula_legs),
            ),
            0.95,
        );
        if self.state.assets.len() >= 2 {
            self.state.copula_legs = self.state.assets.len();
        }
        if let Some((mu, theta, sigma)) = cal.summary.ou_calibrated {
            self.state.ou_mu = mu.clamp(-5.0, 5.0);
            self.state.ou_theta = theta.clamp(0.05, 20.0);
            self.state.ou_sigma = sigma.clamp(0.01, 5.0);
        }
        self.imported = Some(cal.summary);
        self.recalculate_all();
        Ok(())
    }

    pub(super) fn trigger_open_returns_dialog(&mut self) {
        let (tx, rx) = channel();
        self.open_rx = Some(rx);
        spawn_open_file_dialog(tx, OPEN_RETURNS_CONFIG, |path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("returns.csv")
                .to_string()
        });
    }

    pub(super) fn poll_io_channels(&mut self, ctx: &egui::Context) {
        if let Some(res) = self.open_rx.as_ref().and_then(|rx| rx.try_recv().ok()) {
            self.open_rx = None;
            self.handle_open_result(ctx, res);
        }
        if let Some(res) = self.save_rx.as_ref().and_then(|rx| rx.try_recv().ok()) {
            self.save_rx = None;
            match res {
                Ok(msg) => {
                    announce_live_status(ctx, &msg);
                    self.io_status = Some((msg, false));
                }
                Err(err) => {
                    announce_live_assertive_status(ctx, &err);
                    self.io_status = Some((err, true));
                }
            }
        }
    }

    fn handle_open_result(&mut self, ctx: &egui::Context, res: Result<(String, String), String>) {
        match res {
            Ok((name, content)) => match self.import_return_series_str(&name, &content) {
                Ok(()) => {
                    let rows = self.imported.as_ref().map_or(0, |s| s.row_count);
                    let msg = format!(
                        "Imported {rows} rows ({} assets) from {name}",
                        self.state.assets.len()
                    );
                    announce_live_status(ctx, &msg);
                    self.io_status = Some((msg, false));
                }
                Err(err) => {
                    announce_live_assertive_status(ctx, &err);
                    self.io_status = Some((err, true));
                }
            },
            Err(err) => {
                announce_live_assertive_status(ctx, &err);
                self.io_status = Some((err, true));
            }
        }
    }
}

// [cite:clinical_trials_statistics]
