//! Bounded client-side `.csv`, `.tsv`, and `.json` point-cloud import and persistence interval export.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{ToolStateBounds, MAX_JSON_NESTING_DEPTH};
use crate::web_io::{
    spawn_open_file_dialog, spawn_save_as_file_dialog, strip_utf8_bom, OpenDialogConfig,
    SaveAsDialogConfig,
};
use math_explorer::pure_math::statistics::tda::core::{NdPointCloud, Point2D};
use math_explorer::pure_math::statistics::tda::{
    RipsInterval, RipsPersistenceResult, MAX_TDA_DIMENSION, MAX_TDA_POINTS,
};
use std::sync::mpsc::{channel, Receiver};

/// Maximum UTF-8 file size (`512 KiB`) accepted when importing local point clouds.
pub const MAX_IMPORT_BYTES: usize = 524_288;

/// Maximum point count (`250`) supported by the Vietoris-Rips TDA workbench.
pub const MAX_VIETORIS_RIPS_POINTS: usize = MAX_TDA_POINTS;

/// Maximum custom 2D point count (`64`) quantized to 4 decimals so URL share payloads stay under 4 KiB.
pub const MAX_URL_CUSTOM_POINTS_2D: usize = 64;

/// Validated point cloud parsed from a local `.csv`, `.tsv`, or `.json` file.
#[derive(Clone, Debug)]
pub enum ImportedCloud {
    /// Planar 2D point cloud (`d = 2`).
    Points2D(Vec<Point2D>),
    /// Higher-dimensional Euclidean point cloud (`3 <= d <= 32`).
    PointsNd(NdPointCloud),
}

/// Serializable persistence interval table row for `.csv` and `.json` downloads.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IntervalExportRow {
    /// Homology dimension (`0` for $H_0$, `1` for $H_1$).
    pub dimension: usize,
    /// Exact birth scale.
    pub birth: f64,
    /// Exact death scale, or finite `max_radius` when `is_essential` is `true`.
    pub death: f64,
    /// Persistence lifetime `death - birth`.
    pub persistence: f64,
    /// Whether this homology class survives past the maximum filtration scale.
    pub is_essential: bool,
}

impl From<&RipsInterval> for IntervalExportRow {
    fn from(iv: &RipsInterval) -> Self {
        Self {
            dimension: iv.dimension,
            birth: iv.birth,
            death: iv.death,
            persistence: iv.persistence(),
            is_essential: iv.is_essential,
        }
    }
}

#[derive(serde::Serialize)]
struct IntervalJsonEnvelope {
    max_radius: f64,
    truncated_triangles: bool,
    intervals: Vec<IntervalExportRow>,
}

/// Quantizes a coordinate to 4 decimal places (`1e-4` grid) for compact URL encoding.
#[must_use]
pub fn quantize_url_coord(value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }
    let rounded = (value * 10_000.0).round() / 10_000.0;
    if rounded == 0.0 {
        0.0
    } else {
        rounded
    }
}

/// Parses a `.csv`, `.tsv`, or `.json` point-cloud payload strictly within safety bounds.
pub fn parse_point_cloud_text(filename: &str, raw: &str) -> Result<ImportedCloud, String> {
    if raw.len() > MAX_IMPORT_BYTES {
        return Err(format!(
            "Point-cloud file size ({} bytes) exceeds maximum limit ({MAX_IMPORT_BYTES} bytes)",
            raw.len()
        ));
    }
    let cleaned = strip_utf8_bom(raw);
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        return Err("Point-cloud file is empty".to_string());
    }

    let lower = filename.trim().to_ascii_lowercase();
    if lower.ends_with(".json")
        || (!lower.ends_with(".csv") && !lower.ends_with(".tsv") && trimmed.starts_with(['[', '{']))
    {
        parse_json_point_cloud(trimmed)
    } else {
        let delimiter = resolve_delimiter(&lower, trimmed);
        parse_delimited_point_cloud(trimmed, delimiter)
    }
}

fn resolve_delimiter(lower_name: &str, content: &str) -> char {
    if lower_name.ends_with(".tsv") {
        return '\t';
    }
    let first_line = content
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('#'))
        .unwrap_or("");
    if first_line.contains('\t') && !first_line.contains(',') {
        '\t'
    } else {
        ','
    }
}

fn parse_delimited_point_cloud(content: &str, delimiter: char) -> Result<ImportedCloud, String> {
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut expected_dim: Option<usize> = None;

    for (line_idx, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cells: Vec<&str> = line.split(delimiter).map(str::trim).collect();
        if expected_dim.is_none() {
            let dim = validate_dimension_count(cells.len(), line_idx + 1)?;
            expected_dim = Some(dim);
            if is_header_row(&cells)? {
                continue;
            }
        }
        let dim = expected_dim.unwrap_or(2);
        if cells.len() != dim {
            return Err(format!(
                "Jagged row at line {}: expected {dim} columns, got {}",
                line_idx + 1,
                cells.len()
            ));
        }
        if rows.len() >= MAX_VIETORIS_RIPS_POINTS {
            return Err(format!(
                "Point count exceeds maximum limit of {MAX_VIETORIS_RIPS_POINTS} points"
            ));
        }
        rows.push(parse_numeric_cells(&cells, line_idx + 1)?);
    }

    finalize_imported_rows(rows)
}

fn validate_dimension_count(col_count: usize, line_no: usize) -> Result<usize, String> {
    if !(2..=MAX_TDA_DIMENSION).contains(&col_count) {
        return Err(format!(
            "Line {line_no} has {col_count} columns; expected 2..={MAX_TDA_DIMENSION} coordinates"
        ));
    }
    Ok(col_count)
}

fn is_header_row(cells: &[&str]) -> Result<bool, String> {
    if cells.iter().any(|c| c.is_empty()) {
        return Err("Malformed header or first row: contains empty column cell".to_string());
    }
    let numeric_flags: Vec<bool> = cells
        .iter()
        .map(|c| c.parse::<f64>().is_ok_and(f64::is_finite))
        .collect();
    if numeric_flags.iter().all(|&is_num| is_num) {
        Ok(false)
    } else if numeric_flags.iter().all(|&is_num| !is_num) {
        Ok(true)
    } else {
        Err("Malformed CSV/TSV header at line 1: mixes numeric and non-numeric cells".to_string())
    }
}

fn parse_numeric_cells(cells: &[&str], line_no: usize) -> Result<Vec<f64>, String> {
    let mut coords = Vec::with_capacity(cells.len());
    for (col_idx, cell) in cells.iter().enumerate() {
        let val = cell.parse::<f64>().map_err(|_| {
            format!(
                "Invalid coordinate '{}' at line {line_no}, column {}",
                cell,
                col_idx + 1
            )
        })?;
        if !val.is_finite() {
            return Err(format!(
                "Non-finite coordinate at line {line_no}, column {}",
                col_idx + 1
            ));
        }
        coords.push(val);
    }
    Ok(coords)
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum RawJsonCloud {
    Array(Vec<Vec<f64>>),
    Object { points: Vec<Vec<f64>> },
}

fn parse_json_point_cloud(trimmed: &str) -> Result<ImportedCloud, String> {
    let json_bounds = ToolStateBounds::new(
        MAX_IMPORT_BYTES,
        MAX_JSON_NESTING_DEPTH,
        MAX_VIETORIS_RIPS_POINTS + 1,
    );
    json_bounds.validate_json(trimmed)?;
    let parsed: RawJsonCloud =
        serde_json::from_str(trimmed).map_err(|e| format!("Invalid point-cloud JSON: {e}"))?;
    let rows = match parsed {
        RawJsonCloud::Array(r) | RawJsonCloud::Object { points: r } => r,
    };
    finalize_imported_rows(rows)
}

fn finalize_imported_rows(rows: Vec<Vec<f64>>) -> Result<ImportedCloud, String> {
    if rows.is_empty() {
        return Err("Point-cloud file contains no data rows".to_string());
    }
    if rows.len() > MAX_VIETORIS_RIPS_POINTS {
        return Err(format!(
            "Point count ({}) exceeds maximum limit ({MAX_VIETORIS_RIPS_POINTS})",
            rows.len()
        ));
    }
    let dim = validate_dimension_count(rows[0].len(), 1)?;
    for (idx, row) in rows.iter().enumerate() {
        if row.len() != dim {
            return Err(format!(
                "Jagged point row {}: expected {dim} coordinates, got {}",
                idx + 1,
                row.len()
            ));
        }
        if !row.iter().all(|v| v.is_finite()) {
            return Err(format!("Row {} contains non-finite coordinates", idx + 1));
        }
    }

    if dim == 2 {
        let mut pts = Vec::with_capacity(rows.len());
        let quantize = rows.len() <= MAX_URL_CUSTOM_POINTS_2D;
        for row in rows {
            let x = if quantize {
                quantize_url_coord(row[0])
            } else {
                row[0]
            };
            let y = if quantize {
                quantize_url_coord(row[1])
            } else {
                row[1]
            };
            pts.push(Point2D::new(x, y));
        }
        Ok(ImportedCloud::Points2D(pts))
    } else {
        NdPointCloud::new(rows)
            .map(ImportedCloud::PointsNd)
            .map_err(|e| format!("Invalid N-D point cloud: {e}"))
    }
}

/// Formats persistence intervals as a `.csv` table including finite `max_radius` and `is_essential`.
#[must_use]
pub fn format_intervals_csv(result: &RipsPersistenceResult) -> String {
    use std::fmt::Write;
    let mut csv = String::from("dimension,birth,death,persistence,is_essential\n");
    for iv in &result.intervals {
        let row = IntervalExportRow::from(iv);
        let _ = writeln!(
            &mut csv,
            "{},{:.6},{:.6},{:.6},{}",
            row.dimension, row.birth, row.death, row.persistence, row.is_essential
        );
    }
    csv
}

/// Formats persistence intervals as a structured `.json` document.
pub fn format_intervals_json(result: &RipsPersistenceResult) -> Result<String, String> {
    let envelope = IntervalJsonEnvelope {
        max_radius: result.max_radius,
        truncated_triangles: result.truncated_triangles,
        intervals: result
            .intervals
            .iter()
            .map(IntervalExportRow::from)
            .collect(),
    };
    serde_json::to_string_pretty(&envelope)
        .map_err(|e| format!("Failed to serialize persistence intervals: {e}"))
}

/// Asynchronous channel state for non-blocking client-side open/save file dialogs.
#[derive(Default)]
pub struct PendingIoTasks {
    open_rx: Option<Receiver<Result<(String, String), String>>>,
    save_rx: Option<Receiver<Result<String, String>>>,
}

impl PendingIoTasks {
    /// Launches a local `.csv`, `.tsv`, or `.json` file picker via [`spawn_open_file_dialog`].
    pub fn trigger_point_cloud_open(&mut self) {
        let (tx, rx) = channel();
        let cfg = OpenDialogConfig {
            filter_name: "Point Cloud (*.csv, *.tsv, *.json)",
            extensions: &["csv", "tsv", "json"],
            accept: ".csv,.tsv,.json,text/csv,text/tab-separated-values,application/json",
            read_error_prefix: "Failed to read point-cloud file",
        };
        spawn_open_file_dialog(tx, cfg, |path| {
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "cloud.csv".to_string())
        });
        self.open_rx = Some(rx);
    }

    /// Triggers a `.csv` download/save dialog for `result` via [`spawn_save_as_file_dialog`].
    pub fn trigger_intervals_csv_save(&mut self, result: &RipsPersistenceResult) {
        let (tx, rx) = channel();
        let cfg = SaveAsDialogConfig {
            filter_name: "Persistence Intervals CSV (*.csv)",
            extensions: &["csv"],
            fallback_filename: "vietoris_rips_intervals.csv",
            write_error_prefix: "Failed to save intervals CSV",
        };
        let payload = format_intervals_csv(result);
        spawn_save_as_file_dialog(
            tx,
            cfg,
            "vietoris_rips_intervals.csv",
            payload,
            |_, name| format!("Saved persistence intervals to {name}"),
        );
        self.save_rx = Some(rx);
    }

    /// Triggers a `.json` download/save dialog for `result` via [`spawn_save_as_file_dialog`].
    pub fn trigger_intervals_json_save(
        &mut self,
        result: &RipsPersistenceResult,
    ) -> Result<(), String> {
        let payload = format_intervals_json(result)?;
        let (tx, rx) = channel();
        let cfg = SaveAsDialogConfig {
            filter_name: "Persistence Intervals JSON (*.json)",
            extensions: &["json"],
            fallback_filename: "vietoris_rips_intervals.json",
            write_error_prefix: "Failed to save intervals JSON",
        };
        spawn_save_as_file_dialog(
            tx,
            cfg,
            "vietoris_rips_intervals.json",
            payload,
            |_, name| format!("Saved persistence intervals to {name}"),
        );
        self.save_rx = Some(rx);
        Ok(())
    }

    /// Polls any completed file-open outcome without blocking the UI thread.
    pub fn poll_open(&mut self) -> Option<Result<(String, String), String>> {
        let rx = self.open_rx.as_ref()?;
        match rx.try_recv() {
            Ok(outcome) => {
                self.open_rx = None;
                Some(outcome)
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.open_rx = None;
                None
            }
        }
    }

    /// Polls any completed interval-save outcome without blocking the UI thread.
    pub fn poll_save(&mut self) -> Option<Result<String, String>> {
        let rx = self.save_rx.as_ref()?;
        match rx.try_recv() {
            Ok(outcome) => {
                self.save_rx = None;
                Some(outcome)
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.save_rx = None;
                None
            }
        }
    }
}

pub(super) fn poll_tool_io(tool: &mut super::VietorisRipsTool) {
    if let Some(open_res) = tool.io_tasks.poll_open() {
        match open_res {
            Ok((name, text)) => {
                if let Err(err) = tool.import_point_cloud_text(&name, &text) {
                    tool.error_msg = Some(err);
                }
            }
            Err(err) => tool.error_msg = Some(err),
        }
    }
    if let Some(save_res) = tool.io_tasks.poll_save() {
        match save_res {
            Ok(msg) => {
                tool.error_msg = None;
                tool.status_msg = Some(msg);
            }
            Err(err) => tool.error_msg = Some(err),
        }
    }
}

// [cite:graph_parameters_rust]
