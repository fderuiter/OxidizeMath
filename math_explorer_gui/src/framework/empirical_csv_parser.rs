//! Bounded tabular CSV/TSV parser and numeric column extractor for inbound calibration datasets.

use egui_plot::export::split_line;

/// Maximum permitted raw byte length for an inbound empirical CSV file (`2 MiB`).
pub const MAX_EMPIRICAL_CSV_BYTES: usize = 2 * 1024 * 1024;

/// Maximum permitted numeric observation row count per empirical CSV dataset (`10,000`).
pub const MAX_EMPIRICAL_CSV_ROWS: usize = 10_000;

/// Extracted `(x, y, aux)` numeric column vectors sorted by ascending `x`.
pub type EmpiricalTripleSeries = (Vec<f64>, Vec<f64>, Vec<f64>);

/// Validated rectangular table of finite numeric columns parsed from CSV, TSV, or semicolon text.
#[derive(Clone, Debug, PartialEq)]
pub struct EmpiricalCsvTable {
    /// Human-readable source dataset or filename label.
    pub name: String,
    /// Column labels (parsed from header row or synthesized as `Column 1`, `Column 2`, ...).
    pub headers: Vec<String>,
    /// Column-major finite numerical vectors; every column has identical length `>= 2`.
    pub columns: Vec<Vec<f64>>,
}

impl EmpiricalCsvTable {
    /// Parses and validates raw UTF-8 CSV/TSV bytes, enforcing [`MAX_EMPIRICAL_CSV_BYTES`]
    /// and [`MAX_EMPIRICAL_CSV_ROWS`].
    pub fn parse_bytes(name: impl Into<String>, bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_EMPIRICAL_CSV_BYTES {
            return Err(format!(
                "CSV payload ({} bytes) exceeds 2 MiB safety limit ({} bytes)",
                bytes.len(),
                MAX_EMPIRICAL_CSV_BYTES
            ));
        }
        let text = std::str::from_utf8(bytes)
            .map_err(|_| "CSV file is not valid UTF-8 text".to_string())?;
        Self::parse_str(name, text)
    }

    /// Parses and validates tabular CSV/TSV text using `egui_plot::export::split_line`.
    pub fn parse_str(name: impl Into<String>, content: &str) -> Result<Self, String> {
        if content.len() > MAX_EMPIRICAL_CSV_BYTES {
            return Err(format!(
                "CSV payload ({} bytes) exceeds 2 MiB safety limit ({} bytes)",
                content.len(),
                MAX_EMPIRICAL_CSV_BYTES
            ));
        }
        let stripped = content.strip_prefix('\u{feff}').unwrap_or(content);
        let lines: Vec<&str> = stripped
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .collect();
        if lines.is_empty() {
            return Err("CSV dataset is empty".to_string());
        }
        let delimiter = detect_delimiter(lines[0]);
        let first_tokens = split_line(lines[0], delimiter);
        let col_count = first_tokens.len();
        if col_count < 2 {
            return Err("CSV dataset requires at least 2 columns (X and Y)".to_string());
        }
        let has_header = first_tokens
            .iter()
            .any(|tok| parse_finite_f64(tok).is_none());
        let headers = build_headers(&first_tokens, col_count, has_header);
        let data_start = usize::from(has_header);
        let data_lines = &lines[data_start..];
        validate_row_bounds(data_lines.len())?;
        let columns = parse_numeric_columns(data_lines, delimiter, col_count, data_start)?;
        Ok(Self {
            name: sanitize_dataset_label(name.into()),
            headers,
            columns,
        })
    }

    /// Number of validated numeric data rows.
    pub fn row_count(&self) -> usize {
        self.columns.first().map_or(0, Vec::len)
    }

    /// Number of columns in the dataset.
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Extracts `(x, y)` vectors from `x_col` and `y_col` sorted by ascending `x`.
    pub fn extract_xy_sorted(
        &self,
        x_col: usize,
        y_col: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        if x_col >= self.columns.len() || y_col >= self.columns.len() {
            return Err("Selected CSV column index is out of range".to_string());
        }
        let mut pairs: Vec<(f64, f64)> = self.columns[x_col]
            .iter()
            .copied()
            .zip(self.columns[y_col].iter().copied())
            .collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        if pairs.len() < 2 {
            return Err("At least 2 data rows are required for calibration".to_string());
        }
        let x_span = pairs.last().unwrap().0 - pairs.first().unwrap().0;
        if !x_span.is_finite() || x_span.abs() <= 1e-15 {
            return Err("Selected X column has zero variation across rows".to_string());
        }
        let xs = pairs.iter().map(|p| p.0).collect();
        let ys = pairs.iter().map(|p| p.1).collect();
        Ok((xs, ys))
    }

    /// Extracts `(x, y, aux)` vectors sorted by ascending `x`.
    pub fn extract_xyz_sorted(
        &self,
        x_col: usize,
        y_col: usize,
        aux_col: usize,
    ) -> Result<EmpiricalTripleSeries, String> {
        if x_col >= self.columns.len()
            || y_col >= self.columns.len()
            || aux_col >= self.columns.len()
        {
            return Err("Selected CSV column index is out of range".to_string());
        }
        let mut triples: Vec<(f64, f64, f64)> = (0..self.row_count())
            .map(|i| {
                (
                    self.columns[x_col][i],
                    self.columns[y_col][i],
                    self.columns[aux_col][i],
                )
            })
            .collect();
        triples.sort_by(|a, b| a.0.total_cmp(&b.0));
        let xs = triples.iter().map(|t| t.0).collect();
        let ys = triples.iter().map(|t| t.1).collect();
        let zs = triples.iter().map(|t| t.2).collect();
        Ok((xs, ys, zs))
    }

    /// Returns `[x, y]` scatter points for plotting the currently mapped columns.
    pub fn scatter_points(&self, x_col: usize, y_col: usize) -> Vec<[f64; 2]> {
        if x_col >= self.columns.len() || y_col >= self.columns.len() {
            return Vec::new();
        }
        self.columns[x_col]
            .iter()
            .copied()
            .zip(self.columns[y_col].iter().copied())
            .map(|(x, y)| [x, y])
            .collect()
    }
}

fn detect_delimiter(first_line: &str) -> char {
    if first_line.contains('\t') {
        '\t'
    } else if first_line.contains(';') {
        ';'
    } else {
        ','
    }
}

fn parse_finite_f64(token: &str) -> Option<f64> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return None;
    }
    trimmed.parse::<f64>().ok().filter(|v| v.is_finite())
}

fn build_headers(first_tokens: &[String], col_count: usize, has_header: bool) -> Vec<String> {
    if has_header {
        first_tokens
            .iter()
            .enumerate()
            .map(|(idx, raw)| {
                let clean = raw.trim();
                if clean.is_empty() {
                    format!("Column {}", idx + 1)
                } else {
                    clean.to_string()
                }
            })
            .collect()
    } else {
        (0..col_count)
            .map(|idx| format!("Column {}", idx + 1))
            .collect()
    }
}

fn validate_row_bounds(row_count: usize) -> Result<(), String> {
    if row_count < 2 {
        return Err(format!(
            "CSV dataset has {row_count} data row(s); at least 2 rows are required for calibration"
        ));
    }
    if row_count > MAX_EMPIRICAL_CSV_ROWS {
        return Err(format!(
            "CSV dataset has {row_count} rows, exceeding the 10,000-row safety limit"
        ));
    }
    Ok(())
}

fn parse_numeric_columns(
    data_lines: &[&str],
    delimiter: char,
    col_count: usize,
    line_offset: usize,
) -> Result<Vec<Vec<f64>>, String> {
    let mut columns = vec![Vec::with_capacity(data_lines.len()); col_count];
    for (row_idx, line) in data_lines.iter().enumerate() {
        let tokens = split_line(line, delimiter);
        if tokens.len() != col_count {
            return Err(format!(
                "Malformed CSV at line {}: expected {} columns, found {}",
                row_idx + line_offset + 1,
                col_count,
                tokens.len()
            ));
        }
        for (col_idx, tok) in tokens.iter().enumerate() {
            let val = parse_finite_f64(tok).ok_or_else(|| {
                format!(
                    "Non-finite or non-numeric value '{}' at line {}, column {}",
                    tok.trim(),
                    row_idx + line_offset + 1,
                    col_idx + 1
                )
            })?;
            columns[col_idx].push(val);
        }
    }
    Ok(columns)
}

fn sanitize_dataset_label(label: String) -> String {
    let trimmed = label.trim();
    if trimmed.is_empty() {
        "empirical_dataset.csv".to_string()
    } else {
        trimmed.to_string()
    }
}
