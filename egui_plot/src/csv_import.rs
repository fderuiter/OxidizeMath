//! Bounded, deterministic CSV and TSV dataset importer for [`crate::Plot`].
//!
//! Enforces strict untrusted-input caps (`2 MiB` byte payload, `10,000` finite data rows,
//! and `64` series) prior to allocating unbounded point buffers, preserves 3-column `id,x,y`
//! single-curve logs and headerless wide tables, parses decimal commas on `;` and `\t` rows,
//! and reverses CWE-1236 leading single-quote prefixes on series labels.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::export::ImportedDataset;
use crate::file_io::strip_utf8_bom;

/// Maximum untrusted CSV/TSV payload size (`2 MiB`, matching `ToolStateBounds::DEFAULT`).
pub const MAX_CSV_IMPORT_BYTES: usize = 2 * 1024 * 1024;

/// Maximum number of finite data rows imported per CSV/TSV payload.
pub const MAX_CSV_IMPORT_ROWS: usize = 10_000;

/// Maximum number of series imported per CSV/TSV payload.
pub const MAX_CSV_IMPORT_SERIES: usize = 64;

const SAMPLE_SCAN_ROWS: usize = 128;

/// Structured error returned when bounded CSV/TSV parsing fails closed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CsvImportError {
    /// Input byte length exceeded [`MAX_CSV_IMPORT_BYTES`] prior to allocation.
    PayloadTooLarge {
        /// Actual UTF-8 byte length of the untrusted payload.
        byte_len: usize,
        /// Configured byte ceiling (`2 MiB`).
        max_bytes: usize,
    },
    /// Payload contained no finite numeric `(x, y)` data points.
    NoValidData,
}

impl fmt::Display for CsvImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PayloadTooLarge {
                byte_len,
                max_bytes,
            } => write!(
                f,
                "CSV payload ({:.2} MiB) exceeds maximum allowed size of {:.1} MiB.",
                *byte_len as f64 / (1024.0 * 1024.0),
                *max_bytes as f64 / (1024.0 * 1024.0)
            ),
            Self::NoValidData => write!(f, "No valid numeric (x, y) data found in CSV/TSV input."),
        }
    }
}

/// Bounded result of parsing a CSV/TSV dataset payload.
#[derive(Clone, Debug, PartialEq)]
pub struct CsvImportOutcome {
    /// Parsed plottable series in source appearance order.
    pub datasets: Vec<ImportedDataset>,
    /// Count of finite rows accepted across the payload (`<= MAX_CSV_IMPORT_ROWS`).
    pub rows_imported: usize,
    /// `true` when additional finite rows beyond [`MAX_CSV_IMPORT_ROWS`] were truncated.
    pub truncated_rows: bool,
    /// `true` when additional series beyond [`MAX_CSV_IMPORT_SERIES`] were truncated.
    pub truncated_series: bool,
}

/// Strips a single leading quote prefix (`'`) added by `escape_csv_field` or
/// `sanitize_tsv_label` for formula-trigger characters (`=`, `+`, `-`, `@`, `\t`, `\r`).
#[must_use]
pub fn unescape_formula_label(raw: &str) -> String {
    if let Some(rest) = raw.strip_prefix('\'') {
        let triggers = ['=', '+', '-', '@', '\t', '\r'];
        let is_escaped_formula = rest.starts_with(triggers)
            || (rest.starts_with(char::is_whitespace)
                && rest.trim_start().starts_with(['=', '+', '-', '@']));
        if is_escaped_formula {
            return rest.to_owned();
        }
    }
    raw.to_owned()
}

/// Parse a single delimited text line into tokens, respecting RFC 4180 quoted strings.
#[must_use]
pub fn split_line(line: &str, delimiter: char) -> Vec<String> {
    split_line_bounded(line, delimiter, usize::MAX).0
}

fn split_line_bounded(line: &str, delimiter: char, max_tokens: usize) -> (Vec<String>, bool) {
    if line.is_empty() {
        return (vec![String::new()], false);
    }
    let chars: Vec<char> = line.chars().collect();
    let mut tokens = Vec::with_capacity(chars.len().min(8));
    let mut i = 0;
    while i < chars.len() {
        if tokens.len() >= max_tokens {
            return (tokens, true);
        }
        while i < chars.len() && chars[i] != delimiter && chars[i].is_whitespace() {
            i += 1;
        }
        let token = if i < chars.len() && chars[i] == '"' {
            read_quoted_token(&chars, &mut i, delimiter)
        } else {
            read_unquoted_token(&chars, &mut i, delimiter)
        };
        tokens.push(token);
        if i < chars.len() && chars[i] == delimiter {
            i += 1;
            if i == chars.len() {
                if tokens.len() >= max_tokens {
                    return (tokens, true);
                }
                tokens.push(String::new());
                break;
            }
        }
    }
    (tokens, false)
}

fn read_quoted_token(chars: &[char], i: &mut usize, delimiter: char) -> String {
    *i += 1;
    let mut token = String::new();
    while *i < chars.len() {
        if chars[*i] == '"' {
            if *i + 1 < chars.len() && chars[*i + 1] == '"' {
                token.push('"');
                *i += 2;
            } else {
                *i += 1;
                break;
            }
        } else {
            token.push(chars[*i]);
            *i += 1;
        }
    }
    while *i < chars.len() && chars[*i] != delimiter && chars[*i].is_whitespace() {
        *i += 1;
    }
    token
}

fn read_unquoted_token(chars: &[char], i: &mut usize, delimiter: char) -> String {
    let start = *i;
    while *i < chars.len() && chars[*i] != delimiter {
        *i += 1;
    }
    chars[start..*i]
        .iter()
        .collect::<String>()
        .trim()
        .to_owned()
}

fn detect_delimiter(first_line: &str) -> char {
    let (mut in_quotes, mut tabs, mut semis) = (false, 0_usize, 0_usize);
    let mut chars = first_line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '"' {
            if in_quotes && chars.peek() == Some(&'"') {
                let _ = chars.next();
            } else {
                in_quotes = !in_quotes;
            }
        } else if !in_quotes {
            if ch == '\t' {
                tabs += 1;
            } else if ch == ';' {
                semis += 1;
            }
        }
    }
    if tabs > 0 {
        '\t'
    } else if semis > 0 {
        ';'
    } else {
        ','
    }
}

fn parse_num(raw: &str, allow_decimal_comma: bool) -> Option<f64> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(val) = trimmed.parse::<f64>() {
        return val.is_finite().then_some(val);
    }
    if allow_decimal_comma
        && !trimmed.contains('.')
        && trimmed.bytes().filter(|&b| b == b',').count() == 1
    {
        let normalized = trimmed.replace(',', ".");
        if let Ok(val) = normalized.parse::<f64>() {
            return val.is_finite().then_some(val);
        }
    }
    None
}

enum TableLayout {
    Grouped3Col,
    IdXY3Col { series_name: String },
    Wide { series_names: Vec<String> },
}

fn is_id_column_header(h0: &str) -> bool {
    matches!(
        h0.trim().to_ascii_lowercase().as_str(),
        "id" | "#" | "index" | "idx" | "row" | "row_id" | "sample_id" | "point_id"
    )
}

fn classify_header(first_tokens: Vec<String>, allow_decimal_comma: bool) -> (Vec<String>, bool) {
    let all_numeric = first_tokens
        .iter()
        .all(|t| parse_num(t, allow_decimal_comma).is_some());
    if all_numeric {
        return (Vec::new(), false);
    }
    let headerless_labeled_triple = first_tokens.len() >= 3
        && parse_num(&first_tokens[0], allow_decimal_comma).is_none()
        && first_tokens[1..]
            .iter()
            .all(|t| parse_num(t, allow_decimal_comma).is_some());
    if headerless_labeled_triple {
        (Vec::new(), false)
    } else {
        (first_tokens, true)
    }
}

fn detect_3col_layout(
    headers: &[String],
    sample_rows: &[Vec<String>],
    allow_decimal_comma: bool,
) -> Option<TableLayout> {
    let h0 = headers.first().map_or("", |s| s.trim());
    let h1 = headers.get(1).map_or("", |s| s.trim());
    let h2 = headers.get(2).map_or("", |s| s.trim());
    let is_xy = h1.eq_ignore_ascii_case("x") && h2.eq_ignore_ascii_case("y");
    let is_id = is_id_column_header(h0);
    let is_dataset_export = h0.eq_ignore_ascii_case("dataset") && is_xy;

    let mut seen_all = HashSet::new();
    let mut seen_non_num = HashSet::new();
    let (mut valid_rows, mut all_non_num) = (0_usize, true);
    let (mut any_rep, mut non_num_rep) = (false, false);

    for row in sample_rows {
        if row.len() >= 3
            && !row[0].trim().is_empty()
            && parse_num(&row[1], allow_decimal_comma).is_some()
            && parse_num(&row[2], allow_decimal_comma).is_some()
        {
            valid_rows += 1;
            let c0 = row[0].trim();
            if !seen_all.insert(c0) {
                any_rep = true;
            }
            if parse_num(c0, allow_decimal_comma).is_none() {
                if !seen_non_num.insert(c0) {
                    non_num_rep = true;
                }
            } else {
                all_non_num = false;
            }
        }
    }

    if !is_id
        && (non_num_rep
            || (is_xy && any_rep)
            || (is_dataset_export && valid_rows > 0 && all_non_num))
    {
        return Some(TableLayout::Grouped3Col);
    }
    if valid_rows > 0 && (is_xy || all_non_num) {
        let series_name = if !h2.is_empty() && !h2.eq_ignore_ascii_case("y") {
            unescape_formula_label(h2)
        } else {
            "Imported Dataset".to_owned()
        };
        return Some(TableLayout::IdXY3Col { series_name });
    }
    None
}

fn build_wide_names(headers: &[String], max_cols: usize) -> (Vec<String>, bool) {
    let raw_y_cols = max_cols.saturating_sub(1);
    let truncated = raw_y_cols > MAX_CSV_IMPORT_SERIES;
    let y_cols = raw_y_cols.min(MAX_CSV_IMPORT_SERIES);
    if y_cols <= 1 {
        let name = headers
            .get(1)
            .map(|s| unescape_formula_label(s.trim()))
            .filter(|s| !s.is_empty() && !s.eq_ignore_ascii_case("y"))
            .unwrap_or_else(|| "Imported Dataset".to_owned());
        return (vec![name], truncated);
    }
    let mut names = Vec::with_capacity(y_cols);
    let mut counts: HashMap<String, usize> = HashMap::with_capacity(y_cols);
    for col in 1..=y_cols {
        let base = headers
            .get(col)
            .map(|s| unescape_formula_label(s.trim()))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("Series {col}"));
        let entry = counts.entry(base.clone()).or_insert(0);
        *entry += 1;
        if *entry == 1 {
            names.push(base);
        } else {
            names.push(format!("{base} ({entry})"));
        }
    }
    (names, truncated)
}

struct SeriesCollector {
    order: Vec<String>,
    index: HashMap<String, usize>,
    points: Vec<Vec<[f64; 2]>>,
    rows_imported: usize,
    truncated_rows: bool,
    truncated_series: bool,
}

impl SeriesCollector {
    fn new(truncated_series: bool) -> Self {
        Self {
            order: Vec::new(),
            index: HashMap::new(),
            points: Vec::new(),
            rows_imported: 0,
            truncated_rows: false,
            truncated_series,
        }
    }

    fn push_point(&mut self, name: &str, pt: [f64; 2]) -> bool {
        let idx = if let Some(&existing) = self.index.get(name) {
            existing
        } else if self.order.len() < MAX_CSV_IMPORT_SERIES {
            let next = self.order.len();
            self.order.push(name.to_owned());
            self.index.insert(name.to_owned(), next);
            self.points.push(Vec::new());
            next
        } else {
            self.truncated_series = true;
            return false;
        };
        self.points[idx].push(pt);
        true
    }

    fn finish(self) -> Result<CsvImportOutcome, CsvImportError> {
        let datasets: Vec<ImportedDataset> = self
            .order
            .into_iter()
            .zip(self.points)
            .filter(|(_, pts)| !pts.is_empty())
            .map(|(name, pts)| ImportedDataset::new(name, pts))
            .collect();
        if datasets.is_empty() {
            return Err(CsvImportError::NoValidData);
        }
        Ok(CsvImportOutcome {
            datasets,
            rows_imported: self.rows_imported,
            truncated_rows: self.truncated_rows,
            truncated_series: self.truncated_series,
        })
    }
}

fn ingest_wide_row(
    row: &[String],
    series_names: &[String],
    allow_decimal_comma: bool,
    collector: &mut SeriesCollector,
) {
    let Some(x) = row.first().and_then(|s| parse_num(s, allow_decimal_comma)) else {
        return;
    };
    let mut row_had_point = false;
    for (i, y_str) in row.iter().skip(1).enumerate() {
        let Some(y) = parse_num(y_str, allow_decimal_comma) else {
            continue;
        };
        let Some(series_name) = series_names.get(i) else {
            collector.truncated_series = true;
            continue;
        };
        if collector.rows_imported >= MAX_CSV_IMPORT_ROWS {
            collector.truncated_rows = true;
            return;
        }
        let _ = collector.push_point(series_name, [x, y]);
        row_had_point = true;
    }
    if row_had_point {
        collector.rows_imported += 1;
    }
}

fn ingest_row(
    row: &[String],
    layout: &TableLayout,
    allow_decimal_comma: bool,
    collector: &mut SeriesCollector,
) {
    match layout {
        TableLayout::Grouped3Col => {
            if row.len() >= 3
                && !row[0].trim().is_empty()
                && let (Some(x), Some(y)) = (
                    parse_num(&row[1], allow_decimal_comma),
                    parse_num(&row[2], allow_decimal_comma),
                )
            {
                if collector.rows_imported >= MAX_CSV_IMPORT_ROWS {
                    collector.truncated_rows = true;
                    return;
                }
                let label = unescape_formula_label(row[0].trim());
                if collector.push_point(&label, [x, y]) {
                    collector.rows_imported += 1;
                }
            }
        }
        TableLayout::IdXY3Col { series_name } => {
            if row.len() >= 3
                && let (Some(x), Some(y)) = (
                    parse_num(&row[1], allow_decimal_comma),
                    parse_num(&row[2], allow_decimal_comma),
                )
            {
                if collector.rows_imported >= MAX_CSV_IMPORT_ROWS {
                    collector.truncated_rows = true;
                    return;
                }
                if collector.push_point(series_name, [x, y]) {
                    collector.rows_imported += 1;
                }
            }
        }
        TableLayout::Wide { series_names } => {
            ingest_wide_row(row, series_names, allow_decimal_comma, collector);
        }
    }
}

/// Parses untrusted CSV, TSV, or semicolon-delimited text with strict byte, row, and series caps.
fn resolve_table_layout(
    headers: &[String],
    sample_rows: &[Vec<String>],
    allow_decimal_comma: bool,
    truncated_cols: &mut bool,
) -> Result<TableLayout, CsvImportError> {
    let max_cols = headers
        .len()
        .max(sample_rows.iter().map(Vec::len).max().unwrap_or(0));
    if max_cols < 2 {
        return Err(CsvImportError::NoValidData);
    }
    if max_cols == 3
        && let Some(layout) = detect_3col_layout(headers, sample_rows, allow_decimal_comma)
    {
        return Ok(layout);
    }
    let (series_names, trunc) = build_wide_names(headers, max_cols);
    *truncated_cols |= trunc;
    Ok(TableLayout::Wide { series_names })
}

/// Parse bounded CSV/TSV text into a detailed [`CsvImportOutcome`] or structured [`CsvImportError`].
///
/// Fails closed with [`CsvImportError::PayloadTooLarge`] before allocating token or point buffers
/// whenever `content.len() > MAX_CSV_IMPORT_BYTES`.
pub fn parse_csv_bounded(content: &str) -> Result<CsvImportOutcome, CsvImportError> {
    if content.len() > MAX_CSV_IMPORT_BYTES {
        return Err(CsvImportError::PayloadTooLarge {
            byte_len: content.len(),
            max_bytes: MAX_CSV_IMPORT_BYTES,
        });
    }

    let mut clean_lines = strip_utf8_bom(content)
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty() && !s.starts_with('#'));
    let Some(first_line) = clean_lines.next() else {
        return Err(CsvImportError::NoValidData);
    };

    let delimiter = detect_delimiter(first_line);
    let allow_decimal_comma = matches!(delimiter, ';' | '\t');
    let max_tokens = MAX_CSV_IMPORT_SERIES + 1;
    let (first_tokens, mut truncated_cols) = split_line_bounded(first_line, delimiter, max_tokens);
    let (headers, has_header) = classify_header(first_tokens, allow_decimal_comma);

    let mut sample_rows: Vec<Vec<String>> = Vec::with_capacity(SAMPLE_SCAN_ROWS);
    if !has_header {
        let (first_data, trunc) = split_line_bounded(first_line, delimiter, max_tokens);
        truncated_cols |= trunc;
        sample_rows.push(first_data);
    }
    while sample_rows.len() < SAMPLE_SCAN_ROWS {
        let Some(line) = clean_lines.next() else {
            break;
        };
        let (tokens, trunc) = split_line_bounded(line, delimiter, max_tokens);
        truncated_cols |= trunc;
        sample_rows.push(tokens);
    }

    let layout = resolve_table_layout(
        &headers,
        &sample_rows,
        allow_decimal_comma,
        &mut truncated_cols,
    )?;
    let mut collector = SeriesCollector::new(truncated_cols);
    for row in &sample_rows {
        ingest_row(row, &layout, allow_decimal_comma, &mut collector);
    }
    for line in clean_lines {
        if collector.truncated_rows {
            break;
        }
        let (tokens, trunc) = split_line_bounded(line, delimiter, max_tokens);
        collector.truncated_series |= trunc;
        ingest_row(&tokens, &layout, allow_decimal_comma, &mut collector);
    }
    collector.finish()
}

/// Parse tabular dataset text (CSV, TSV, semicolon-separated) into imported datasets.
///
/// Auto-detects delimiters and headers, enforces [`MAX_CSV_IMPORT_BYTES`], [`MAX_CSV_IMPORT_ROWS`],
/// and [`MAX_CSV_IMPORT_SERIES`], converts rows into finite `[f64; 2]` pairs, and unescapes
/// formula-trigger quote prefixes on series labels.
#[must_use]
pub fn parse_csv(content: &str) -> Vec<ImportedDataset> {
    parse_csv_bounded(content)
        .map(|outcome| outcome.datasets)
        .unwrap_or_default()
}

#[cfg(test)]
#[path = "csv_import_tests.rs"]
mod tests;
