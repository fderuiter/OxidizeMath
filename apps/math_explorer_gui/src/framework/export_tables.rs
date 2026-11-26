//! Typed multi-column table and stride-bounded 2D matrix export types and serializers.

use egui_plot::export::{escape_csv_field, escape_json_string};

/// Maximum number of rows serialized for any bounded 2D raster/matrix table export.
pub const MAX_RASTER_TABLE_ROWS: usize = 65_536;

/// Typed scalar cell value inside an [`ExportTable`].
#[derive(Clone, Debug, PartialEq)]
pub enum ExportCell {
    /// Signed integer value serialized without decimal formatting noise.
    Int(i64),
    /// Floating-point value serialized when finite, or empty/`null` when non-finite.
    Float(f64),
    /// Boolean flag serialized as `true` or `false`.
    Bool(bool),
    /// UTF-8 string value sanitized against CSV formula injection (CWE-1236) and JSON control chars.
    Text(String),
}

impl ExportCell {
    /// Convenience constructor for [`ExportCell::Text`].
    #[must_use]
    pub fn text(value: impl Into<String>) -> Self {
        Self::Text(value.into())
    }

    /// Formats this cell as an RFC 4180 CSV field, sanitizing text via [`escape_csv_field`].
    #[must_use]
    pub fn to_csv_field(&self) -> String {
        match self {
            Self::Int(v) => v.to_string(),
            Self::Float(v) => {
                if v.is_finite() {
                    v.to_string()
                } else {
                    String::new()
                }
            }
            Self::Bool(v) => v.to_string(),
            Self::Text(s) => escape_csv_field(s),
        }
    }

    /// Formats this cell as an RFC 8259 JSON value, mapping non-finite floats to `null`.
    #[must_use]
    pub fn to_json_value(&self) -> String {
        match self {
            Self::Int(v) => v.to_string(),
            Self::Float(v) => {
                if v.is_finite() {
                    v.to_string()
                } else {
                    "null".to_string()
                }
            }
            Self::Bool(v) => v.to_string(),
            Self::Text(s) => format!("\"{}\"", escape_json_string(s)),
        }
    }
}

/// Named multi-column table returned by [`super::InteractiveTool::export_tables`].
#[derive(Clone, Debug, PartialEq)]
pub struct ExportTable {
    /// Human-readable table name used in multi-table CSV `Table` columns and JSON object keys.
    pub name: String,
    /// Ordered column header labels.
    pub headers: Vec<String>,
    /// Row records aligned positionally with `headers`.
    pub rows: Vec<Vec<ExportCell>>,
}

impl ExportTable {
    /// Creates a named [`ExportTable`] from column headers and row cells.
    #[must_use]
    pub fn new(name: impl Into<String>, headers: Vec<String>, rows: Vec<Vec<ExportCell>>) -> Self {
        Self {
            name: name.into(),
            headers,
            rows,
        }
    }

    /// Returns `true` when this table has at least one column header and one data row.
    #[must_use]
    pub fn is_non_empty(&self) -> bool {
        !self.headers.is_empty() && !self.rows.is_empty()
    }

    /// Builds a stride-sampled 2D matrix table capped at [`MAX_RASTER_TABLE_ROWS`] rows.
    #[must_use]
    pub fn from_bounded_grid_2d<F>(
        name: impl Into<String>,
        width: usize,
        height: usize,
        headers: Vec<String>,
        mut cell_fn: F,
    ) -> Self
    where
        F: FnMut(usize, usize) -> Vec<ExportCell>,
    {
        if width == 0 || height == 0 {
            return Self {
                name: name.into(),
                headers,
                rows: Vec::new(),
            };
        }

        let (stride_x, stride_y) = compute_grid_2d_stride(width, height, MAX_RASTER_TABLE_ROWS);
        let cap = width.div_ceil(stride_x) * height.div_ceil(stride_y);
        let mut rows = Vec::with_capacity(cap);
        let mut y = 0;
        while y < height {
            let mut x = 0;
            while x < width {
                rows.push(cell_fn(x, y));
                x += stride_x;
            }
            y += stride_y;
        }

        Self {
            name: name.into(),
            headers,
            rows,
        }
    }
}

/// Computes `(stride_x, stride_y)` so `width.div_ceil(stride_x) * height.div_ceil(stride_y) <= max_rows`.
#[must_use]
pub fn compute_grid_2d_stride(width: usize, height: usize, max_rows: usize) -> (usize, usize) {
    let limit = max_rows.max(1);
    if width == 0 || height == 0 || width.saturating_mul(height) <= limit {
        return (1, 1);
    }
    let ratio = (width as f64 * height as f64) / (limit as f64);
    let init = (ratio.sqrt().ceil() as usize).max(1);
    let mut sx = init.min(width);
    let mut sy = init.min(height);
    while width.div_ceil(sx).saturating_mul(height.div_ceil(sy)) > limit {
        if width.div_ceil(sx) >= height.div_ceil(sy) {
            sx += 1;
        } else {
            sy += 1;
        }
    }
    (sx.max(1), sy.max(1))
}

/// Formats one or more [`ExportTable`]s into an RFC 4180 CSV payload.
///
/// Single-table exports emit the table's escaped column headers directly; multi-table exports
/// prepend a `Table` column over the ordered union of column headers across all non-empty tables.
#[must_use]
pub fn format_tables_csv(tables: &[ExportTable]) -> Option<String> {
    let non_empty: Vec<&ExportTable> = tables.iter().filter(|t| t.is_non_empty()).collect();
    match non_empty.as_slice() {
        [] => None,
        [single] => Some(format_single_table_csv(single)),
        many => Some(format_multi_table_csv(many)),
    }
}

fn format_single_table_csv(table: &ExportTable) -> String {
    let col_count = table.headers.len();
    let header_line: Vec<String> = table.headers.iter().map(|h| escape_csv_field(h)).collect();
    let mut csv = header_line.join(",");
    csv.push('\n');
    for row in &table.rows {
        for col_idx in 0..col_count {
            if col_idx > 0 {
                csv.push(',');
            }
            if let Some(cell) = row.get(col_idx) {
                csv.push_str(&cell.to_csv_field());
            }
        }
        csv.push('\n');
    }
    csv
}

fn collect_union_headers<'a>(tables: &[&'a ExportTable]) -> Vec<&'a str> {
    let mut union_headers: Vec<&'a str> = Vec::new();
    for table in tables {
        for h in &table.headers {
            if !union_headers.contains(&h.as_str()) {
                union_headers.push(h.as_str());
            }
        }
    }
    union_headers
}

fn format_multi_table_csv(tables: &[&ExportTable]) -> String {
    let union_headers = collect_union_headers(tables);
    let mut csv = escape_csv_field("Table");
    for h in &union_headers {
        csv.push(',');
        csv.push_str(&escape_csv_field(h));
    }
    csv.push('\n');

    let mut row_slots = vec![String::new(); union_headers.len()];
    for table in tables {
        let escaped_table = escape_csv_field(&table.name);
        let col_map: Vec<Option<usize>> = table
            .headers
            .iter()
            .map(|h| union_headers.iter().position(|&uh| uh == h.as_str()))
            .collect();
        for row in &table.rows {
            row_slots.iter_mut().for_each(String::clear);
            for (idx, cell) in row.iter().enumerate() {
                if let Some(Some(target_col)) = col_map.get(idx) {
                    row_slots[*target_col] = cell.to_csv_field();
                }
            }
            csv.push_str(&escaped_table);
            for slot in &row_slots {
                csv.push(',');
                csv.push_str(slot);
            }
            csv.push('\n');
        }
    }
    csv
}

/// Formats one or more [`ExportTable`]s into an RFC 8259 JSON payload.
///
/// Single-table exports emit a JSON array of row objects; multi-table exports group named tables
/// under `{ "tables": { "<name>": [ ... ] } }`.
#[must_use]
pub fn format_tables_json(tables: &[ExportTable]) -> Option<String> {
    let non_empty: Vec<&ExportTable> = tables.iter().filter(|t| t.is_non_empty()).collect();
    match non_empty.as_slice() {
        [] => None,
        [single] => Some(format_table_rows_json(single, "  ")),
        many => Some(format_multi_table_json(many)),
    }
}

fn format_table_rows_json(table: &ExportTable, row_indent: &str) -> String {
    let escaped_headers: Vec<String> = table
        .headers
        .iter()
        .map(|h| escape_json_string(h))
        .collect();
    let mut json = String::from("[\n");
    for (r_idx, row) in table.rows.iter().enumerate() {
        if r_idx > 0 {
            json.push_str(",\n");
        }
        json.push_str(row_indent);
        json.push('{');
        for (c_idx, key) in escaped_headers.iter().enumerate() {
            if c_idx > 0 {
                json.push_str(", ");
            }
            let val = row
                .get(c_idx)
                .map_or_else(|| "null".to_string(), ExportCell::to_json_value);
            json.push('"');
            json.push_str(key);
            json.push_str("\": ");
            json.push_str(&val);
        }
        json.push('}');
    }
    json.push('\n');
    if row_indent.len() >= 2 {
        json.push_str(&row_indent[..row_indent.len() - 2]);
    }
    json.push(']');
    json
}

fn format_multi_table_json(tables: &[&ExportTable]) -> String {
    let mut json = String::from("{\n  \"tables\": {\n");
    let mut used_names: Vec<String> = Vec::with_capacity(tables.len());
    for (idx, table) in tables.iter().enumerate() {
        if idx > 0 {
            json.push_str(",\n");
        }
        let base_name = if table.name.trim().is_empty() {
            format!("table_{idx}")
        } else {
            table.name.clone()
        };
        let unique_name = if used_names.contains(&base_name) {
            format!("{base_name}_{idx}")
        } else {
            base_name
        };
        used_names.push(unique_name.clone());
        let rows_json = format_table_rows_json(table, "      ");
        json.push_str("    \"");
        json.push_str(&escape_json_string(&unique_name));
        json.push_str("\": ");
        json.push_str(&rows_json);
    }
    json.push_str("\n  }\n}");
    json
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_grid_stride_caps_rows_at_65536() {
        let small = ExportTable::from_bounded_grid_2d(
            "Small",
            16,
            8,
            vec!["x".into(), "y".into(), "v".into()],
            |x, y| {
                vec![
                    ExportCell::Int(x as i64),
                    ExportCell::Int(y as i64),
                    ExportCell::Float(1.0),
                ]
            },
        );
        assert_eq!(small.rows.len(), 128);

        let large = ExportTable::from_bounded_grid_2d(
            "Large",
            512,
            512,
            vec!["x".into(), "y".into(), "v".into()],
            |x, y| {
                vec![
                    ExportCell::Int(x as i64),
                    ExportCell::Int(y as i64),
                    ExportCell::Float(0.5),
                ]
            },
        );
        assert!(large.rows.len() <= MAX_RASTER_TABLE_ROWS);
        assert_eq!(large.rows.len(), 65_536);

        let rect = ExportTable::from_bounded_grid_2d(
            "Rect",
            1024,
            512,
            vec!["x".into(), "y".into()],
            |x, y| vec![ExportCell::Int(x as i64), ExportCell::Int(y as i64)],
        );
        assert!(rect.rows.len() <= MAX_RASTER_TABLE_ROWS);
    }

    #[test]
    fn test_cwe1236_formula_injection_and_multi_table_csv_json() {
        let t1 = ExportTable::new(
            "=EvilTable",
            vec!["+col_a".into(), "shared".into()],
            vec![vec![ExportCell::text("@cmd"), ExportCell::Float(1.5)]],
        );
        let t2 = ExportTable::new(
            "Second",
            vec!["shared".into(), "extra".into()],
            vec![vec![ExportCell::Float(f64::NAN), ExportCell::Bool(true)]],
        );

        let csv = format_tables_csv(&[t1.clone(), t2.clone()]).expect("CSV");
        assert!(csv.starts_with("Table,\"'+col_a\",shared,extra\n"));
        assert!(csv.contains("\"'=EvilTable\",\"'@cmd\",1.5,\n"));
        assert!(csv.contains("Second,,\"\",true\n") || csv.contains("Second,,,true\n"));

        let json = format_tables_json(&[t1, t2]).expect("JSON");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        let tables = parsed["tables"].as_object().expect("tables object");
        assert_eq!(tables.len(), 2);
        assert!(tables["Second"][0]["shared"].is_null());
    }
}
