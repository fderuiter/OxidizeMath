//! Typed tabular export structures, bounded 2D matrix stride samplers, and RFC 4180 / RFC 8259 serializers.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use egui_plot::export::{escape_csv_field, escape_json_string};

use super::{ExportFormat, SimulationFramework};

/// Maximum number of cell records emitted when serializing a 2D raster matrix table (`65,536` rows).
pub const MAX_EXPORT_MATRIX_ENTRIES: usize = 65_536;

/// Typed scalar cell value inside an [`ExportTable`].
#[derive(Clone, Debug, PartialEq)]
pub enum ExportCell {
    /// UTF-8 text string sanitized against spreadsheet formula injection in CSV and escaped in JSON.
    Text(String),
    /// Signed 64-bit integer value.
    Int(i64),
    /// Unsigned 64-bit integer value.
    UInt(u64),
    /// 64-bit floating-point value (non-finite values serialize as empty string in CSV and `null` in JSON).
    Float(f64),
    /// Boolean flag (`true` or `false`).
    Bool(bool),
}

impl ExportCell {
    /// Formats this cell for RFC 4180 CSV output, routing text values through [`escape_csv_field`].
    #[must_use]
    pub fn to_csv_field(&self) -> String {
        match self {
            Self::Text(s) => escape_csv_field(s),
            Self::Int(v) => v.to_string(),
            Self::UInt(v) => v.to_string(),
            Self::Float(v) => {
                if v.is_finite() {
                    v.to_string()
                } else {
                    String::new()
                }
            }
            Self::Bool(b) => b.to_string(),
        }
    }

    /// Formats this cell as an RFC 8259 JSON value literal.
    #[must_use]
    pub fn to_json_value(&self) -> String {
        match self {
            Self::Text(s) => format!("\"{}\"", escape_json_string(s)),
            Self::Int(v) => v.to_string(),
            Self::UInt(v) => v.to_string(),
            Self::Float(v) => {
                if v.is_finite() {
                    v.to_string()
                } else {
                    "null".to_string()
                }
            }
            Self::Bool(b) => b.to_string(),
        }
    }
}

/// Named multi-column table returned by [`super::InteractiveTool::export_tables`].
#[derive(Clone, Debug, PartialEq)]
pub struct ExportTable {
    /// Logical table name (emitted in multi-table CSV `Table` column and multi-table JSON object keys).
    pub name: String,
    /// Ordered column header labels.
    pub headers: Vec<String>,
    /// Optional explicit JSON object property keys aligned with `headers` (used for legacy plot compatibility).
    pub json_keys: Option<Vec<String>>,
    /// Row-major cell records aligned with `headers`.
    pub rows: Vec<Vec<ExportCell>>,
}

impl ExportTable {
    /// Creates a named [`ExportTable`] whose JSON object keys match `headers`.
    #[must_use]
    pub fn new(name: impl Into<String>, headers: Vec<String>, rows: Vec<Vec<ExportCell>>) -> Self {
        Self {
            name: name.into(),
            headers,
            json_keys: None,
            rows,
        }
    }

    /// Converts legacy 2D `(Dataset, X, Y)` plot series into a single [`ExportTable`].
    #[must_use]
    pub fn from_datasets_2d(datasets: &[(String, Vec<[f64; 2]>)]) -> Option<Self> {
        let mut rows = Vec::new();
        for (series_name, pts) in datasets {
            for p in pts {
                if p[0].is_finite() && p[1].is_finite() {
                    rows.push(vec![
                        ExportCell::Text(series_name.clone()),
                        ExportCell::Float(p[0]),
                        ExportCell::Float(p[1]),
                    ]);
                }
            }
        }
        if rows.is_empty() {
            return None;
        }
        Some(Self {
            name: "Plot Series".to_string(),
            headers: vec!["Dataset".into(), "X".into(), "Y".into()],
            json_keys: Some(vec!["dataset".into(), "x".into(), "y".into()]),
            rows,
        })
    }

    /// Converts legacy 3D `(Dataset, X, Y, Z)` plot series into a single [`ExportTable`].
    #[must_use]
    pub fn from_datasets_3d(datasets: &[(String, Vec<[f64; 3]>)]) -> Option<Self> {
        let mut rows = Vec::new();
        for (series_name, pts) in datasets {
            for p in pts {
                if p[0].is_finite() && p[1].is_finite() && p[2].is_finite() {
                    rows.push(vec![
                        ExportCell::Text(series_name.clone()),
                        ExportCell::Float(p[0]),
                        ExportCell::Float(p[1]),
                        ExportCell::Float(p[2]),
                    ]);
                }
            }
        }
        if rows.is_empty() {
            return None;
        }
        Some(Self {
            name: "Plot Series 3D".to_string(),
            headers: vec!["Dataset".into(), "X".into(), "Y".into(), "Z".into()],
            json_keys: Some(vec!["dataset".into(), "x".into(), "y".into(), "z".into()]),
            rows,
        })
    }

    /// Converts legacy 3D or 2D plot datasets into a single-element `Vec<ExportTable>` when non-empty.
    #[must_use]
    pub fn from_legacy_datasets(
        datasets_3d: &[(String, Vec<[f64; 3]>)],
        datasets_2d: &[(String, Vec<[f64; 2]>)],
    ) -> Vec<Self> {
        if let Some(t3) = Self::from_datasets_3d(datasets_3d) {
            return vec![t3];
        }
        Self::from_datasets_2d(datasets_2d).into_iter().collect()
    }
}

/// Computes deterministic `(row_stride, col_stride)` step sizes so sampled `(rows, cols)` grids
/// never exceed [`MAX_EXPORT_MATRIX_ENTRIES`] (`65,536`) entries.
#[must_use]
pub fn compute_matrix_stride(rows: usize, cols: usize) -> (usize, usize) {
    if rows == 0 || cols == 0 {
        return (1, 1);
    }
    let mut r_stride = 1usize;
    let mut c_stride = 1usize;
    while rows
        .div_ceil(r_stride)
        .saturating_mul(cols.div_ceil(c_stride))
        > MAX_EXPORT_MATRIX_ENTRIES
    {
        if rows.div_ceil(r_stride) >= cols.div_ceil(c_stride) {
            r_stride += 1;
        } else {
            c_stride += 1;
        }
    }
    (r_stride, c_stride)
}

fn non_empty_tables(tables: &[ExportTable]) -> Vec<&ExportTable> {
    tables
        .iter()
        .filter(|t| !t.headers.is_empty() && !t.rows.is_empty())
        .collect()
}

/// Serializes one or more [`ExportTable`]s into an RFC 4180 CSV string.
///
/// Single-table exports emit `<H1>,<H2>,...\n`. Multi-table exports prepend a `Table` column and
/// merge distinct column headers cleanly across tables.
#[must_use]
pub fn format_tables_csv(tables: &[ExportTable]) -> String {
    let active = non_empty_tables(tables);
    if active.is_empty() {
        return String::new();
    }
    if active.len() == 1 {
        return format_single_table_csv(active[0]);
    }
    format_multi_table_csv(&active)
}

fn format_single_table_csv(table: &ExportTable) -> String {
    let mut csv = table
        .headers
        .iter()
        .map(|h| escape_csv_field(h))
        .collect::<Vec<_>>()
        .join(",");
    csv.push('\n');
    let col_count = table.headers.len();
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

fn format_multi_table_csv(tables: &[&ExportTable]) -> String {
    let mut merged_headers: Vec<String> = Vec::new();
    for table in tables {
        for h in &table.headers {
            if !merged_headers.contains(h) {
                merged_headers.push(h.clone());
            }
        }
    }
    let mut csv = String::from("Table");
    for h in &merged_headers {
        csv.push(',');
        csv.push_str(&escape_csv_field(h));
    }
    csv.push('\n');

    for table in tables {
        let escaped_table = escape_csv_field(&table.name);
        let col_map: Vec<Option<usize>> = merged_headers
            .iter()
            .map(|mh| table.headers.iter().position(|th| th == mh))
            .collect();
        for row in &table.rows {
            csv.push_str(&escaped_table);
            for mapped in &col_map {
                csv.push(',');
                if let Some(cell) = mapped.and_then(|idx| row.get(idx)) {
                    csv.push_str(&cell.to_csv_field());
                }
            }
            csv.push('\n');
        }
    }
    csv
}

/// Serializes one or more [`ExportTable`]s into an RFC 8259 JSON string.
///
/// Single-table exports emit a flat JSON array of row objects; multi-table exports group named
/// tables inside a top-level JSON object keyed by table name.
#[must_use]
pub fn format_tables_json(tables: &[ExportTable]) -> String {
    let active = non_empty_tables(tables);
    if active.is_empty() {
        return "[]".to_string();
    }
    if active.len() == 1 {
        return format_table_rows_json(active[0], "  ");
    }
    let mut out = String::from("{\n");
    let mut used_names: Vec<String> = Vec::with_capacity(active.len());
    for (idx, table) in active.iter().enumerate() {
        if idx > 0 {
            out.push_str(",\n");
        }
        let base = if table.name.trim().is_empty() {
            format!("table_{}", idx + 1)
        } else {
            table.name.clone()
        };
        let unique = if used_names.contains(&base) {
            format!("{base}_{}", idx + 1)
        } else {
            base
        };
        used_names.push(unique.clone());
        let rows_json = format_table_rows_json(table, "    ");
        out.push_str(&format!(
            "  \"{}\": {}",
            escape_json_string(&unique),
            rows_json
        ));
    }
    out.push_str("\n}");
    out
}

fn format_table_rows_json(table: &ExportTable, row_indent: &str) -> String {
    let keys = table.json_keys.as_ref().unwrap_or(&table.headers);
    let escaped_keys: Vec<String> = keys.iter().map(|k| escape_json_string(k)).collect();
    let close_indent = if row_indent.len() >= 2 {
        &row_indent[..row_indent.len() - 2]
    } else {
        ""
    };
    let mut json = String::from("[\n");
    for (r_idx, row) in table.rows.iter().enumerate() {
        if r_idx > 0 {
            json.push_str(",\n");
        }
        json.push_str(row_indent);
        json.push('{');
        for (c_idx, key) in escaped_keys.iter().enumerate() {
            if c_idx > 0 {
                json.push_str(", ");
            }
            let val = row
                .get(c_idx)
                .map_or_else(|| "null".to_string(), ExportCell::to_json_value);
            json.push_str(&format!("\"{key}\": {val}"));
        }
        json.push('}');
    }
    json.push('\n');
    json.push_str(close_indent);
    json.push(']');
    json
}

impl SimulationFramework {
    /// Resolves typed [`ExportTable`]s for the active tool, falling back to cached 3D/2D plot datasets
    /// when thread-local `egui_plot` capture buffers were cleared prior to modal dispatch.
    pub(crate) fn resolve_active_export_tables(&self) -> Vec<ExportTable> {
        let Some(tool) = &self.active_tool else {
            return Vec::new();
        };
        let direct = tool.export_tables();
        if !non_empty_tables(&direct).is_empty() {
            return direct;
        }
        ExportTable::from_legacy_datasets(
            &self.export_datasets_3d_with_fallback(),
            &self.export_datasets_with_fallback(),
        )
    }

    pub(super) fn serialize_active_dataset_export(
        &self,
        tool_name: &str,
        format: ExportFormat,
    ) -> Result<String, String> {
        let tables = self.resolve_active_export_tables();
        if non_empty_tables(&tables).is_empty() {
            return Err(format!(
                "No dataset points available to export from {tool_name}."
            ));
        }
        match format {
            ExportFormat::Csv => Ok(format_tables_csv(&tables)),
            ExportFormat::Json => Ok(format_tables_json(&tables)),
            _ => Err("Unsupported dataset format".to_string()),
        }
    }
}
