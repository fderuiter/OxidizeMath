//! Unit tests for [`super::export_table`] CSV/JSON serialization, formula sanitization, and matrix stride bounds.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::export_table::{
    compute_matrix_stride, format_tables_csv, format_tables_json, ExportCell, ExportTable,
    MAX_EXPORT_MATRIX_ENTRIES,
};

#[test]
fn test_single_table_csv_and_json_escape_formulas_while_preserving_negative_numbers() {
    let table = ExportTable::new(
        "Metrics",
        vec![
            "Label".into(),
            "Spin".into(),
            "Energy".into(),
            "Active".into(),
        ],
        vec![
            vec![
                ExportCell::Text("=SUM(A1:A2)".into()),
                ExportCell::Int(-1),
                ExportCell::Float(-2.5),
                ExportCell::Bool(true),
            ],
            vec![
                ExportCell::Text("row, \"quoted\"\nnext".into()),
                ExportCell::Int(1),
                ExportCell::Float(f64::NAN),
                ExportCell::Bool(false),
            ],
        ],
    );

    let csv = format_tables_csv(std::slice::from_ref(&table));
    assert!(csv.starts_with("Label,Spin,Energy,Active\n"));
    assert!(
        csv.contains("\"'=SUM(A1:A2)\",-1,-2.5,true\n"),
        "Formula text must be neutralized while negative numbers remain unquoted: {csv}"
    );
    assert!(csv.contains("\"row, \"\"quoted\"\"\nnext\",1,,false\n"));

    let json = format_tables_json(&[table]);
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid RFC 8259 JSON");
    let arr = parsed.as_array().expect("single table serializes as array");
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0]["Label"], "=SUM(A1:A2)");
    assert_eq!(arr[0]["Spin"], -1);
    assert_eq!(arr[0]["Energy"], -2.5);
    assert_eq!(arr[0]["Active"], true);
    assert!(arr[1]["Energy"].is_null());
}

#[test]
fn test_multi_table_csv_prepends_table_column_and_json_groups_by_name() {
    let t1 = ExportTable::new(
        "Nodes",
        vec!["id".into(), "state".into()],
        vec![vec![
            ExportCell::UInt(0),
            ExportCell::Text("Infected".into()),
        ]],
    );
    let t2 = ExportTable::new(
        "Edges",
        vec!["source".into(), "target".into()],
        vec![vec![ExportCell::UInt(0), ExportCell::UInt(1)]],
    );

    let csv = format_tables_csv(&[t1.clone(), t2.clone()]);
    assert!(csv.starts_with("Table,id,state,source,target\n"), "{csv}");
    assert!(csv.contains("Nodes,0,Infected,,\n"), "{csv}");
    assert!(csv.contains("Edges,,,0,1\n"), "{csv}");

    let json = format_tables_json(&[t1, t2]);
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid grouped JSON");
    assert_eq!(parsed["Nodes"][0]["state"], "Infected");
    assert_eq!(parsed["Edges"][0]["target"], 1);
}

#[test]
fn test_legacy_2d_and_3d_tables_match_egui_plot_formatters_byte_for_byte() {
    let ds_2d = vec![("Sine".to_string(), vec![[-1.0, 0.5], [2.0, -0.25]])];
    let tables_2d = ExportTable::from_legacy_datasets(&[], &ds_2d);
    assert_eq!(
        format_tables_csv(&tables_2d),
        egui_plot::export::format_csv(&ds_2d)
    );
    assert_eq!(
        format_tables_json(&tables_2d),
        egui_plot::export::format_json(&ds_2d)
    );

    let ds_3d = vec![("Helix".to_string(), vec![[1.0, -2.0, 3.5]])];
    let tables_3d = ExportTable::from_legacy_datasets(&ds_3d, &ds_2d);
    assert_eq!(
        format_tables_csv(&tables_3d),
        egui_plot::export::format_csv_3d(&ds_3d)
    );
    assert_eq!(
        format_tables_json(&tables_3d),
        egui_plot::export::format_json_3d(&ds_3d)
    );
}

#[test]
fn test_matrix_stride_caps_million_cell_grid_within_65536_entries() {
    let (rs, cs) = compute_matrix_stride(1000, 1000);
    let sampled = 1000usize.div_ceil(rs) * 1000usize.div_ceil(cs);
    assert!(
        sampled > 0 && sampled <= MAX_EXPORT_MATRIX_ENTRIES,
        "1000x1000 sampled count {sampled} exceeded {MAX_EXPORT_MATRIX_ENTRIES}"
    );
    assert_eq!(compute_matrix_stride(256, 256), (1, 1));
}
