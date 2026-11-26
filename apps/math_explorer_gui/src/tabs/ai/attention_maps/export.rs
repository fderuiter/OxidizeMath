#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{ExportCell, ExportTable};
use nalgebra::DMatrix;

use super::AttentionMapsTool;

fn matrix_to_table(name: &str, matrix: &DMatrix<f64>) -> ExportTable {
    let mut headers = Vec::with_capacity(matrix.ncols() + 1);
    headers.push("row".to_string());
    for col in 0..matrix.ncols() {
        headers.push(format!("col_{col}"));
    }
    let mut rows = Vec::with_capacity(matrix.nrows());
    for r in 0..matrix.nrows() {
        let mut row = Vec::with_capacity(matrix.ncols() + 1);
        row.push(ExportCell::Int(r as i64));
        for c in 0..matrix.ncols() {
            row.push(ExportCell::Float(matrix[(r, c)]));
        }
        rows.push(row);
    }
    ExportTable::new(name, headers, rows)
}

pub(super) fn build_attention_tables(tool: &AttentionMapsTool) -> Vec<ExportTable> {
    vec![
        matrix_to_table("Queries (Q)", &tool.q_matrix),
        matrix_to_table("Keys (K)", &tool.k_matrix),
        matrix_to_table("Values (V)", &tool.v_matrix),
        matrix_to_table("Attention Weights", &tool.attention_weights),
        matrix_to_table("Output", &tool.output_matrix),
    ]
}
