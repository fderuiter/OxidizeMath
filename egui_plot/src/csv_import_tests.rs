use super::{
    CsvImportError, MAX_CSV_IMPORT_BYTES, MAX_CSV_IMPORT_ROWS, MAX_CSV_IMPORT_SERIES, parse_csv,
    parse_csv_bounded, split_line, unescape_formula_label,
};
use crate::data_table::format::{TableSeriesRef, format_tsv, sanitize_tsv_label};
use crate::export::{escape_csv_field, format_csv};
use egui::Color32;

#[test]
fn probe_01_standard_two_column_with_header() {
    let csv = "Time,Velocity\n0.0,1.5\n1.0,3.0\n2.0,4.5\n";
    let ds = parse_csv(csv);
    assert_eq!(ds.len(), 1);
    assert_eq!(ds[0].name, "Velocity");
    assert_eq!(ds[0].points, vec![[0.0, 1.5], [1.0, 3.0], [2.0, 4.5]]);
    assert!(ds[0].show_points);
    assert!(!ds[0].show_lines);
}

#[test]
fn probe_02_headerless_two_column() {
    let csv = "0.0, 10.0\n1.0, 20.0\n2.0, 30.0\n";
    let ds = parse_csv(csv);
    assert_eq!(ds.len(), 1);
    assert_eq!(ds[0].name, "Imported Dataset");
    assert_eq!(ds[0].points, vec![[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]]);
}

#[test]
fn probe_03_three_column_repeated_series_labels_grouped() {
    let csv = "Dataset,X,Y\nTrial A,0.0,1.0\nTrial A,1.0,2.0\nTrial B,0.0,5.0\nTrial B,1.0,6.0\n";
    let ds = parse_csv(csv);
    assert_eq!(ds.len(), 2);
    assert_eq!(ds[0].name, "Trial A");
    assert_eq!(ds[0].points, vec![[0.0, 1.0], [1.0, 2.0]]);
    assert_eq!(ds[1].name, "Trial B");
    assert_eq!(ds[1].points, vec![[0.0, 5.0], [1.0, 6.0]]);
}

#[test]
fn probe_04_three_column_id_x_y_stays_single_curve() {
    let numeric_ids = "id,x,y\n1,0.0,10.0\n2,1.0,20.0\n3,2.0,30.0\n";
    let ds_num = parse_csv(numeric_ids);
    assert_eq!(ds_num.len(), 1);
    assert_eq!(
        ds_num[0].points,
        vec![[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]]
    );

    let string_ids = "sample_name,x,y\ns_01,0.0,2.5\ns_02,1.0,5.0\ns_03,2.0,7.5\n";
    let ds_str = parse_csv(string_ids);
    assert_eq!(ds_str.len(), 1);
    assert_eq!(ds_str[0].points, vec![[0.0, 2.5], [1.0, 5.0], [2.0, 7.5]]);
}

#[test]
fn probe_05_wide_multi_series_with_headers() {
    let csv = "t,Susceptible,Infected,Recovered\n0.0,990,10,0\n1.0,950,45,5\n2.0,880,95,25\n";
    let ds = parse_csv(csv);
    assert_eq!(ds.len(), 3);
    assert_eq!(ds[0].name, "Susceptible");
    assert_eq!(ds[1].name, "Infected");
    assert_eq!(ds[2].name, "Recovered");
    assert_eq!(ds[1].points, vec![[0.0, 10.0], [1.0, 45.0], [2.0, 95.0]]);
}

#[test]
fn probe_06_headerless_wide_table_preserves_columns_3_plus() {
    let csv = "0.0, 1.0, 2.0, 3.0\n1.0, 4.0, 5.0, 6.0\n2.0, 7.0, 8.0, 9.0\n";
    let ds = parse_csv(csv);
    assert_eq!(ds.len(), 3);
    assert_eq!(ds[0].name, "Series 1");
    assert_eq!(ds[0].points, vec![[0.0, 1.0], [1.0, 4.0], [2.0, 7.0]]);
    assert_eq!(ds[1].name, "Series 2");
    assert_eq!(ds[1].points, vec![[0.0, 2.0], [1.0, 5.0], [2.0, 8.0]]);
    assert_eq!(ds[2].name, "Series 3");
    assert_eq!(ds[2].points, vec![[0.0, 3.0], [1.0, 6.0], [2.0, 9.0]]);
}

#[test]
fn probe_07_tsv_spreadsheet_and_data_table_roundtrip() {
    let pts_a = [[0.0, 1.0], [1.0, 2.0]];
    let pts_b = [[0.0, 3.0], [1.0, 4.0]];
    let refs = [
        TableSeriesRef {
            name: "-2.0V",
            points: &pts_a,
            color: Color32::RED,
        },
        TableSeriesRef {
            name: "+2.0V",
            points: &pts_b,
            color: Color32::BLUE,
        },
    ];
    let tsv = format_tsv(&refs);
    assert!(tsv.contains("'-2.0V") && tsv.contains("'+2.0V"));

    let ds = parse_csv(&tsv);
    assert_eq!(ds.len(), 2);
    assert_eq!(ds[0].name, "-2.0V");
    assert_eq!(ds[1].name, "+2.0V");
    assert_eq!(ds[0].points, pts_a);
    assert_eq!(ds[1].points, pts_b);
}

#[test]
fn probe_08_semicolon_delimited_decimal_commas() {
    let csv = "Zeit;Spannung;Strom\n0,0;1,5;-0,25\n0,5;2,75;0,125\n1,0;4,0;0,5\n";
    let ds = parse_csv(csv);
    assert_eq!(ds.len(), 2);
    assert_eq!(ds[0].name, "Spannung");
    assert_eq!(ds[0].points, vec![[0.0, 1.5], [0.5, 2.75], [1.0, 4.0]]);
    assert_eq!(ds[1].name, "Strom");
    assert_eq!(ds[1].points, vec![[0.0, -0.25], [0.5, 0.125], [1.0, 0.5]]);
}

#[test]
fn probe_09_tab_delimited_decimal_commas() {
    let tsv = "X\tMessung\n0,0\t-1,25\n1,5\t3,75\n";
    let ds = parse_csv(tsv);
    assert_eq!(ds.len(), 1);
    assert_eq!(ds[0].name, "Messung");
    assert_eq!(ds[0].points, vec![[0.0, -1.25], [1.5, 3.75]]);
}

#[test]
fn probe_10_signed_and_formula_labels_unescape_while_exports_stay_escaped() {
    assert_eq!(escape_csv_field("-2.0V"), "\"'-2.0V\"");
    assert_eq!(sanitize_tsv_label("-2.0V"), "'-2.0V");
    assert_eq!(unescape_formula_label("'-2.0V"), "-2.0V");
    assert_eq!(unescape_formula_label("'+1.0V"), "+1.0V");
    assert_eq!(unescape_formula_label("'=SUM(A1)"), "=SUM(A1)");
    assert_eq!(unescape_formula_label("'@probe"), "@probe");
    assert_eq!(unescape_formula_label("'ordinary"), "'ordinary");

    let exported = format_csv(&[
        ("-2.0V".to_owned(), vec![[0.0, -2.0], [1.0, -1.8]]),
        ("+5.0V".to_owned(), vec![[0.0, 5.0], [1.0, 4.9]]),
    ]);
    assert!(exported.contains("\"'-2.0V\""));
    let reimported = parse_csv(&exported);
    assert_eq!(reimported.len(), 2);
    assert_eq!(reimported[0].name, "-2.0V");
    assert_eq!(reimported[1].name, "+5.0V");
}

#[test]
fn probe_11_bom_comments_nan_inf_and_missing_cells_filtered() {
    let csv = "\u{feff}# Sensor calibration log\n\nX,Signal\n0.0,1.0\n1.0,NaN\n2.0,inf\n3.0,\n# mid-file comment\n4.0,5.0\n";
    let ds = parse_csv(csv);
    assert_eq!(ds.len(), 1);
    assert_eq!(ds[0].name, "Signal");
    assert_eq!(ds[0].points, vec![[0.0, 1.0], [4.0, 5.0]]);
}

#[test]
fn probe_12_rfc4180_quotes_and_untrusted_input_caps() {
    let quoted = "Dataset,X,Y\n\"Series; A, 1\",0.0,1.0\n\"Series; A, 1\",1.0,2.0\n\"\"\"Quoted\"\"\",0.0,3.0\n";
    let ds_q = parse_csv(quoted);
    assert_eq!(ds_q.len(), 2);
    assert_eq!(ds_q[0].name, "Series; A, 1");
    assert_eq!(ds_q[1].name, "\"Quoted\"");
    assert_eq!(split_line("\"a;b\",1", ','), vec!["a;b", "1"]);

    let oversized = "0,1\n".repeat((MAX_CSV_IMPORT_BYTES / 4) + 2);
    assert!(oversized.len() > MAX_CSV_IMPORT_BYTES);
    assert_eq!(
        parse_csv_bounded(&oversized),
        Err(CsvImportError::PayloadTooLarge {
            byte_len: oversized.len(),
            max_bytes: MAX_CSV_IMPORT_BYTES,
        })
    );
    assert!(parse_csv(&oversized).is_empty());

    let mut many_rows = String::from("x,y\n");
    for i in 0..(MAX_CSV_IMPORT_ROWS + 50) {
        many_rows.push_str(&format!("{i},{i}\n"));
    }
    let row_outcome = parse_csv_bounded(&many_rows).expect("bounded rows");
    assert_eq!(row_outcome.rows_imported, MAX_CSV_IMPORT_ROWS);
    assert!(row_outcome.truncated_rows);
    assert_eq!(row_outcome.datasets[0].points.len(), MAX_CSV_IMPORT_ROWS);

    let mut wide_hdr = String::from("x");
    let mut wide_row = String::from("1.0");
    for s in 1..=(MAX_CSV_IMPORT_SERIES + 10) {
        wide_hdr.push_str(&format!(",s{s}"));
        wide_row.push_str(&format!(",{s}.0"));
    }
    let wide_csv = format!("{wide_hdr}\n{wide_row}\n");
    let wide_outcome = parse_csv_bounded(&wide_csv).expect("bounded series");
    assert_eq!(wide_outcome.datasets.len(), MAX_CSV_IMPORT_SERIES);
    assert!(wide_outcome.truncated_series);
}
