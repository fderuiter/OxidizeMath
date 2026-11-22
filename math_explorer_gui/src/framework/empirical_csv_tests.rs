use super::*;
use crate::accessibility::{all_announcements, clear_test_announcements};

#[test]
fn test_parse_valid_csv_tsv_semicolon_and_bom() {
    let csv = "\u{feff}\"time_s\",\"velocity\"\n0.0,1.25\n2.0,3.75\n1.0,2.50\n";
    let table = EmpiricalCsvTable::parse_str("kinetics.csv", csv).expect("valid CSV");
    assert_eq!(table.headers, vec!["time_s", "velocity"]);
    assert_eq!(table.row_count(), 3);
    let (xs, ys) = table.extract_xy_sorted(0, 1).expect("sorted xy");
    assert_eq!(xs, vec![0.0, 1.0, 2.0]);
    assert_eq!(ys, vec![1.25, 2.50, 3.75]);

    let semi = "0.5;10.0\n1.5;20.0\n";
    let no_header = EmpiricalCsvTable::parse_str("semi.csv", semi).expect("valid semicolon CSV");
    assert_eq!(no_header.headers, vec!["Column 1", "Column 2"]);
    assert_eq!(no_header.row_count(), 2);
}

#[test]
fn test_rejects_empty_single_row_malformed_and_oversized_csv() {
    assert!(EmpiricalCsvTable::parse_str("empty.csv", "   \n# comment\n").is_err());
    assert!(EmpiricalCsvTable::parse_str("header_only.csv", "x,y\n").is_err());
    assert!(EmpiricalCsvTable::parse_str("one_row.csv", "x,y\n1.0,2.0\n").is_err());
    assert!(EmpiricalCsvTable::parse_str("one_col.csv", "x\n1.0\n2.0\n").is_err());
    assert!(EmpiricalCsvTable::parse_str("nan.csv", "x,y\n1.0,NaN\n2.0,3.0\n").is_err());
    assert!(EmpiricalCsvTable::parse_str("inf.csv", "x,y\n1.0,inf\n2.0,3.0\n").is_err());
    assert!(EmpiricalCsvTable::parse_str("bad.csv", "x,y\n1.0,abc\n2.0,3.0\n").is_err());
    assert!(EmpiricalCsvTable::parse_str("ragged.csv", "x,y\n1.0,2.0,3.0\n4.0,5.0\n").is_err());

    let oversized_bytes = vec![b'a'; MAX_EMPIRICAL_CSV_BYTES + 1];
    assert!(EmpiricalCsvTable::parse_bytes("huge.csv", &oversized_bytes).is_err());

    let mut too_many_rows = String::from("x,y\n");
    for i in 0..=MAX_EMPIRICAL_CSV_ROWS {
        too_many_rows.push_str(&format!("{i}.0,1.0\n"));
    }
    assert!(EmpiricalCsvTable::parse_str("10001_rows.csv", &too_many_rows).is_err());
}

#[test]
fn test_residuals_csv_formatting_and_assertive_live_region_alert() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    clear_test_announcements();

    let rows = vec![
        ResidualRow {
            x: 0.0,
            y_observed: 1.0,
            y_fitted: 0.9984,
            residual: 0.0016,
        },
        ResidualRow {
            x: 1.0,
            y_observed: 2.5183,
            y_fitted: 2.5141,
            residual: 0.0042,
        },
    ];
    let formatted = format_residuals_csv(&rows).expect("finite residual csv");
    assert_eq!(
        formatted,
        "x,y_observed,y_fitted,residual\n0.0000,1.0000,0.9984,0.0016\n1.0000,2.5183,2.5141,0.0042\n"
    );

    let bad_rows = vec![ResidualRow {
        x: 0.0,
        y_observed: 1.0,
        y_fitted: f64::NAN,
        residual: 0.0,
    }];
    assert!(format_residuals_csv(&bad_rows).is_err());

    let ctx = egui::Context::default();
    let mut state = EmpiricalCsvState::default();
    state.record_error(Some(&ctx), "CSV dataset is empty");
    assert!(state.is_error);
    let anns = all_announcements();
    assert!(
        anns.iter()
            .any(|(msg, prio)| msg.contains("CSV dataset is empty") && prio == "assertive"),
        "Expected assertive live-region alert, got {anns:?}"
    );
}
