use crate::framework::{ExportFormat, ExportStatusSeverity, SimulationFramework};

pub(super) fn count_rfc4180_fields(line: &str) -> usize {
    let mut fields = 1usize;
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    let _ = chars.next();
                } else {
                    in_quotes = false;
                }
            }
            '"' => in_quotes = true,
            ',' if !in_quotes => fields += 1,
            _ => {}
        }
    }
    assert!(!in_quotes, "Unclosed RFC 4180 quote in CSV line: {line}");
    fields
}

pub(super) fn assert_valid_dataset_csv(meta_name: &str, csv: &str) {
    assert!(
        !csv.contains("NaN") && !csv.contains("Infinity"),
        "Non-finite token in CSV for {meta_name}"
    );
    let mut lines = csv.lines().filter(|l| !l.is_empty());
    let header = lines
        .next()
        .unwrap_or_else(|| panic!("Missing CSV header for {meta_name}"));
    let expected_cols = count_rfc4180_fields(header);
    assert!(
        expected_cols >= 2,
        "Expected >= 2 CSV columns for {meta_name}, got header: {header}"
    );
    let mut data_rows = 0usize;
    for (idx, row) in lines.enumerate() {
        let cols = count_rfc4180_fields(row);
        assert_eq!(
            cols, expected_cols,
            "Mismatched RFC 4180 column count on row {idx} for {meta_name}: header='{header}', row='{row}'"
        );
        data_rows += 1;
    }
    assert!(
        data_rows >= 1,
        "Expected >= 1 CSV data row for {meta_name}, got header only: {header}"
    );
}

pub(super) fn assert_valid_dataset_json(meta_name: &str, json: &str) {
    assert!(
        !json.contains("NaN"),
        "Non-finite token in JSON for {meta_name}"
    );
    let parsed: serde_json::Value = serde_json::from_str(json)
        .unwrap_or_else(|e| panic!("Invalid RFC 8259 JSON for {meta_name}: {e}"));
    let non_empty = match &parsed {
        serde_json::Value::Array(arr) => !arr.is_empty(),
        serde_json::Value::Object(obj) => !obj.is_empty(),
        _ => false,
    };
    assert!(
        non_empty,
        "Expected non-empty JSON array or object for {meta_name}"
    );
}

pub(super) fn verify_tool_csv_and_json_exports(
    meta_name: &str,
    framework: &mut SimulationFramework,
    open_modal: bool,
) {
    super::run_deferred_export_frame(framework, ExportFormat::Csv, open_modal);
    assert_eq!(
        framework.export_status_severity,
        ExportStatusSeverity::Success,
        "Deferred CSV failed for {meta_name}: {:?}",
        framework.export_status_message
    );
    let csv = framework
        .export_queue
        .last_exported_content
        .as_deref()
        .unwrap_or_else(|| panic!("Missing CSV content for {meta_name}"));
    assert_valid_dataset_csv(meta_name, csv);

    super::run_deferred_export_frame(framework, ExportFormat::Json, open_modal);
    assert_eq!(
        framework.export_status_severity,
        ExportStatusSeverity::Success,
        "Deferred JSON failed for {meta_name}: {:?}",
        framework.export_status_message
    );
    let json = framework
        .export_queue
        .last_exported_content
        .as_deref()
        .unwrap_or_else(|| panic!("Missing JSON content for {meta_name}"));
    assert_valid_dataset_json(meta_name, json);
}
