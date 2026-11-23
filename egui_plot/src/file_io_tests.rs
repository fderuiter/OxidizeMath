use super::*;

#[test]
fn test_infer_mime_type_all_supported_extensions() {
    assert_eq!(infer_mime_type("session.mws"), "application/json");
    assert_eq!(infer_mime_type("BACKUP.MWS"), "application/json");
    assert_eq!(infer_mime_type("preset.json"), "application/json");
    assert_eq!(infer_mime_type("data.csv"), "text/csv");
    assert_eq!(infer_mime_type("table.tsv"), "text/tab-separated-values");
    assert_eq!(infer_mime_type("notes.txt"), "text/plain");
    assert_eq!(infer_mime_type("mandelbrot_webgl2.frag"), "text/plain");
    assert_eq!(infer_mime_type(" DOMAIN_COLOR.FRAG "), "text/plain");
    assert_eq!(infer_mime_type("potential_flow.glsl"), "text/plain");
    assert_eq!(infer_mime_type("KERNEL.GLSL"), "text/plain");
    assert_eq!(infer_mime_type("chart.png"), "image/png");
    assert_eq!(infer_mime_type("chart.svg"), "image/svg+xml");
    assert_eq!(infer_mime_type("anim.gif"), "image/gif");
    assert_eq!(infer_mime_type("mesh.obj"), "application/octet-stream");
}

#[test]
fn test_sanitize_download_filename_paths_and_edge_cases() {
    assert_eq!(
        sanitize_download_filename("/var/tmp/sir_data.csv", "fallback.csv"),
        "sir_data.csv"
    );
    assert_eq!(
        sanitize_download_filename("/tmp/shaders/fractal_webgl2.frag", "shader.frag"),
        "fractal_webgl2.frag"
    );
    assert_eq!(
        sanitize_download_filename("C:\\shaders\\conformal\0\r\n.glsl", "shader.glsl"),
        "conformal.glsl"
    );
    assert_eq!(
        sanitize_download_filename("C:\\fakepath\\sir_data.csv", "fallback.csv"),
        "sir_data.csv"
    );
    assert_eq!(
        sanitize_download_filename("report\0\r\n.json", "fallback.json"),
        "report.json"
    );
    assert_eq!(
        sanitize_download_filename(" .. ", "fallback.csv"),
        "fallback.csv"
    );
    assert_eq!(
        sanitize_download_filename("C:\\", "C:\\nested\\default.mws"),
        "default.mws"
    );
    assert_eq!(sanitize_download_filename("   ", " .. "), "download.bin");
}

#[test]
fn test_strip_utf8_bom() {
    assert_eq!(strip_utf8_bom("\u{feff}X,Y\n1,2"), "X,Y\n1,2");
    assert_eq!(strip_utf8_bom("X,Y\n1,2"), "X,Y\n1,2");
    assert_eq!(strip_utf8_bom(""), "");
}

#[test]
fn test_process_plot_csv_import_valid_invalid_and_multi_plot_isolation() {
    let ctx = egui::Context::default();
    let plot_a = egui::Id::new("test_csv_import_plot_a");
    let plot_b = egui::Id::new("test_csv_import_plot_b");

    process_plot_csv_import(
        &ctx,
        plot_a,
        Ok((
            "empirical.csv".to_string(),
            "\u{feff}X,Infected\n0.0,10.0\n1.0,25.0\n".to_string(),
        )),
    );
    process_plot_csv_import(
        &ctx,
        plot_b,
        Ok((
            "invalid.csv".to_string(),
            "HeaderOnly,NoNumbers\nfoo,bar\n".to_string(),
        )),
    );

    // Draining plot_b first must not consume plot_a's pending datasets or status
    assert!(drain_imported_datasets(plot_b).is_empty());
    let status_b = drain_import_status(plot_b).expect("plot_b should have error status");
    assert!(status_b.1);
    assert!(status_b.0.contains("Invalid or empty CSV file"));

    let imported_a = drain_imported_datasets(plot_a);
    assert_eq!(imported_a.len(), 1);
    assert_eq!(imported_a[0].name, "Infected");
    assert_eq!(imported_a[0].points, vec![[0.0, 10.0], [1.0, 25.0]]);
    let status_a = drain_import_status(plot_a).expect("plot_a should have success status");
    assert!(!status_a.1);
    assert_eq!(status_a.0, "Imported 1 dataset(s) from empirical.csv");

    // FileReader error forwarding and ARIA assertive message
    process_plot_csv_import(
        &ctx,
        plot_a,
        Err("Browser failed to read selected file".to_string()),
    );
    let err_status = drain_import_status(plot_a).expect("plot_a should receive read error");
    assert_eq!(
        err_status,
        ("Browser failed to read selected file".to_string(), true)
    );
    let aria_err = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_assertive_message")));
    assert_eq!(
        aria_err.as_deref(),
        Some("Browser failed to read selected file")
    );
}

#[test]
fn test_bounded_plot_import_queue_evicts_oldest_unrendered_entries() {
    let ctx = egui::Context::default();
    let target_plot = egui::Id::new("bounded_target_plot");

    for idx in 0..(MAX_PENDING_PLOT_IMPORTS + 8) {
        let stale_id = egui::Id::new(("stale_unrendered_plot", idx));
        process_plot_csv_import(
            &ctx,
            stale_id,
            Ok(("stale.csv".to_string(), "0.0,1.0\n1.0,2.0\n".to_string())),
        );
    }

    let queue_len = pending_imports_queue()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .len();
    assert!(queue_len <= MAX_PENDING_PLOT_IMPORTS);

    process_plot_csv_import(
        &ctx,
        target_plot,
        Ok(("fresh.csv".to_string(), "0.0,3.0\n1.0,6.0\n".to_string())),
    );
    let datasets = drain_imported_datasets(target_plot);
    assert_eq!(datasets.len(), 1);
    assert_eq!(datasets[0].points, vec![[0.0, 3.0], [1.0, 6.0]]);
    let _ = drain_import_status(target_plot);

    // Clear remaining synthetic stale entries
    pending_imports_queue()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clear();
    pending_status_queue()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clear();
}
