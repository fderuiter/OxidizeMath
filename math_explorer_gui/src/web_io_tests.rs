#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use std::sync::mpsc::channel;

#[test]
fn test_infer_mime_type_json_and_mws() {
    assert_eq!(infer_mime_type("analysis_preset.json"), "application/json");
    assert_eq!(infer_mime_type("PRESET.JSON"), "application/json");
    assert_eq!(infer_mime_type("workspace.mws"), "application/json");
    assert_eq!(infer_mime_type("Session_Backup.MWS"), "application/json");
}

#[test]
fn test_infer_mime_type_other_formats() {
    assert_eq!(infer_mime_type("points.csv"), "text/csv");
    assert_eq!(infer_mime_type("plot.svg"), "image/svg+xml");
    assert_eq!(infer_mime_type("figure.png"), "image/png");
    assert_eq!(infer_mime_type("anim.gif"), "image/gif");
    assert_eq!(infer_mime_type("manifold.stl"), "model/stl");
    assert_eq!(infer_mime_type("splats.ply"), "model/ply");
    assert_eq!(infer_mime_type("surface.obj"), "model/obj");
    assert_eq!(infer_mime_type("raw.bin"), "application/octet-stream");
}

#[test]
fn test_sanitize_download_filename_strips_paths_and_falls_back() {
    assert_eq!(
        sanitize_download_filename("workspace.mws", "fallback.mws"),
        "workspace.mws"
    );
    assert_eq!(
        sanitize_download_filename("/tmp/nested/session.mws", "workspace.mws"),
        "session.mws"
    );
    assert_eq!(
        sanitize_download_filename("C:\\fakepath\\preset.json", "fallback.json"),
        "preset.json"
    );
    assert_eq!(
        sanitize_download_filename("   ", "workspace.mws"),
        "workspace.mws"
    );
    assert_eq!(
        sanitize_download_filename("..", "preset.json"),
        "preset.json"
    );
}

#[test]
fn test_platform_instant_monotonic_duration() {
    let earlier = PlatformInstant::now();
    let later = PlatformInstant::now();
    assert!(later.duration_since(earlier) >= Duration::ZERO);
    assert_eq!(earlier.duration_since(later), Duration::ZERO);
    assert!(earlier.elapsed() >= Duration::ZERO);
}

#[test]
fn test_dialog_configs_preserve_metadata() {
    let open_cfg = OpenDialogConfig {
        filter_name: "JSON Preset",
        extensions: &["json"],
        accept: ".json,application/json",
        read_error_prefix: "Failed to read file",
    };
    assert_eq!(open_cfg.extensions, &["json"]);
    assert_eq!(open_cfg.read_error_prefix, "Failed to read file");

    let save_cfg = SaveAsDialogConfig {
        filter_name: "Workspace Session (*.mws, *.json)",
        extensions: &["mws", "json"],
        fallback_filename: "workspace.mws",
        write_error_prefix: "Failed to write workspace file",
    };
    assert_eq!(
        sanitize_download_filename("nested/dir/custom.mws", save_cfg.fallback_filename),
        "custom.mws"
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_spawn_save_direct_file_retains_full_path_and_formats_write_errors() {
    let temp_dir = std::env::temp_dir();
    let target_path = temp_dir.join(format!("web_io_direct_save_{}.mws", std::process::id()));
    let _ = std::fs::remove_file(&target_path);

    let config = DirectSaveConfig {
        fallback_filename: "workspace.mws",
        write_error_prefix: "Failed to write workspace file",
    };
    let (tx, rx) = channel();
    let payload = r#"{"schema_version":"1.0"}"#.to_string();
    spawn_save_direct_file(tx, config, target_path.clone(), payload.clone());

    let saved_path = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("direct save channel should deliver result")
        .expect("direct save to temp dir should succeed");
    assert_eq!(saved_path, target_path);
    assert_eq!(
        std::fs::read_to_string(&target_path).expect("saved file should exist"),
        payload
    );
    let _ = std::fs::remove_file(&target_path);

    let missing_parent = temp_dir
        .join(format!("nonexistent_web_io_dir_{}", std::process::id()))
        .join("session.mws");
    let (err_tx, err_rx) = channel();
    spawn_save_direct_file(err_tx, config, missing_parent, payload);
    let err = err_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("error channel should deliver result")
        .expect_err("writing to missing parent directory should fail");
    assert!(err.starts_with("Failed to write workspace file: "));
}

#[cfg(target_arch = "wasm32")]
#[test]
fn test_spawn_save_direct_file_wasm_typecheck_and_channel_delivery() {
    let config = DirectSaveConfig {
        fallback_filename: "workspace.mws",
        write_error_prefix: "Failed to write workspace file",
    };
    let (tx, rx) = channel();
    spawn_save_direct_file(
        tx,
        config,
        PathBuf::from("browser_session.mws"),
        "{}".to_string(),
    );
    assert!(rx.try_recv().is_ok());
}
