use super::{
    spawn_save_as_file_dialog, spawn_save_direct_file, DirectSaveConfig, PlatformInstant,
    SaveAsDialogConfig,
};
use std::path::PathBuf;
use std::sync::mpsc::channel;
use std::time::Duration;
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn wasm_platform_instant_measures_non_negative_browser_elapsed_time() {
    let start = PlatformInstant::now();
    let mut spin = 0_u32;
    for idx in 0..2_000 {
        spin = spin.wrapping_add(idx);
    }
    assert!(spin > 0);
    let later = PlatformInstant::now();
    let delta = later.duration_since(start);
    assert!(delta < Duration::from_secs(30));
    assert_eq!(start.duration_since(later), Duration::ZERO);
    assert!(start.elapsed() < Duration::from_secs(30));
}

#[wasm_bindgen_test]
fn wasm_spawn_save_dialogs_trigger_browser_download_and_deliver_paths() {
    let direct_cfg = DirectSaveConfig {
        fallback_filename: "workspace.mws",
        write_error_prefix: "Failed to write workspace file",
    };
    let (direct_tx, direct_rx) = channel();
    spawn_save_direct_file(
        direct_tx,
        direct_cfg,
        PathBuf::from("sub/folder/session_export.mws"),
        r#"{"schema_version":"1.0"}"#.to_string(),
    );
    let saved = direct_rx
        .try_recv()
        .expect("direct browser download should send synchronously on channel")
        .expect("direct browser download should succeed");
    assert_eq!(saved, PathBuf::from("session_export.mws"));

    let save_as_cfg = SaveAsDialogConfig {
        filter_name: "JSON Preset",
        extensions: &["json"],
        fallback_filename: "preset.json",
        write_error_prefix: "Failed to write preset",
    };
    let (save_tx, save_rx) = channel();
    spawn_save_as_file_dialog(
        save_tx,
        save_as_cfg,
        "nested/custom_preset.json",
        r#"{"preset":1}"#.to_string(),
        |path, leaf| (path, leaf.to_string()),
    );
    let (mapped_path, leaf_name) = save_rx
        .try_recv()
        .expect("save-as browser download should send synchronously on channel")
        .expect("save-as browser download should succeed");
    assert_eq!(mapped_path, PathBuf::from("custom_preset.json"));
    assert_eq!(leaf_name, "custom_preset.json");
}
