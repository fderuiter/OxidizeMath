use super::{
    download_browser_file, infer_mime_type, spawn_save_as_file_dialog,
    spawn_save_binary_file_dialog, spawn_save_direct_file, DirectSaveConfig, PlatformInstant,
    SaveAsDialogConfig,
};
use oxidize_core::mesh::{
    build_uv_sphere, export_colored_mesh_to_ply_binary, export_mesh_to_stl_binary,
    export_splats_to_ply_binary, ColoredMesh, GaussianSplatPly, Point3D, VertexColor,
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

    let wav_cfg = SaveAsDialogConfig {
        filter_name: "WAV Audio",
        extensions: &["wav"],
        fallback_filename: "sonification.wav",
        write_error_prefix: "Failed to write WAV",
    };
    let (wav_tx, wav_rx) = channel();
    spawn_save_binary_file_dialog(
        wav_tx,
        wav_cfg,
        "clips/lorenz.wav",
        vec![b'R', b'I', b'F', b'F'],
    );
    let wav_saved = wav_rx
        .try_recv()
        .expect("binary browser download should send synchronously on channel")
        .expect("binary browser download should succeed");
    assert_eq!(wav_saved, PathBuf::from("lorenz.wav"));
}

#[wasm_bindgen_test]
fn wasm_download_browser_file_binary_stl_and_ply_blobs() {
    assert_eq!(infer_mime_type("surface_mesh.stl"), "model/stl");
    assert_eq!(infer_mime_type("curvature_mesh.ply"), "model/ply");
    assert_eq!(infer_mime_type("surface_mesh.obj"), "model/obj");

    let sphere = build_uv_sphere(Point3D::new(0.0, 0.0, 0.0), 1.0, 12, 8)
        .expect("UV sphere should build cleanly");
    let stl_bytes = export_mesh_to_stl_binary(&sphere).expect("STL binary should serialize");
    assert_eq!(stl_bytes.len(), 84 + 50 * (sphere.indices.len() / 3));
    download_browser_file("surface_mesh.stl", &stl_bytes)
        .expect("binary STL Blob download should succeed in browser");

    let colors = vec![VertexColor::new(40, 180, 240); sphere.vertices.len()];
    let colored = ColoredMesh::new(sphere, colors);
    let ply_bytes =
        export_colored_mesh_to_ply_binary(&colored).expect("colored PLY binary should serialize");
    assert!(ply_bytes.starts_with(b"ply\nformat binary_little_endian 1.0\n"));
    download_browser_file("curvature_mesh.ply", &ply_bytes)
        .expect("colored PLY Blob download should succeed in browser");

    let splat = GaussianSplatPly {
        mean: Point3D::new(0.0, 0.0, 0.0),
        scale: Point3D::new(0.4, 0.3, 0.2),
        rotation: [1.0, 0.0, 0.0, 0.0],
        opacity: 0.85,
        color: [0.9, 0.4, 0.2],
    };
    let splat_bytes =
        export_splats_to_ply_binary(&[splat]).expect("3DGS PLY binary should serialize");
    download_browser_file("gaussian_splatting_mesh.ply", &splat_bytes)
        .expect("3DGS PLY Blob download should succeed in browser");
}
