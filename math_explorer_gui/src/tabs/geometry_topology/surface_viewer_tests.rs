#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use std::sync::mpsc;

#[test]
fn test_surface_viewer_non_blocking_obj_export_polling_and_live_announcements() {
    crate::accessibility::clear_test_announcements();
    let mut viewer = SurfaceViewer::default();
    let obj = viewer
        .build_obj_mesh_string()
        .expect("OBJ mesh generation should succeed");
    assert!(obj.contains("\nv ") || obj.starts_with("v ") || obj.contains("\nf "));

    let (tx, rx) = mpsc::channel();
    viewer.obj_export_rx = Some(rx);
    let ctx = egui::Context::default();

    // Frame while dialog/write is still in flight does not block and keeps channel active
    let _ = ctx.run(egui::RawInput::default(), |sub_ctx| viewer.show(sub_ctx));
    assert!(viewer.obj_export_rx.is_some());
    assert!(viewer.obj_export_status.is_none());

    // Deliver asynchronous save completion and verify polite ARIA announcement
    tx.send(Ok("torus_mesh.obj".to_string()))
        .expect("send result");
    let _ = ctx.run(egui::RawInput::default(), |sub_ctx| viewer.show(sub_ctx));
    assert!(viewer.obj_export_rx.is_none());
    assert_eq!(
        viewer.obj_export_status,
        Some(("Saved OBJ mesh to torus_mesh.obj".to_string(), false))
    );
    let live_msg = ctx
        .data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")))
        .expect("Polite live status expected");
    assert!(live_msg.contains("torus_mesh.obj"));

    // Deliver asynchronous save failure and verify assertive ARIA alert
    let (tx_err, rx_err) = mpsc::channel();
    viewer.obj_export_rx = Some(rx_err);
    tx_err
        .send(Err("Failed to save OBJ mesh: permission denied".to_string()))
        .expect("send error");
    let _ = ctx.run(egui::RawInput::default(), |sub_ctx| viewer.show(sub_ctx));
    assert_eq!(
        viewer.obj_export_status,
        Some((
            "Failed to save OBJ mesh: permission denied".to_string(),
            true
        ))
    );
    let alert_msg = ctx
        .data(|d| d.get_temp::<String>(egui::Id::new("aria_live_assertive_message")))
        .expect("Assertive live alert expected");
    assert!(alert_msg.contains("permission denied"));
}

#[test]
fn test_surface_viewer_contains_no_blocking_rfd_or_wasm_vfs_calls() {
    let main_src = include_str!("surface_viewer.rs");
    let ui_src = include_str!("surface_viewer_ui.rs");
    for src in [main_src, ui_src] {
        let banned_rfd = ["rfd::", "FileDialog"].concat();
        let banned_vfs = ["Wasm", "Vfs"].concat();
        assert!(!src.contains(&banned_rfd));
        assert!(!src.contains(&banned_vfs));
    }
}
