#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::accessibility::{all_announcements, clear_test_announcements};
use crate::framework::{ExportFormat, ExportStatusSeverity, ExportTarget, SimulationFramework};

#[test]
fn test_surface_viewer_deferred_obj_stl_ply_exports_and_live_announcements() {
    clear_test_announcements();
    let viewer = SurfaceViewer::default();
    let obj = viewer.build_obj_string();
    assert!(obj.contains("\nv ") || obj.starts_with("v ") || obj.contains("\nf "));

    let mut fw = SimulationFramework::new("__surface_viewer_export__");
    fw.active_tool = Some(Box::new(viewer));
    let ctx = egui::Context::default();

    let _ = ctx.run(egui::RawInput::default(), |sub_ctx| {
        fw.queue_export(ExportFormat::Obj);
        fw.show(sub_ctx, "surface_obj_audit");
    });
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    assert!(all_announcements().iter().any(|(msg, prio)| {
        msg.contains("Wavefront OBJ Mesh saved successfully") && prio == "polite"
    }));

    clear_test_announcements();
    let _ = ctx.run(egui::RawInput::default(), |sub_ctx| {
        fw.export_queue
            .push_target(ExportFormat::Obj, ExportTarget::Clipboard);
        fw.show(sub_ctx, "surface_obj_clipboard_audit");
    });
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Warning);
    assert!(all_announcements()
        .iter()
        .any(|(_, prio)| prio == "assertive"));
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
