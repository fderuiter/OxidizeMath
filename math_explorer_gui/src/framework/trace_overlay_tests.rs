#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_add_snapshot_and_capacity_cap() {
    let mut manager = TraceSnapshotManager::new();
    assert_eq!(manager.len(), 0);
    assert!(manager.is_empty());
    assert!(!manager.is_full());

    for i in 1..=MAX_SNAPSHOTS {
        let success = manager.add_snapshot(
            format!("Baseline {}", i),
            format!("k={:.1}", i as f64),
            vec![TraceSeries::new("y(t)", vec![[0.0, i as f64]])],
        );
        assert!(success);
    }

    assert!(manager.is_full());
    assert!(!manager.add_snapshot(
        "Baseline 11",
        "k=11.0",
        vec![TraceSeries::new("y(t)", vec![[0.0, 11.0]])]
    ));
}

#[test]
fn test_fifo_history_runs_buffer() {
    let mut manager = TraceSnapshotManager::new();
    assert_eq!(manager.history_len(), 0);

    for i in 1..=7 {
        manager.push_history_run(
            format!("k={:.1}", i as f64),
            vec![TraceSeries::new("y(t)", vec![[0.0, i as f64]])],
        );
    }

    assert_eq!(manager.history_len(), MAX_HISTORY_RUNS);
    assert_eq!(manager.history_runs()[0].param_summary, "k=3.0");
    assert_eq!(manager.history_runs()[4].param_summary, "k=7.0");
}

#[test]
fn test_export_datasets_includes_history_and_snapshots() {
    let mut manager = TraceSnapshotManager::new();
    manager.push_history_run("k=1.0", vec![TraceSeries::new("y(t)", vec![[0.0, 1.0]])]);
    manager.add_snapshot(
        "Baseline 1",
        "k=2.0",
        vec![TraceSeries::new("y(t)", vec![[0.0, 2.0]])],
    );

    let exported = manager.export_datasets();
    assert_eq!(exported.len(), 2);
    assert!(exported[0].0.contains("Run 1"));
    assert!(exported[1].0.contains("Baseline 1"));
}

#[test]
fn test_serde_roundtrip() {
    let mut manager = TraceSnapshotManager::new();
    manager.push_history_run("k=1.0", vec![TraceSeries::new("y(t)", vec![[0.0, 1.0]])]);
    manager.add_snapshot(
        "Baseline 1",
        "k=2.0",
        vec![TraceSeries::new("y(t)", vec![[0.0, 2.0]])],
    );

    let json = serde_json::to_string(&manager).expect("Serialization failed");
    let restored: TraceSnapshotManager =
        serde_json::from_str(&json).expect("Deserialization failed");

    assert_eq!(restored.history_len(), 1);
    assert_eq!(restored.len(), 1);
    assert_eq!(restored.history_runs()[0].param_summary, "k=1.0");
    assert_eq!(restored.snapshots()[0].label, "Baseline 1");
}

fn find_trace_button_pos(output: &egui::FullOutput, exact_label: &str) -> egui::Pos2 {
    let ak = output
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit tree must exist");
    let bounds = ak
        .nodes
        .iter()
        .find_map(|(_, node)| {
            if node.role() == egui::accesskit::Role::Button && node.label() == Some(exact_label) {
                node.bounds()
            } else {
                None
            }
        })
        .unwrap_or_else(|| panic!("Trace button '{exact_label}' must exist"));
    egui::pos2(
        ((bounds.x0 + bounds.x1) * 0.5) as f32,
        ((bounds.y0 + bounds.y1) * 0.5) as f32,
    )
}

fn run_trace_frame(
    ctx: &egui::Context,
    manager: &mut TraceSnapshotManager,
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    let screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(900.0, 600.0),
    ));
    ctx.run(
        egui::RawInput {
            screen_rect,
            events,
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                manager.render_ui(ui, || None);
            });
        },
    )
}

fn verify_trace_hover_and_escape(
    ctx: &egui::Context,
    manager: &mut TraceSnapshotManager,
    btn_label: &str,
    expected_tooltip: &str,
) {
    let out = run_trace_frame(
        ctx,
        manager,
        vec![egui::Event::PointerMoved(egui::pos2(890.0, 590.0))],
    );
    let pos = find_trace_button_pos(&out, btn_label);

    crate::accessibility::clear_test_announcements();
    let _ = run_trace_frame(ctx, manager, vec![egui::Event::PointerMoved(pos)]);
    assert!(
        crate::accessibility::all_announcements()
            .iter()
            .any(|(msg, prio)| msg == expected_tooltip && prio == "polite"),
        "Expected announcement '{expected_tooltip}' for '{btn_label}'"
    );

    crate::accessibility::clear_test_announcements();
    let esc = egui::Event::Key {
        key: egui::Key::Escape,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    };
    let _ = run_trace_frame(ctx, manager, vec![egui::Event::PointerMoved(pos), esc]);
    assert!(
        crate::accessibility::all_announcements().is_empty(),
        "Trace overlay tooltip '{expected_tooltip}' must dismiss on Escape"
    );
}

#[test]
fn test_trace_overlay_buttons_announce_and_dismiss_on_escape() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();

    let mut manager = TraceSnapshotManager::new();
    manager.push_history_run("k=1.0", vec![TraceSeries::new("y(t)", vec![[0.0, 1.0]])]);

    for (btn, tooltip) in [
        (
            "Clear All",
            "Remove all history ghost runs and pinned baselines",
        ),
        (
            "📍 Pin Current Snapshot",
            "Save current trajectory curve as a pinned baseline reference",
        ),
        ("👁", "Toggle ghost trace visibility"),
        ("❌", "Delete this history run"),
        ("📍 Pin", "Pin this history run as baseline"),
    ] {
        verify_trace_hover_and_escape(&ctx, &mut manager, btn, tooltip);
    }

    let mut full_manager = TraceSnapshotManager::new();
    for i in 1..=MAX_SNAPSHOTS {
        full_manager.add_snapshot(
            format!("Baseline {i}"),
            "k=1.0",
            vec![TraceSeries::new("y(t)", vec![[0.0, 1.0]])],
        );
    }

    for (btn, tooltip) in [
        (
            "Pin Baseline (Max 10)",
            "Maximum limit of 10 pinned baselines reached.",
        ),
        ("👁", "Toggle baseline visibility"),
        ("❌", "Delete this snapshot"),
    ] {
        verify_trace_hover_and_escape(&ctx, &mut full_manager, btn, tooltip);
    }
}

#[test]
fn test_trace_points_compact_codec_lossless_and_legacy_compat() {
    let mut curve: Vec<[f64; 2]> = (0..128)
        .map(|i| [i as f64 * 0.1, 800.0 - i as f64 * 0.5])
        .collect();
    curve.push([-0.0, f64::INFINITY]);
    curve.push([f64::NEG_INFINITY, f64::NAN]);
    let pts = TracePoints::from(curve.clone());

    let json = serde_json::to_string(&pts).expect("compact serialize");
    assert!(
        json.starts_with("\"tp1:"),
        "Expected compact tp1: string, got {json}"
    );
    let decoded: TracePoints = serde_json::from_str(&json).expect("compact deserialize");
    assert_eq!(decoded.len(), curve.len());
    for (orig, got) in curve.iter().zip(decoded.iter()) {
        assert_eq!(orig[0].to_bits(), got[0].to_bits());
        assert_eq!(orig[1].to_bits(), got[1].to_bits());
    }

    let legacy_json = "[[0.0,1.5],[2.25,-3.75],[4.5,6.0]]";
    let legacy_pts: TracePoints = serde_json::from_str(legacy_json).expect("legacy array");
    assert_eq!(
        legacy_pts.as_ref(),
        &[[0.0, 1.5], [2.25, -3.75], [4.5, 6.0]]
    );
    assert!(serde_json::from_str::<TracePoints>("\"tp1:AAAA\"").is_err());
    assert!(serde_json::from_str::<TracePoints>("\"invalid_prefix\"").is_err());
}
