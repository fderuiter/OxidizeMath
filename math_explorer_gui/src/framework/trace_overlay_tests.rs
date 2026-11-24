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
fn test_legacy_snapshot_deserialization_and_deny_unknown_fields() {
    let legacy_json = r#"{
        "snapshots": [{
            "id": 1,
            "label": "Baseline 1",
            "param_summary": "dose=100",
            "series": [{"name": "C(t)", "points": [[0.0, 0.0], [1.0, 2.5]]}],
            "visible": true,
            "color_rgb": [255, 127, 14]
        }],
        "history_runs": [],
        "next_id": 2,
        "history_counter": 0
    }"#;
    let restored: TraceSnapshotManager =
        serde_json::from_str(legacy_json).expect("legacy JSON without kpis must deserialize");
    assert_eq!(restored.len(), 1);
    assert!(restored.snapshots()[0].kpis.is_empty());

    let unknown_snapshot_field = r#"{
        "id": 1,
        "label": "Bad",
        "param_summary": "",
        "series": [],
        "visible": true,
        "color_rgb": [0, 0, 0],
        "unexpected": 123
    }"#;
    assert!(serde_json::from_str::<TraceSnapshot>(unknown_snapshot_field).is_err());

    let unknown_kpi_field = r#"{"name": "AUC", "value": 12.0, "extra": true}"#;
    assert!(serde_json::from_str::<ScenarioKpi>(unknown_kpi_field).is_err());
}

#[test]
fn test_kpi_comparison_rows_and_csv_export() {
    let mut manager = TraceSnapshotManager::new();
    manager.add_snapshot_with_kpis(
        "Baseline 1",
        "dose=100, ke=0.1",
        vec![TraceSeries::new("C(t)", vec![[0.0, 10.0]])],
        vec![
            ScenarioKpi::new("AUC", 100.0, "mg*h/L"),
            ScenarioKpi::new("Residual", 0.0, ""),
        ],
    );
    manager.add_snapshot_with_kpis(
        "High, Dose",
        "dose=150",
        vec![TraceSeries::new("C(t)", vec![[0.0, 15.0]])],
        vec![
            ScenarioKpi::new("AUC", 150.0, "mg*h/L"),
            ScenarioKpi::new("Residual", 0.25, ""),
        ],
    );
    let live = vec![ScenarioKpi::new("AUC", 80.0, "mg*h/L")];
    let rows = manager.kpi_comparison_rows(&live);
    assert_eq!(rows.len(), 5);
    assert!(rows[0].is_reference);
    assert_eq!(rows[0].abs_delta, 0.0);
    assert_eq!(rows[0].pct_delta, Some(0.0));

    let high_auc = rows
        .iter()
        .find(|r| r.scenario_label.starts_with("High, Dose") && r.kpi_name == "AUC")
        .expect("High Dose AUC row");
    assert!((high_auc.abs_delta - 50.0).abs() < 1e-9);
    assert_eq!(high_auc.pct_delta, Some(50.0));

    let zero_base = rows
        .iter()
        .find(|r| r.scenario_label.starts_with("High, Dose") && r.kpi_name == "Residual")
        .expect("zero baseline row");
    assert!((zero_base.abs_delta - 0.25).abs() < 1e-9);
    assert_eq!(zero_base.pct_delta, None);

    let live_auc = rows
        .iter()
        .find(|r| r.scenario_label == "Current" && r.kpi_name == "AUC")
        .expect("Current AUC row");
    assert!((live_auc.abs_delta + 20.0).abs() < 1e-9);
    assert_eq!(live_auc.pct_delta, Some(-20.0));

    let csv = manager.export_kpi_csv(&live);
    assert!(csv.starts_with("scenario,kpi,unit,value,baseline,abs_delta,pct_delta\n"));
    assert!(
        csv.contains("\"High, Dose: dose=150\",AUC,mg*h/L,150.000000,100.000000,50.000000,50.0000")
    );
    assert!(csv.contains("Current,AUC,mg*h/L,80.000000,100.000000,-20.000000,-20.0000"));

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    verify_trace_hover_and_escape(
        &ctx,
        &mut manager,
        "⬇ Export KPI CSV",
        "Export scenario KPI comparison table to CSV",
    );
}
