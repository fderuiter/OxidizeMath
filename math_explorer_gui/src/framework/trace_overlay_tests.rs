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
