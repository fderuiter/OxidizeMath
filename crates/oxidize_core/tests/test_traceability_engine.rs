//! Tests for oxidize_core TraceabilityEngine.

use oxidize_core::traceability::TraceabilityEngine;
use oxidize_core::vfs::DefaultVfs;

#[test]
fn test_traceability_engine_scan() {
    let (is_root, _) = oxidize_core::workspace::get_workspace_members();
    let prefix = if is_root { "" } else { "../../" };
    let engine = TraceabilityEngine::new(DefaultVfs);
    let d = format!("{}crates/pure_math/src", prefix);
    let dirs_ref = [d.as_str()];
    let papers_dir = format!("{}papers", prefix);
    let res = futures::executor::block_on(engine.scan_repository(&dirs_ref, &papers_dir, false));
    assert!(res.is_ok(), "Traceability scan of pure_math should succeed");
    let report = res.unwrap();
    assert!(report.scanned_files > 0);
}
