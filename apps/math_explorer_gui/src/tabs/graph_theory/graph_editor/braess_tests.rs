#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::tabs::graph_theory::graph_editor::GraphEditorTool;

#[test]
fn test_braess_paradox_latency_jump_150_to_200_and_nash_trap_callout() {
    let mut tool = GraphEditorTool::default();
    tool.load_braess_preset(false);

    let summary_without = tool.braess.status.expect("Wardrop without shortcut");
    assert!(
        (summary_without.commuter_latency - 1.50).abs() < 1e-6,
        "Expected 1.50 latency without u->v shortcut, got {}",
        summary_without.commuter_latency
    );
    assert!((summary_without.beckmann_potential - 1.25).abs() < 1e-6);

    tool.set_braess_shortcut(true);
    let summary_with = tool.braess.status.expect("Wardrop with shortcut");
    assert!(
        (summary_with.commuter_latency - 2.00).abs() < 1e-6,
        "Expected 2.00 latency with zero-cost u->v shortcut, got {}",
        summary_with.commuter_latency
    );
    assert!((summary_with.beckmann_potential - 1.00).abs() < 1e-6);
    assert_eq!(tool.braess.without_shortcut_latency, Some(1.50));

    let trap_callout = tool
        .braess
        .callout_manager
        .callouts()
        .iter()
        .find(|c| c.tag.as_deref() == Some("nash_trap"))
        .expect("nash_trap callout must be set");
    assert!(trap_callout.text.contains("Nash Trap"));
    assert!(trap_callout.text.contains("1.50 -> 2.00"));
}

#[test]
fn test_frank_wolfe_zero_heap_allocations_under_verified_allocator() {
    let mut tool = GraphEditorTool::default();
    tool.load_braess_preset(true);

    let mut flows = vec![0.0; tool.edges.len()];
    let mut lats = vec![0.0; tool.edges.len()];
    let _ = solve_wardrop_in_place(
        &mut tool.braess.scratch,
        0,
        3,
        1.0,
        None,
        &mut flows,
        &mut lats,
    )
    .unwrap();

    let summary = {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (res, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            solve_wardrop_in_place(
                &mut tool.braess.scratch,
                0,
                3,
                1.0,
                None,
                &mut flows,
                &mut lats,
            )
        });
        res.unwrap()
    };
    assert!((summary.commuter_latency - 2.00).abs() < 1e-6);
}

#[test]
fn test_unreachable_pair_and_negative_latency_clamping() {
    let clamped = AffineLatency::clamped(-3.5, -1.2);
    assert_eq!(clamped, AffineLatency { a: 0.0, b: 0.0 });

    let mut tool = GraphEditorTool::default();
    tool.load_braess_preset(false);
    // Remove all outgoing edges from source node 0 so sink 3 becomes unreachable.
    tool.edges.retain(|&(u, _, _)| u != 0);
    tool.braess.edge_latencies.truncate(tool.edges.len());
    tool.recompute_braess();
    assert_eq!(tool.braess.status, Err(WardropError::Unreachable));
}

// [cite:dijkstra]
