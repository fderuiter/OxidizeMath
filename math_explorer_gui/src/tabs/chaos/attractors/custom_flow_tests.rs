#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::async_sim::SimulationRunner;
use crate::framework::InteractiveTool;
use std::collections::BTreeMap;

#[test]
fn test_custom_3d_flow_worker_streaming_zero_alloc_and_roundtrip() {
    let mut plotter = AttractorPlotter {
        mode: AttractorMode::Custom,
        ..Default::default()
    };
    plotter.flow_bank.set_expressions(&[
        "sin(y) - b * x",
        "sin(z) - b * y",
        "sin(x) - b * z",
    ]);
    let mut pmap = BTreeMap::new();
    pmap.insert("b".to_string(), 0.19);
    plotter.flow_bank.set_params_map(&pmap);
    plotter.on_slider_changed();

    // Verify zero heap allocations inside CompiledFlowJob::rk4_step
    let job = CompiledFlowJob::from_bank(&plotter.flow_bank);
    {
        let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (state, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            let mut s = Vector3::new(1.0, 0.5, -0.5);
            for _ in 0..50 {
                s = job.rk4_step(s, 0.02);
            }
            s
        });
        assert!(state.x.is_finite() && state.y.is_finite() && state.z.is_finite());
    }

    // Stream via AttractorRunner sharing Arc<CompiledFlowJob>
    let mut runner = AttractorRunner::from_params(Arc::clone(&plotter.params));
    runner.step();
    assert_eq!(runner.execution_state, SolverExecutionState::Running);
    assert_eq!(runner.history.len(), plotter.simulation_speed);
    assert!(runner
        .history
        .iter()
        .all(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite()));

    // Verify UI render hides Lorenz fixed-point callouts in custom mode without stalling
    let ctx = eframe::egui::Context::default();
    let out = ctx.run(eframe::egui::RawInput::default(), |ctx| plotter.show(ctx));
    for clipped in &out.shapes {
        if let eframe::egui::epaint::Shape::Text(t) = &clipped.shape {
            assert!(!t.galley.text().contains("Fixed Point C+"));
        }
    }

    // Verify JSON / #state= persistence round-trip (< 4096 bytes)
    let saved = plotter.save_state().expect("custom attractor state saves");
    assert!(saved.len() < crate::url_state::MAX_DECODED_URL_STATE_BYTES);
    let mut restored = AttractorPlotter::default();
    restored
        .try_load_state(&saved)
        .expect("custom attractor state loads");
    assert_eq!(restored.mode, AttractorMode::Custom);
    assert_eq!(
        restored.flow_bank.expressions(),
        vec!["sin(y) - b * x", "sin(z) - b * y", "sin(x) - b * z"]
    );
    assert_eq!(restored.flow_bank.params_map(), pmap);
}
