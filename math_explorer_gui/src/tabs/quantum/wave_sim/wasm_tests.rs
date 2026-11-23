#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::async_sim::SimulationRunner;
use crate::async_sim::atlas::{AtlasPassMeta, AtlasRenderRequest, AtlasRunner, RegimeCartographer};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, RwLock};
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn wasm_smoke_quantum_tunneling_atlas_and_double_barrier_latch() {
    let params = wave_parameter_constraints();
    let expr = presets::default_compiled_expr();
    let cart = Arc::new(QuantumTunnelingCartographer::new(
        &params,
        ("k0", "v0"),
        WavePotentialPreset::DoubleBarrier,
        4.6385,
        20.0,
        1.0,
        0.55,
        &expr,
        &[],
    ));
    let res = cart.evaluate_cell(4.6385, 20.0);
    assert!(res.primary > 0.99);

    let req = Arc::new(RwLock::new(AtlasRenderRequest {
        generation: 1,
        width: 32,
        height: 32,
        high_contrast: true,
        cartographer: cart,
    }));
    let generation = Arc::new(AtomicU64::new(1));
    let mut atlas_runner = AtlasRunner::new(req, generation);
    for _ in 0..48 {
        atlas_runner.step();
        let snap = atlas_runner.get_snapshot();
        if snap
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<AtlasPassMeta>())
            .is_some_and(|m| m.is_final)
        {
            break;
        }
    }
    let final_snap = atlas_runner.get_snapshot();
    let meta = final_snap
        .structured_data
        .as_ref()
        .and_then(|d| d.downcast_ref::<AtlasPassMeta>())
        .expect("WASM AtlasPassMeta");
    assert!(meta.is_final);
    assert_eq!(meta.stride, 1);
}

#[wasm_bindgen_test]
fn wasm_smoke_double_barrier_quasi_bound_latch() {
    let cfg = WaveSimConfig {
        preset: WavePotentialPreset::DoubleBarrier,
        x0: -2.5,
        k0: 4.6385,
        sigma: 0.55,
        v0: 20.0,
        width: 1.0,
        dt: 0.02,
        steps_per_frame: 5,
        custom_expr: presets::default_compiled_expr(),
        sparse_potential: Vec::new(),
    };
    let mut runner = WaveRunner::new(Arc::new(RwLock::new(cfg)));
    for _ in 0..160 {
        runner.step();
        if runner.latched_transmission().is_some() {
            break;
        }
    }
    assert!(
        runner
            .latched_transmission()
            .is_some_and(|t| t.is_finite() && t > 0.08)
    );
}
