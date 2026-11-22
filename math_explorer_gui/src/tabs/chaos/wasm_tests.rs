#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::attractors::{AttractorPlotter, LorenzRegimeMapRunner, LorenzRegimeRequest};
use super::bifurcation::{BifurcationDiagram, BifurcationRunner, BifurcationSweepRequest};
use super::fractals::{FractalMode, FractalRenderRequest, FractalRunner, FractalViewer};
use crate::async_sim::SimulationRunner;
use crate::framework::InteractiveTool;
use crate::web_io::PlatformInstant;
use num_complex::Complex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn wasm_fractal_runner_cooperative_slices_complete_cleanly() {
    let f_req = FractalRenderRequest {
        generation: 1,
        width: 64,
        height: 48,
        center: Complex::new(-0.75, 0.0),
        zoom: 1.0,
        max_iter: 120,
        mode: FractalMode::Mandelbrot,
        julia_c: Complex::new(-0.4, 0.6),
        high_contrast: false,
    };
    let f_shared = Arc::new(RwLock::new(FractalRenderRequest {
        generation: 0,
        width: 16,
        height: 12,
        ..f_req
    }));
    let f_gen = Arc::new(AtomicU64::new(0));
    let mut f_runner = FractalRunner::new(Arc::clone(&f_shared), Arc::clone(&f_gen));
    *f_shared.write().unwrap() = f_req;
    f_gen.store(1, Ordering::Release);
    for _ in 0..32 {
        let t0 = PlatformInstant::now();
        f_runner.step();
        assert!(t0.elapsed().as_millis() < 50);
        if f_runner.is_complete() {
            break;
        }
    }
    assert!(f_runner.is_complete());
}

#[wasm_bindgen_test]
fn wasm_bifurcation_and_regime_runners_complete_cleanly() {
    let b_req = BifurcationSweepRequest {
        generation: 1,
        r_min: 2.5,
        r_max: 4.0,
        steps: 240,
    };
    let b_shared = Arc::new(RwLock::new(BifurcationSweepRequest {
        generation: 0,
        ..b_req
    }));
    let b_gen = Arc::new(AtomicU64::new(0));
    let mut b_runner = BifurcationRunner::new(
        Arc::clone(&b_shared),
        Arc::clone(&b_gen),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    *b_shared.write().unwrap() = b_req;
    b_gen.store(1, Ordering::Release);
    for _ in 0..32 {
        b_runner.step();
        if b_runner.is_complete() {
            break;
        }
    }
    assert!(b_runner.is_complete());

    let r_req = LorenzRegimeRequest {
        generation: 1,
        beta: 8.0 / 3.0,
        width: 12,
        height: 8,
        high_contrast: false,
    };
    let r_shared = Arc::new(RwLock::new(r_req));
    let r_gen = Arc::new(AtomicU64::new(1));
    let mut r_runner = LorenzRegimeMapRunner::new(r_shared, r_gen);
    for _ in 0..32 {
        r_runner.step();
        if r_runner.is_complete() {
            break;
        }
    }
    assert!(r_runner.is_complete());
}

#[wasm_bindgen_test]
fn wasm_mandelbrot_to_julia_teleport_cobweb_selection_and_state_roundtrips() {
    let mut fractals = FractalViewer {
        center: Complex::new(-0.55, 0.25),
        zoom: 3.0,
        ..Default::default()
    };
    fractals.sync_active_viewport_cache();
    fractals.teleport_to_julia(Complex::new(-0.7, 0.27015));
    assert_eq!(fractals.mode, FractalMode::Julia);
    assert!((fractals.julia_c.re + 0.7).abs() < 1e-9);

    let f_json = fractals.save_state().expect("fractal save_state");
    let mut f_restored = FractalViewer::default();
    f_restored
        .try_load_state(&f_json)
        .expect("fractal try_load_state");
    assert_eq!(f_restored.mode, FractalMode::Julia);

    fractals.switch_mode(FractalMode::Mandelbrot);
    assert!((fractals.center.re + 0.55).abs() < 1e-9);
    assert!((fractals.zoom - 3.0).abs() < 1e-9);

    let mut bif = BifurcationDiagram::default();
    bif.select_r(3.83);
    assert!(bif.selected_lyapunov() < 0.0);
    bif.select_r(3.95);
    assert!(bif.selected_lyapunov() > 0.0);
    let b_json = bif.save_state().expect("bifurcation save_state");
    let mut b_restored = BifurcationDiagram::default();
    b_restored
        .try_load_state(&b_json)
        .expect("bifurcation try_load_state");
    assert!((b_restored.selected_r - 3.95).abs() < 1e-9);

    let mut attr = AttractorPlotter::default();
    attr.teleport_regime(12.0, 10.0);
    assert!(attr.latest_lyapunov.expect("stable lyapunov") < 0.0);
    let a_json = attr.save_state().expect("attractor save_state");
    let mut a_restored = AttractorPlotter::default();
    a_restored
        .try_load_state(&a_json)
        .expect("attractor try_load_state");
    assert!((a_restored.system.rho - 12.0).abs() < 1e-9);
}
