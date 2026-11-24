#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::attractors::{AttractorCartographer, AttractorMode, AttractorPlotter, CompiledFlowJob};
use super::bifurcation::{
    BifurcationDiagram, BifurcationMapMode, BifurcationRunner, BifurcationSweepRequest,
};
use super::fractals::{FractalMode, FractalRenderRequest, FractalRunner, FractalViewer};
use crate::async_sim::SimulationRunner;
use crate::async_sim::atlas::{AtlasRenderRequest, AtlasRunner, AtlasTeleportEvent};
use crate::framework::InteractiveTool;
use crate::web_io::PlatformInstant;
use num_complex::Complex;
use scientific_metadata::theory::TheoryDescribable;
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
    for _ in 0..256 {
        let t0 = PlatformInstant::now();
        f_runner.step();
        assert!(t0.elapsed().as_millis() < 100);
        if f_runner.is_complete() {
            break;
        }
    }
    assert!(f_runner.is_complete());
}

#[wasm_bindgen_test]
fn wasm_bifurcation_and_atlas_runners_complete_cleanly() {
    let b_req = BifurcationSweepRequest {
        generation: 1,
        map_mode: BifurcationMapMode::Logistic,
        r_min: 2.5,
        r_max: 4.0,
        steps: 240,
        x0: 0.5,
    };
    let b_shared = Arc::new(RwLock::new(BifurcationSweepRequest {
        generation: 0,
        ..b_req
    }));
    let b_formula = Arc::new(RwLock::new(None));
    let b_gen = Arc::new(AtomicU64::new(0));
    let mut b_runner = BifurcationRunner::new(
        Arc::clone(&b_shared),
        Arc::clone(&b_gen),
        Arc::clone(&b_formula),
    );
    *b_shared.write().unwrap() = b_req;
    b_gen.store(1, Ordering::Release);
    for _ in 0..256 {
        b_runner.step();
        if b_runner.is_complete() {
            break;
        }
    }
    assert!(b_runner.is_complete());

    let plotter = AttractorPlotter::default();
    let cart = Arc::new(AttractorCartographer::new(
        &plotter.theory_parameters(),
        ("sigma", "rho"),
        AttractorMode::Lorenz,
        CompiledFlowJob::from_bank(&plotter.flow_bank),
        (10.0, 28.0, 8.0 / 3.0),
        (0.2, 0.2),
    ));
    let a_shared = Arc::new(RwLock::new(AtlasRenderRequest {
        generation: 1,
        width: 12,
        height: 8,
        high_contrast: false,
        cartographer: cart,
    }));
    let a_gen = Arc::new(AtomicU64::new(1));
    let mut a_runner = AtlasRunner::new(a_shared, a_gen);
    for _ in 0..256 {
        a_runner.step();
        if a_runner.is_complete() {
            break;
        }
    }
    assert!(a_runner.is_complete());
}

#[wasm_bindgen_test]
fn wasm_mandelbrot_to_julia_teleport_cobweb_selection_and_state_roundtrips() {
    let mut fractals = FractalViewer {
        center: Complex::new(-0.55, 0.25),
        zoom: 3.0,
        ..Default::default()
    };
    fractals.sync_active_plane_view();
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
    bif.select_probe_r(3.83);
    assert!(bif.probe_lyapunov().expect("period-3 lyapunov") < 0.0);
    bif.select_probe_r(3.95);
    assert!(bif.probe_lyapunov().expect("chaotic lyapunov") > 0.0);
    let b_json = bif.save_state().expect("bifurcation save_state");
    let mut b_restored = BifurcationDiagram::default();
    b_restored
        .try_load_state(&b_json)
        .expect("bifurcation try_load_state");
    assert!((b_restored.probe_r - 3.95).abs() < 1e-9);

    let mut attr = AttractorPlotter::default();
    attr.apply_atlas_teleport(AtlasTeleportEvent {
        x_key: "sigma",
        x_val: 10.0,
        y_key: "rho",
        y_val: 12.0,
        regime_label: "Equilibrium Sink",
    });
    assert!(attr.current_lyapunov_exponent().expect("stable lyapunov") < 0.0);
    let a_json = attr.save_state().expect("attractor save_state");
    let mut a_restored = AttractorPlotter::default();
    a_restored
        .try_load_state(&a_json)
        .expect("attractor try_load_state");
    assert!((a_restored.system.rho - 12.0).abs() < 1e-9);
}
