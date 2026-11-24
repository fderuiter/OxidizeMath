//! Integration tests for bench lbm custom.
#[cfg(not(target_arch = "wasm32"))]
use domain_physics::physics::fluid_dynamics::lattice_boltzmann::LatticeBoltzmannD2Q9;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(not(target_arch = "wasm32"))]
use verified_engine::allocator::VerifiedAllocator;
#[cfg(not(target_arch = "wasm32"))]
use verified_engine::engine::VerifiedEngine;
#[cfg(not(target_arch = "wasm32"))]
use verified_engine::metrics::ComplexityMetrics;

#[cfg(not(target_arch = "wasm32"))]
#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[cfg(not(target_arch = "wasm32"))]
fn assert_lbm_invariants(
    rho: &[f64],
    ux: &[f64],
    uy: &[f64],
    expected_cells: usize,
    metrics: &ComplexityMetrics,
) {
    assert_eq!(rho.len(), expected_cells);
    assert!(rho.iter().all(|&d| d.is_finite() && d > 0.0));
    let total_mass: f64 = rho.iter().sum();
    assert!(
        (total_mass - expected_cells as f64).abs() < 1e-6,
        "LBM mass conservation violated: expected {}, got {}",
        expected_cells,
        total_mass
    );
    assert!(ux.iter().all(|v| v.is_finite()));
    assert!(uy.iter().all(|v| v.is_finite()));
    assert!(metrics.function_calls > 0);
    assert!(metrics.arithmetic_ops > 0);
    assert!(metrics.memory_loads > 0);
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::disallowed_methods)]
#[test]
#[verified_engine::verified]
fn bench_lbm_step() {
    let width = 100;
    let height = 100;
    let tau = 0.6; // Typical value
    let mut solver = LatticeBoltzmannD2Q9::new(width, height, tau);

    // Warmup
    for _ in 0..10 {
        solver.step();
    }

    let start = Instant::now();
    let iterations = 100;
    let ((), metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..iterations {
            solver.step();
        }
    });
    let duration = start.elapsed();

    assert_lbm_invariants(
        solver.state.density(),
        solver.state.velocity_x(),
        solver.state.velocity_y(),
        width * height,
        &metrics,
    );
    println!("LBM Step Benchmark (100x100, 100 iters): {:?}", duration);
    println!("Average time per step: {:?}", duration / iterations as u32);
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::disallowed_methods)]
#[test]
#[verified_engine::verified]
fn bench_lbm_obstacle_telemetry_zero_alloc() {
    let width = 100;
    let height = 50;
    let tau = 0.6;
    let mut solver = LatticeBoltzmannD2Q9::new(width, height, tau);
    for y in 20..=30 {
        for x in 28..=32 {
            solver.set_obstacle(x, y, true);
        }
    }

    for _ in 0..10 {
        solver.set_inlet(0, 0, 2, height, 0.1, 0.0);
        solver.step();
    }

    let start = Instant::now();
    let iterations = 100;
    let ((fx, fy, curl, solid), metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..iterations {
            solver.set_inlet(0, 0, 2, height, 0.1, 0.0);
            solver.step();
        }
        let (fx, fy) = solver.net_obstacle_force();
        let curl = solver.vorticity_curl(36, 22);
        let solid = solver.obstacle_cell_count();
        (fx, fy, curl, solid)
    });
    let duration = start.elapsed();

    assert_eq!(solid, 55);
    assert!(
        fx > 0.0,
        "Expected positive bounce-back drag force, got {fx}"
    );
    assert!(fy.is_finite() && curl.is_finite());
    assert!(metrics.function_calls > 0 && metrics.arithmetic_ops > 0);
    println!(
        "LBM Obstacle Telemetry Benchmark (100x50, 100 iters): {:?}",
        duration
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn bench_lbm_aborts_if_hot_loop_allocates() {
    const PROBE_ENV: &str = "OXIDIZE_LBM_ALLOC_ABORT_PROBE";
    if std::env::var_os(PROBE_ENV).is_some() {
        VerifiedEngine::run_verified(|| {
            let allocating_solver = LatticeBoltzmannD2Q9::new(8, 8, 0.6);
            std::hint::black_box(allocating_solver.state.density().len());
        });
        return;
    }
    let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .env(PROBE_ENV, "1")
        .args(["--exact", "bench_lbm_aborts_if_hot_loop_allocates"])
        .output()
        .expect("spawn LBM allocation probe");
    assert!(
        !output.status.success(),
        "VerifiedAllocator must abort when LBM hot loop allocates on the heap"
    );
}
