//! Integration tests for bench lbm custom.
#[cfg(not(target_arch = "wasm32"))]
use domain_physics::physics::fluid_dynamics::lattice_boltzmann::LatticeBoltzmannD2Q9;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

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
    for _ in 0..iterations {
        solver.step();
    }
    let duration = start.elapsed();

    println!("LBM Step Benchmark (100x100, 100 iters): {:?}", duration);
    println!("Average time per step: {:?}", duration / iterations as u32);
}
