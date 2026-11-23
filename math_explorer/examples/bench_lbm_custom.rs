#![allow(missing_docs)]
#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::LatticeBoltzmannD2Q9;
use std::time::Instant;

fn main() {
    let width = 100;
    let height = 100;
    let tau = 0.6;
    let mut solver = LatticeBoltzmannD2Q9::new(width, height, tau);

    println!("Running LBM Benchmark (100x100 grid, tau=0.6)...");

    // Warmup
    for _ in 0..100 {
        solver.step();
    }

    let iterations = 2000;
    let start = Instant::now();
    for _ in 0..iterations {
        solver.step();
    }
    let duration = start.elapsed();

    println!("Time for {} iterations: {:?}", iterations, duration);
    let avg_ns = duration.as_nanos() as f64 / iterations as f64;
    println!("Average per step: {:.2} ns", avg_ns);
    // Simple verification (checksum) to prevent compiler from optimizing away the loop
    println!(
        "Final density checksum: {:.6}",
        solver.state.density().iter().sum::<f64>()
    );
}
