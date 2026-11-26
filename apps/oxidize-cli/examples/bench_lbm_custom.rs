//! Example target for bench lbm custom.
#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]
use oxidizemath::physics::fluid_dynamics::lattice_boltzmann::LatticeBoltzmannD2Q9;
use std::time::Instant;

fn main() {
    let smoke = std::env::var("MATH_EXPLORER_SMOKE")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    let (width, height, warmup, iterations) = if smoke {
        (24, 24, 5, 20)
    } else {
        (100, 100, 100, 2000)
    };
    let tau = 0.6;
    let mut solver = LatticeBoltzmannD2Q9::new(width, height, tau);

    println!(
        "Running LBM Benchmark ({}x{} grid, tau=0.6)...",
        width, height
    );

    // Warmup
    for _ in 0..warmup {
        solver.step();
    }

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
