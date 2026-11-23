#![allow(missing_docs)]
#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]
use math_explorer::physics::stat_mech::KB;
use math_explorer::physics::stat_mech::ising::SpinLattice;
use std::time::Instant;

fn main() {
    let width = 100;
    let height = 100;
    let j_coupling = 1.0;
    let h_field = 0.0;
    // T < Tc to ensure interesting dynamics (clustering)
    let temp = 2.0 * j_coupling / KB;

    let iterations = 10_000_000;

    println!(
        "Benchmarking Ising Model {}x{} for {} iterations...",
        width, height, iterations
    );

    // --- Before: metropolis_step loop ---
    let mut lattice = SpinLattice::new(width, height, None);
    let start = Instant::now();
    for _ in 0..iterations {
        lattice.metropolis_step(temp, j_coupling, h_field);
    }
    let duration = start.elapsed();
    let old_speed = (iterations as f64 / duration.as_secs_f64()) / 1_000_000.0;
    println!(
        "Before (metropolis_step): {:.4}s, {:.2} M/s",
        duration.as_secs_f64(),
        old_speed
    );

    // --- After: evolve batch ---
    let mut lattice = SpinLattice::new(width, height, None);
    let start = Instant::now();
    lattice.evolve(iterations, temp, j_coupling, h_field);
    let duration = start.elapsed();
    let new_speed = (iterations as f64 / duration.as_secs_f64()) / 1_000_000.0;
    println!(
        "After (evolve):           {:.4}s, {:.2} M/s",
        duration.as_secs_f64(),
        new_speed
    );

    let speedup = new_speed / old_speed;
    println!("Speedup: {:.2}x", speedup);
}
