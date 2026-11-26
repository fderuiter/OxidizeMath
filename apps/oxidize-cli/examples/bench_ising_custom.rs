//! Example target for bench ising custom.
#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]
use oxidizemath::physics::stat_mech::KB;
use oxidizemath::physics::stat_mech::ising::SpinLattice;
use std::time::Instant;

fn main() {
    let smoke = std::env::var("MATH_EXPLORER_SMOKE")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    let (width, height, iterations) = if smoke {
        (20, 20, 5_000)
    } else {
        (100, 100, 10_000_000)
    };
    let j_coupling = 1.0;
    let h_field = 0.0;
    // T < Tc to ensure interesting dynamics (clustering)
    let temp = 2.0 * j_coupling / KB;

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
    let old_speed = (iterations as f64 / duration.as_secs_f64().max(1e-9)) / 1_000_000.0;
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
    let new_speed = (iterations as f64 / duration.as_secs_f64().max(1e-9)) / 1_000_000.0;
    println!(
        "After (evolve):           {:.4}s, {:.2} M/s",
        duration.as_secs_f64(),
        new_speed
    );

    let speedup = new_speed / old_speed;
    println!("Speedup: {:.2}x", speedup);
}
