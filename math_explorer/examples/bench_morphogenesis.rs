//! Example target for bench morphogenesis.
#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]
use math_explorer::biology::morphogenesis::TuringSystem;
use std::time::Instant;

fn main() {
    let smoke = std::env::var("MATH_EXPLORER_SMOKE")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    let (size, warmup, iterations) = if smoke {
        (1_000, 5, 20)
    } else {
        (100_000, 100, 1000)
    };
    let mut system = TuringSystem::new(
        math_explorer::math_kernel::types::Dimension(size),
        math_explorer::biology::morphogenesis::DiffusionCoeff(0.1),
        math_explorer::biology::morphogenesis::DiffusionCoeff(0.05),
        math_explorer::math_kernel::types::StepSize(1.0),
    );

    // Warmup
    for _ in 0..warmup {
        system.step(0.01);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        system.step(0.01);
    }
    let duration = start.elapsed();

    println!(
        "Time for {} iterations with size {}: {:?}",
        iterations, size, duration
    );
    let avg_ns = duration.as_nanos() as f64 / iterations as f64;
    println!("Average per step: {:.2} ns", avg_ns);
}
