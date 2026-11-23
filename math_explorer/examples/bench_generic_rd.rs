//! Example target for bench generic rd.
#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]
use math_explorer::biology::diffusion::FiniteDifference1D;
use math_explorer::biology::morphogenesis::SchnakenbergKinetics;
use math_explorer::biology::reaction_diffusion::ReactionDiffusionSystem;
use std::time::Instant;

fn main() {
    let smoke = std::env::var("MATH_EXPLORER_SMOKE")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    let (size, warmup, iterations) = if smoke {
        (1_000, 2, 10)
    } else {
        (100_000, 10, 100)
    };

    let kinetics = SchnakenbergKinetics::default();
    let diffusion = FiniteDifference1D::new(math_explorer::math_kernel::types::StepSize(1.0));
    let diffusion_coeffs = vec![1.0, 40.0]; // u, v

    let mut system = ReactionDiffusionSystem::builder()
        .num_species(2)
        .grid_size(size)
        .reaction(kinetics)
        .diffusion(diffusion)
        .diffusion_coeffs(diffusion_coeffs)
        .build()
        .unwrap();

    // Initialize with noise
    for i in 0..size {
        system.state.species_mut(0)[i] = 1.0 + (i as f64 * 0.01).sin();
        system.state.species_mut(1)[i] = 0.5 + (i as f64 * 0.02).cos();
    }

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
