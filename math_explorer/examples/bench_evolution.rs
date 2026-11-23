//! Benchmark for Hawk-Dove Population simulation.
//!
//! This example measures the performance of the `update_frequencies` method,
//! which is expected to be called in a tight loop.
#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]

use math_explorer::biology::evolution::HawkDovePopulation;
use math_explorer::primitives::UnitInterval;
use std::time::Instant;

fn main() {
    let iterations = 10_000_000;
    let v = 2.0;
    let c = 10.0;
    let population = HawkDovePopulation::new(v, c);

    let mut hawk_freq = UnitInterval::new(0.9).unwrap();
    let dt = 0.0001; // Small dt for stability

    println!(
        "Benchmarking HawkDovePopulation::update_frequencies for {} iterations...",
        iterations
    );

    let start = Instant::now();

    for _ in 0..iterations {
        // We use unwrap here because we want to measure the raw performance of the function
        // including its internal overhead, but not the error handling overhead of the caller.
        hawk_freq = population.update_frequencies(hawk_freq, dt).unwrap();
    }

    let duration = start.elapsed();

    println!("Final Hawk Frequency: {:.6}", hawk_freq.value());
    println!("Total Time: {:?}", duration);
    println!("Time per iteration: {:?}", duration / iterations as u32);
}
