//! Example target for bench replicator.
#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]
use math_explorer::applied::game_theory::evolutionary::ReplicatorDynamics;
use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::pure_math::analysis::ode::RungeKutta4;
use std::time::Instant;

fn main() {
    // Scenario 2: Small system (N=10), many iterations.
    // This highlights allocation overhead.
    let size = 10;
    println!(
        "Benchmarking ReplicatorDynamics with {} strategies...",
        size
    );

    let mut data = Vec::with_capacity(size * size);
    for i in 0..size {
        for j in 0..size {
            let val = (i as f64 * std::f64::consts::PI + j as f64 * std::f64::consts::E).sin();
            data.push(val);
        }
    }
    let payoff = DMatrix::from_row_slice(size, size, &data);
    let system = ReplicatorDynamics::new(payoff).unwrap();

    let initial_state = DVector::from_element(size, 1.0 / size as f64);

    // Warmup
    let _ = system.derivative(&initial_state);

    let iterations = 1_000_000;
    println!("Running {} derivative calculations...", iterations);

    let start = Instant::now();
    let mut dummy_acc = 0.0;
    for _ in 0..iterations {
        let deriv = system.derivative(&initial_state);
        dummy_acc += deriv[0];
    }
    let duration = start.elapsed();

    println!("Time: {:.2?}", duration);
    println!(
        "Average time per call: {:.2?}",
        duration / iterations as u32
    );
    println!("Dummy: {}", dummy_acc);

    // Benchmarking simulation
    println!("Running long simulation with RK4...");
    let mut solver = RungeKutta4::new(&initial_state);
    // Simulate 10000 steps
    let steps = 10_000;
    let dt = 0.01;
    let time_horizon = steps as f64 * dt;

    let start_sim = Instant::now();
    let _traj = system.simulate_with_strategy(initial_state.clone(), time_horizon, dt, &mut solver);
    let duration_sim = start_sim.elapsed();
    println!("Simulation Time: {:.2?}", duration_sim);
}
