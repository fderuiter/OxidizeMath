//! Example target for lorenz chaos.
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState};
use math_explorer::physics::chaos::metrics::lorenz_lyapunov;
use math_explorer::pure_math::analysis::ode::RungeKutta4;
use std::fmt;

// Simple ANSI color wrapper for better UX
enum Color {
    Cyan,
    Green,
    Yellow,
    Red,
    Bold,
    Reset,
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let code = match self {
            Color::Cyan => "\x1b[36m",
            Color::Green => "\x1b[32m",
            Color::Yellow => "\x1b[33m",
            Color::Red => "\x1b[31m",
            Color::Bold => "\x1b[1m",
            Color::Reset => "\x1b[0m",
        };
        write!(f, "{}", code)
    }
}

fn print_header() {
    println!();
    println!(
        "{}  🌌 LORENZ ATTRACTOR: THE BUTTERFLY EFFECT 🦋  {}",
        Color::Bold,
        Color::Reset
    );
    println!(
        "{}     (Deterministic Chaos Demonstration)      {}",
        Color::Cyan,
        Color::Reset
    );
    println!();

    println!("{}Description:{}", Color::Bold, Color::Reset);
    println!("We will simulate two Lorenz systems with almost identical initial conditions.");
    println!("Notice how the trajectories diverge exponentially over time.");
    println!();
}

fn run_simulation() {
    let x0 = 10.0;
    let y0 = 10.0;
    let z0 = 10.0;
    let epsilon = 0.0001;

    let state1 = LorenzState::new(x0, y0, z0);
    let state2 = LorenzState::new(x0 + epsilon, y0, z0);

    let mut sys1 = LorenzBuilder::new().build(state1);
    let mut sys2 = LorenzBuilder::new().build(state2);

    println!("{}Initial Conditions:{}", Color::Bold, Color::Reset);
    println!("System 1: ({:.4}, {:.4}, {:.4})", x0, y0, z0);
    println!(
        "System 2: ({:.4}, {:.4}, {:.4})  <-- +epsilon ({})",
        x0 + epsilon,
        y0,
        z0,
        epsilon
    );
    let mut rk4 = RungeKutta4::new(&state1.vec);
    let lambda_max = lorenz_lyapunov(&sys1, &mut rk4, state1.vec, 0.01, 200, 0.5)
        .expect("lorenz_lyapunov should succeed for standard chaotic Lorenz parameters");
    println!(
        "Maximal Lyapunov Exponent (λ_max): {}{:+.4}{} ({})",
        if lambda_max > 0.0 {
            Color::Red
        } else {
            Color::Green
        },
        lambda_max,
        Color::Reset,
        if lambda_max > 0.0 {
            "Chaotic: exponential divergence"
        } else {
            "Stable: bounded convergence"
        }
    );
    println!();

    simulate_systems(&mut sys1, &mut sys2);
}

fn simulate_systems(
    sys1: &mut math_explorer::physics::chaos::lorenz::LorenzSystem,
    sys2: &mut math_explorer::physics::chaos::lorenz::LorenzSystem,
) {
    let dt = 0.01;
    let steps = 3000;
    let print_interval = 200;

    println!("{}Simulation Log:{}", Color::Bold, Color::Reset);
    println!("   ┌────────┬──────────────┬──────────────┬──────────────┐");
    println!("   │  Time  │  Sys1 (x)    │  Sys2 (x)    │ Divergence Δ │");
    println!("   ├────────┼──────────────┼──────────────┼──────────────┤");

    for i in 0..=steps {
        if i % print_interval == 0 {
            let t = (i as f64) * dt;
            let x1 = sys1.state.vec.x;
            let x2 = sys2.state.vec.x;
            let diff = (sys1.state.vec - sys2.state.vec).norm();

            let divergence_color = if diff < 0.1 {
                Color::Green
            } else if diff < 5.0 {
                Color::Yellow
            } else {
                Color::Red
            };

            println!(
                "   │ {:6.2} │ {:12.4} │ {:12.4} │ {}{:12.4}{} │",
                t,
                x1,
                x2,
                divergence_color,
                diff,
                Color::Reset
            );
        }

        sys1.step(dt);
        sys2.step(dt);
    }

    println!("   └────────┴──────────────┴──────────────┴──────────────┘");
    println!();
    println!("{}Observation:{}", Color::Bold, Color::Reset);
    println!(
        "Initially, the systems track closely ({}Green{}).",
        Color::Green,
        Color::Reset
    );
    println!(
        "As time progresses, tiny differences amplify ({}Yellow{}).",
        Color::Yellow,
        Color::Reset
    );
    println!(
        "Eventually, they become completely uncorrelated ({}Red{}).",
        Color::Red,
        Color::Reset
    );
    println!();
}

fn main() {
    print_header();
    run_simulation();
}
