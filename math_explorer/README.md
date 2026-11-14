# Math Explorer Crate

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Crate](https://img.shields.io/crates/v/math_explorer.svg)

The core library for the **Math Explorer** project. This crate provides a collection of mathematical algorithms and models, implemented from first principles in Rust.

> **Note:** This is the inner library crate. For the full repository context, see the [Project Root README](../README.md).

##  Install

Add this crate to your dependencies:

```toml
[dependencies]
math_explorer = { path = "path/to/math_explorer" }
```

##  Usage

### Example: Computational Biology (Neuroscience)

Simulate a neuron's membrane potential using the Hodgkin-Huxley model:

```bash
cargo run --release --package math_explorer --example hodgkin_huxley_demo
```

*(See `math_explorer/examples/hodgkin_huxley_demo.rs` for implementation details)*

```rust
use math_explorer::biology::neuroscience::HodgkinHuxleyNeuron;

fn main() {
    // Initialize a neuron at resting potential (-65.0 mV)
    let mut neuron = HodgkinHuxleyNeuron::new(-65.0);
    let dt = 0.01; // 0.01 ms time step

    // Simulate for 10ms with 10 uA/cm^2 current injection
    for step in 0..1000 {
        neuron.update(dt, 10.0);
        if step % 100 == 0 {
            println!("t={:.2}ms, V={:.2}mV", step as f64 * dt, neuron.v());
        }
    }
}
```

### Example: Chaos Theory

Calculate the Lyapunov exponent for a Lorenz System:

```bash
cargo run --release --package math_explorer --example lorenz_chaos
```

*(See `math_explorer/examples/lorenz_chaos.rs` for implementation details)*

```rust
use math_explorer::physics::chaos::lorenz::{LorenzSystem, LorenzState};
use math_explorer::physics::chaos::metrics::lorenz_lyapunov;
use math_explorer::pure_math::analysis::ode::RungeKutta4;

fn main() {
    let initial_state = LorenzState::new(10.0, 10.0, 10.0);
    let system = LorenzSystem::default_chaotic(initial_state);
    let mut solver = RungeKutta4::new(&initial_state.vec);

    // Calculate the maximal Lyapunov exponent
    // Args: System, Solver, Initial Vec, Time Step, Iterations, Evolution Time
    let lambda = lorenz_lyapunov(&system, &mut solver, initial_state.vec, 0.01, 1000, 1.0).unwrap();
    println!("Lyapunov Exponent: {:.4}", lambda);
}
```

### Example: AI Transformer

Initialize a Transformer Encoder stack:

```bash
cargo run --release --package math_explorer --example transformer_demo
```

*(See `math_explorer/examples/transformer_demo.rs` for implementation details)*

```rust
use math_explorer::ai::transformer::Encoder;
use nalgebra::DMatrix;

fn main() {
    // 6 layers, 512 embedding dim, 8 heads, 2048 feed-forward dim
    let encoder = Encoder::new(6, 512, 8, 2048);

    // Dummy input: Sequence length 10, Embedding dim 512
    let input = DMatrix::zeros(10, 512);
    let _output = encoder.forward(input, None);
}
```

##  Config

The library is organized into high-level domains:

- **`ai`**: Artificial Intelligence primitives, including Transformer components (Attention, Encoder/Decoder), NeRF-Diffusion, and Self-Calibration loops.
- **`applied`**: Mathematical models applied to specific domains like **Clinical Trials** (Win Ratio), **Game Theory**, **Favoritism** (Satire), and **Isosurface Extraction**.
- **`biology`**: Biological modeling, including **Neuroscience** (Hodgkin-Huxley) and **Morphogenesis**.
- **`climate`**: Climate modeling tools, featuring the **CERA** autoencoder framework.
- **`epidemiology`**: Disease modeling, from standard **SIR/SEIR** compartmental models to stochastic network dynamics.
- **`physics`**: Simulations of physical systems, including **Quantum Mechanics**, **Fluid Dynamics**, **Chaos Theory** (Lorenz System), and **MRI Physics**.
- **`pure_math`**: Foundational mathematics, covering **Algebra**, **Number Theory** (Partitions, Q-Series), **Graph Theory**, and **Differential Geometry**.

##  Contributing

Please refer to [CONTRIBUTING.md](../CONTRIBUTING.md) in the project root.

To run the comprehensive test suite for all modules:

```bash
cargo test
```
