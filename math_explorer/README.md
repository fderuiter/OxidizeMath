# Math Explorer Crate

![License](https://img.shields.io/badge/license-AGPL-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Crate](https://img.shields.io/crates/v/math_explorer.svg)

Unified public facade crate re-exporting verified mathematical kernels, runtime verification primitives, and domain simulation modules.

> **Note:** This is the inner library crate. For the full repository context, see the [Project Root README](../README.md).

## 1. Overview & Architecture

The `math_explorer` crate serves as the primary library entrypoint for downstream Rust consumers, examples, benchmarks, and the interactive GUI. It unifies foundational primitives (`diagnostics`, `math_commons`, `scientific_metadata`, `verified_engine`) with feature-gated scientific domains under a stable public namespace.

### Module Structure

- **Core Always-On Modules**:
  - `error` & `prelude`: Unified `MathError`, `MathResult`, bounded numeric primitives (`PositiveFloat`, `NonNegativeFloat`, `UnitInterval`), and `Diagnostic` traits.
  - `ast`, `constants`, `fit`, `math_kernel`, `poi`, `primitives`, `registry`, `series`, `wigner`: Expression parsing/compilation, zero-allocation curve fitting and critical-point solvers, physical constants, tolerances, and Clebsch-Gordan coefficients.
  - `verified_engine` & `scientific_metadata`: Zero-allocation runtime verification harness (`VerifiedEngine`), `#[verified]`, `#[derive(Theory)]`, and `#[embed_theory]`.
- **Feature-Gated Domain Modules**:
  - **`ai`**: Artificial Intelligence primitives, including Transformer components (Attention, Encoder/Decoder), Reinforcement Learning, NeRF-Diffusion, and Self-Calibration loops.
  - **`applied`**: Applied mathematical models including **Clinical Trials** (Win Ratio), **Game Theory**, **Battery Degradation**, **Pharmacokinetics**, **Favoritism**, and **Isosurface Extraction**.
  - **`biology`**: Biological modeling, including **Neuroscience** (Hodgkin-Huxley) and **Morphogenesis** (Turing Reaction-Diffusion).
  - **`climate`**: Climate modeling tools featuring the **CERA** representation-alignment autoencoder framework.
  - **`epidemiology`**: Disease transmission modeling across **SIR/SEIR** compartmental ODEs, stochastic Gillespie processes, and contact networks.
  - **`physics`**: Physical simulations spanning **Quantum Mechanics**, **Fluid Dynamics**, **Solid State** (BCS Superconductivity, Ising), **Chaos Theory** (Lorenz System), and **MRI Physics**.
  - **`pure_math`**: Foundational mathematics covering **Abstract Algebra**, **Number Theory** (Partitions, Q-Series), **Graph Theory**, **Statistics** (Glicko-2, Markov, TDA), and **Differential Geometry**.

---

## 2. Developer Instructions

### Installation & Feature Selection

Add `math_explorer` to your `Cargo.toml` and enable the domain features required by your application:

```toml
[dependencies]
math_explorer = { path = "path/to/math_explorer", features = ["biology", "physics", "pure_math", "ai"] }
```

### Example: Zero-Allocation Curve Fitting & Critical Points (Default Features)

Fit polynomial or nonlinear models and locate sub-segment crossings without enabling any domain or GUI features:

```rust
use math_explorer::fit::{FitModel, fit_model};
use math_explorer::poi::{closest_threshold_crossing, intersect_polylines};

let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
let ys = [1.0, 2.7182818, 7.3890561, 20.085537, 54.59815];
let series = (&xs[..], &ys[..]);

let fit = fit_model(&series, FitModel::Exponential, false).expect("exponential fit");
assert!(fit.r_squared > 0.999_999);

let root_x = closest_threshold_crossing(&series, 10.0, 2.0).expect("threshold root");
assert!((root_x - 10.0_f64.ln()).abs() < 0.02);

let ref_line = [[0.0, 15.0], [4.0, 15.0]];
let hits = intersect_polylines(&series, &ref_line, [0.0, 4.0]);
assert_eq!(hits.len(), 1);
```
### Example: Computational Biology (Neuroscience)

Simulate a neuron's membrane potential using the Hodgkin-Huxley model:

```bash
cargo run --release --package math_explorer --features biology --example hodgkin_huxley_demo
```

*(See `math_explorer/examples/hodgkin_huxley_demo.rs` for implementation details)*

```rust
# #[cfg(feature = "biology")]
use math_explorer::biology::neuroscience::HodgkinHuxleyNeuron;

# #[cfg(feature = "biology")]
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
# #[cfg(not(feature = "biology"))]
# fn main() {}
```

### Example: Chaos Theory

Calculate the Lyapunov exponent for a Lorenz System:

```bash
cargo run --release --package math_explorer --features physics --example lorenz_chaos
```

*(See `math_explorer/examples/lorenz_chaos.rs` for implementation details)*

```rust
# #[cfg(all(feature = "physics", feature = "pure_math"))]
use math_explorer::physics::chaos::lorenz::{LorenzSystem, LorenzState};
# #[cfg(all(feature = "physics", feature = "pure_math"))]
use math_explorer::physics::chaos::metrics::lorenz_lyapunov;
# #[cfg(all(feature = "physics", feature = "pure_math"))]
use math_explorer::pure_math::analysis::ode::RungeKutta4;

# #[cfg(all(feature = "physics", feature = "pure_math"))]
fn main() {
    let initial_state = LorenzState::new(10.0, 10.0, 10.0);
    let system = LorenzSystem::default_chaotic(initial_state);
    let mut solver = RungeKutta4::new(&initial_state.vec);

    // Calculate the maximal Lyapunov exponent
    let lambda = lorenz_lyapunov(&system, &mut solver, initial_state.vec, 0.01, 1000, 1.0).unwrap();
    println!("Lyapunov Exponent: {:.4}", lambda);
}
# #[cfg(not(all(feature = "physics", feature = "pure_math")))]
# fn main() {}
```

### Example: AI Transformer

Initialize a Transformer Encoder stack:

```bash
cargo run --release --package math_explorer --features ai --example transformer_demo
```

*(See `math_explorer/examples/transformer_demo.rs` for implementation details)*

```rust
# #[cfg(feature = "ai")]
use math_explorer::ai::transformer::Encoder;
# #[cfg(feature = "ai")]
use math_explorer::nalgebra::DMatrix;

# #[cfg(feature = "ai")]
fn main() {
    // 6 layers, 512 embedding dim, 8 heads, 2048 feed-forward dim
    let encoder = Encoder::new(6, 512, 8, 2048);

    // Dummy input: Sequence length 10, Embedding dim 512
    let input = DMatrix::zeros(10, 512);
    let _output = encoder.forward(input, None);
}
# #[cfg(not(feature = "ai"))]
# fn main() {}
```

---

## 3. Ecosystem Integration

Please refer to [CONTRIBUTING.md](../CONTRIBUTING.md) in the project root for contributor standards and facade export hygiene rules. Run `cargo test -p math_explorer --all-features` to execute the facade integration and doctest suites.
