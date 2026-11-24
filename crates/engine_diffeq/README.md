# `engine_diffeq` — `#![no_std]` Stochastic Differential Equation & Langevin Engine

![Maturity: Validated](https://img.shields.io/badge/maturity-Validated-1f6feb.svg)

> **Maturity Tier:** `Validated` — `#![no_std]` stochastic and Langevin differential equation engine verified with seeded `OxidizeRng` convergence-order benchmarks and zero-allocation runtime tests.

## 1. Overview & Architecture

`engine_diffeq` is OxidizeMath's Tier 2 `#![no_std]` ordinary, stochastic, and geometric Langevin differential equation solver crate. It consolidates 0D stochastic differential equation (SDE) integrators, underdamped Hamiltonian/Langevin splittings, and discrete jump propensities over the Tier 0 [`Scalar`](math_commons::math_kernel::types::Scalar), [`SolverWorkspace`](math_commons::math_kernel::types::SolverWorkspace), and `rand_core::RngCore` contracts.

Core design invariants:
- **`#![no_std]` First with Optional `alloc`**: Compiles with `--no-default-features` on bare-metal and `wasm32-unknown-unknown` targets. Enabling `alloc` adds optional heap-backed workspace wrappers while all stepping loops remain strictly zero-allocation over caller-borrowed [`SolverWorkspace`](math_commons::math_kernel::types::SolverWorkspace) slices.
- **Multivariate Itô & Stratonovich Calculi**: [`StochasticCalculus`] selects either Itô ($\int G \, dW_t$) or Stratonovich ($\int G \circ dW_t$) integration across [`EulerMaruyamaSolver`] (strong order $0.5$ / Euler-Heun), [`MilsteinSolver`] (strong order $1.0$ with optional truncated Karhunen-Loève Lévy area), and [`Sriw1Solver`] (Rössler-type derivative-free strong-order-$1.5$ Stochastic Runge-Kutta).
- **Geometric Langevin Splitting**: [`BaoabLangevinSolver`] implements Leimkuhler-Matthews symmetric `B-A-O-A-B` splitting with exact Ornstein-Uhlenbeck fluctuation-dissipation thermostatting.
- **Explicit Deterministic Entropy**: Every stochastic step draws Box-Muller Gaussian or uniform samples exclusively from a caller-supplied `RngCore` stream.

```rust
use engine_diffeq::{
    EulerMaruyamaSolver, SdeSystem, StochasticCalculus,
};
use math_commons::math_kernel::types::SolverWorkspace;
use oxidize_core::rng::OxidizeRng;

struct OrnsteinUhlenbeck;

impl SdeSystem<f64> for OrnsteinUhlenbeck {
    fn state_dim(&self) -> usize { 1 }
    fn wiener_dim(&self) -> usize { 1 }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = -2.0 * x[0];
    }
    fn diffusion(&self, _t: f64, _x: &[f64], out: &mut [f64]) {
        out[0] = 0.5;
    }
}

let solver = EulerMaruyamaSolver::new(StochasticCalculus::Ito);
let mut state = [1.0_f64];
let mut scratch = [0.0_f64; 8];
let mut ws = SolverWorkspace::new(&mut scratch);
let mut rng = OxidizeRng::new(42);
solver.step(&OrnsteinUhlenbeck, 0.0, 0.01, &mut state, &mut ws, &mut rng);
assert!(state[0].is_finite());
```

## 2. Developer Instructions

Verify `#![no_std]`, `alloc`, native `std`, and WebAssembly builds from the workspace root:

```bash
cargo check -p engine_diffeq --no-default-features
cargo check -p engine_diffeq --no-default-features --features alloc
cargo check -p engine_diffeq --target wasm32-unknown-unknown --all-features
cargo test -p engine_diffeq
```
