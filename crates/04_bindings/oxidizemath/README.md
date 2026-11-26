# OxidizeMath Facade Crate

![License](https://img.shields.io/badge/license-AGPL-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Crate](https://img.shields.io/crates/v/oxidizemath.svg)
![Maturity](https://img.shields.io/badge/maturity-Validated%20%2F%20Certified%20Kernels-2ea44f.svg)

Unified public facade crate re-exporting verified mathematical kernels, `#![no_std]` Tier 2 numerical engines, runtime verification primitives, and domain simulation modules.

> **Note:** This is the primary library facade crate. For the full repository context, see the [Project Root README](../../../README.md).

## 1. Overview & Architecture

The `oxidizemath` crate serves as the primary library entrypoint for downstream Rust consumers, CLI demonstrations (`oxidize-cli`), benchmarks, and the compatibility `math_explorer` crate. It unifies foundational primitives (`diagnostics`, `math_commons`, `scientific_metadata`, `verified_engine`) and Tier 2 engines (`engine_linalg`, `engine_diffeq`, `engine_pde`) with feature-gated scientific domains under a minimal `default = []` public namespace.

### Module Structure

- **Core Always-On Modules**:
  - `error` & `prelude`: Unified `MathError`, `MathResult`, `LinalgError`, `DiffeqError`, `PdeError`, bounded numeric primitives (`PositiveFloat`, `NonNegativeFloat`, `UnitInterval`), and `Diagnostic` traits.
  - `ast`, `constants`, `fit`, `math_kernel`, `poi`, `primitives`, `registry`, `series`, `stochastic`, `wigner`: Expression parsing/compilation, zero-allocation curve fitting and critical-point solvers, physical constants, tolerances, and Clebsch-Gordan coefficients.
  - `engine_linalg`, `engine_diffeq`, `engine_pde`: `#![no_std]` numerical linear algebra, stochastic differential equation steppers, and continuum PDE/SPDE solvers.
  - `verified_engine` & `scientific_metadata`: Zero-allocation runtime verification harness (`VerifiedEngine`), `#[verified]`, `#[derive(Theory)]`, and `#[embed_theory]`.
- **Feature-Gated Domain Modules**:
  - **`ai`**: Artificial Intelligence primitives, including Transformer components (Attention, Encoder/Decoder), Reinforcement Learning, NeRF-Diffusion, and Self-Calibration loops.
  - **`applied`**: Applied mathematical models including **Clinical Trials** (Win Ratio, Kaplan-Meier survival & bootstrap confidence bands), **Game Theory** (mechanism design revenue simulation), **Battery Degradation**, **Pharmacokinetics**, **Favoritism**, and **Isosurface Extraction**.
  - **`biology`**: Biological modeling, including **Neuroscience** (Hodgkin-Huxley) and **Morphogenesis** (Turing Reaction-Diffusion).
  - **`climate`**: Climate modeling tools featuring the **CERA** representation-alignment autoencoder framework.
  - **`epidemiology`**: Disease transmission modeling across **SIR/SEIR** compartmental ODEs, stochastic Gillespie processes, and contact networks.
  - **`physics`**: Physical simulations spanning **Quantum Mechanics**, **Fluid Dynamics**, **Solid State** (BCS Superconductivity, Ising), **Chaos Theory** (Lorenz System), and **MRI Physics**.
  - **`pure_math`**: Foundational mathematics covering **Abstract Algebra**, **Number Theory** (Partitions, Q-Series), **Graph Theory**, **Analysis** (`MultivariateIntegrator`), **Statistics** (`monte_carlo`, `qmc`, `bootstrap`, Multivariate Copulas, Glicko-2, Markov, TDA), and **Differential Geometry**.
  - **`serde`**: Opt-in `serde` serialization across foundational primitives, Tier 2 engines, and active external math re-exports (`nalgebra`, `num_complex`, `petgraph`, `rand`, `dashu`).

### Maturity Tiers & Module Matrix

| Facade Namespace | Default Tier | Promoted / Submodule Highlights | Verification & SemVer Contract |
| :--- | :--- | :--- | :--- |
| `fit`, `poi`, `ast`, `math_kernel`, `wigner` | `[CERT] Certified` | `FitModel`, Hermite/Lagrange `poi`, stack bytecode `CompiledExpr` | Non-vacuous `theory_verification!`, `VerifiedAllocator` zero-alloc proofs, SemVer-locked |
| `engine_linalg`, `engine_diffeq`, `engine_pde` | `[VAL] Validated` | `#![no_std]` QR/SVD/tridiagonal solvers, SDE/Langevin steppers, 1D/2D/3D PDE stencils | `#![no_std]` & `wasm32` verified; `LinalgError`, `DiffeqError`, `PdeError` unified in `error` |
| `scientific_metadata`, `verified_engine` | `[CERT] Certified` | `MaturityTier`, `TheoryDescribable`, `VerifiedEngine::run_verified` | Deterministic metadata & zero-allocation runtime harness, SemVer-locked |
| `pure_math` | `[VAL] Validated` | `[CERT]` `analysis::ode` (`OdeModel`), `roots`, `wave_2d`, `algebra`, `number_theory`, `elliptic_curves`; `[EXP]` `algorithmic_info` | `Certified` solvers SemVer-locked; `Validated` modules seeded & golden-tested |
| `physics` | `[VAL] Validated` | `[CERT]` `quantum`, `chaos`, `stat_mech::sandpile`; `[EXP]` `nuclear`, `solid_state` | Symplectic & conservation invariants verified |
| `biology` | `[VAL] Validated` | `[CERT]` `neuroscience` (Hodgkin-Huxley), `kinetics`, `morphogenesis` | Deterministic ODE/PDE spike & pattern benchmarks |
| `epidemiology` | `[VAL] Validated` | `[VAL]` `compartmental` (`SIRModel`, `SEIRModel`), `networks`, `stochastic` | Seeded `OxidizeRng` & conservation assertions |
| `applied` | `[VAL] Validated` | `[CERT]` `clinical_trials`, `pharmacokinetics`; `[VAL]` `win_ratio`, `battery_degradation`; `[EXP]` `favoritism`, `lorahub`, `grpo` | Clinical & ADME solvers SemVer-locked; `favoritism` fluid `Experimental` |
| `ai` | `[VAL] Validated` | `[VAL]` `transformer`, `activations`, `reinforcement_learning`; `[EXP]` `gaussian_splatting`, `sds` | Seeded matrix & Bellman convergence tests |
| `climate` | `[EXP] Experimental` | `[EXP]` `cera` (`Cera` representation-alignment autoencoder) | Exploratory research prototype; fluid API evolution |

---

## 2. Developer Instructions

### Installation & Feature Selection

Add `oxidizemath` to your `Cargo.toml` and enable only the domain features required by your application (`default = []` keeps the core minimal and CLI-free):

```toml
[dependencies]
oxidizemath = { path = "path/to/oxidizemath", features = ["biology", "physics", "pure_math", "ai"] }
```

### Example: Zero-Allocation Curve Fitting & Critical Points (Default Features)

Fit polynomial or nonlinear models, query their `MaturityTier`, and locate sub-segment crossings without enabling any domain or GUI features:

```rust
use oxidizemath::fit::{FitModel, fit_model};
use oxidizemath::poi::{closest_threshold_crossing, intersect_polylines};
use oxidizemath::prelude::{MaturityTier, TheoryDescribable};

assert_eq!(FitModel::Exponential.maturity_tier(), MaturityTier::Certified);

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

Simulate a neuron's membrane potential using the Hodgkin-Huxley model via `oxidize-cli`:

```bash
cargo run --release --package oxidize-cli --features biology --example hodgkin_huxley_demo
```

*(See `apps/oxidize-cli/examples/hodgkin_huxley_demo.rs` for implementation details)*

```rust
# #[cfg(feature = "biology")]
use oxidizemath::biology::neuroscience::HodgkinHuxleyNeuron;

# #[cfg(feature = "biology")]
fn main() {
    let mut neuron = HodgkinHuxleyNeuron::new(-65.0);
    let dt = 0.01;
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

---

## 3. Ecosystem Integration

Please refer to [CONTRIBUTING.md](../../../CONTRIBUTING.md) in the project root for contributor standards and facade export hygiene rules.

To run the comprehensive test suite across all domain modules:

```bash
cargo test -p oxidizemath --all-features
```
