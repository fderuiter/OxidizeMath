# Math Explorer Compatibility Facade

![License](https://img.shields.io/badge/license-AGPL-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Crate](https://img.shields.io/crates/v/math_explorer.svg)

Transitional compatibility facade crate re-exporting the primary [`oxidizemath`](../oxidizemath/README.md) library namespace with identical `default = []` feature flags.

> **Note:** New downstream integrations should depend directly on `oxidizemath` (`crates/04_bindings/oxidizemath`). CLI demonstrations live in [`oxidize-cli`](../../../apps/oxidize-cli/README.md). For full repository context, see the [Project Root README](../../../README.md).

## 1. Overview & Architecture

`math_explorer` preserves source and feature-flag compatibility for existing consumers (`math_explorer_gui`, benchmarks, and external callers) by forwarding every domain feature (`pure_math`, `applied`, `ai`, `biology`, `climate`, `epidemiology`, `physics`, and `serde`) to `oxidizemath` and re-exporting `pub use oxidizemath::*;`.

- **Zero CLI Target Bloat**: All 32 runnable domain examples reside in `apps/oxidize-cli`, keeping library builds minimal.
- **Exact Feature Parity**: `default = []` compiles only core primitives and `#![no_std]` Tier 2 engines; enabling any domain feature activates the corresponding `oxidizemath` namespace.

## 2. Developer Instructions

### Compatibility Dependency Declaration

```toml
[dependencies]
math_explorer = { path = "path/to/math_explorer", features = ["applied", "physics"] }
```

### Example: Re-Exported Curve Fitting (`default = []`)

```rust
use math_explorer::fit::{FitModel, fit_model};

let xs = [0.0, 1.0, 2.0, 3.0];
let ys = [1.0, 3.0, 5.0, 7.0];
let fit = fit_model(&(&xs[..], &ys[..]), FitModel::Linear, false).expect("linear fit");
assert!(fit.r_squared > 0.999_999);
```

### Running Criterion Benchmarks

```bash
cargo bench --package math_explorer --features applied --bench sorting_benchmark
```

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for workspace facade governance.
