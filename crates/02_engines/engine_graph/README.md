# `engine_graph`: `#![no_std]` Probabilistic Graphical Model & Belief Propagation Engine

## 1. Overview & Architecture

`engine_graph` is OxidizeMath's Tier 2 `#![no_std]` graphical inference engine (`crates/02_engines/engine_graph`), providing zero-allocation message passing over discrete factor graphs, Markov chains, and Potts Markov random fields with optional `alloc` convenience wrappers.

Core subsystems:
- **`factor_graph`**: Slice-backed `FactorGraphView` and `FactorSpec` supporting unary, pairwise, and higher-order ($k \le 4$) discrete factors without heap allocations.
- **`bp_sum_product`**: Damped sum-product belief propagation (`sum_product_bp_into`) over caller-provided scratch slices, computing exact node marginals and Bethe log-partition $\ln Z$ on trees/chains and loopy BP marginals on cyclic graphs.
- **`bp_max_product`**: Log-domain max-product (max-sum) belief propagation (`max_product_bp_into`) computing MAP state assignments and normalized max-marginals without underflow.
- **`chain_mrf`**: Specialized zero-allocation 1D Markov chain scaled forward-backward (`chain_forward_backward_into`) and Potts MRF neighborhood posterior (`potts_neighborhood_posteriors_into`) kernels.

```rust
# use engine_graph::__doctest_facade as math_explorer;
use math_explorer::engine_graph::{BpConfig, FactorGraphView, FactorSpec, sum_product_bp_into};

let cards = [2, 2];
let factors = [
    FactorSpec::unary(0, 0, 2),
    FactorSpec::pairwise(0, 1, 2, 2, 2),
];
let potentials = [0.7, 0.3, 0.9, 0.1, 0.2, 0.8];
let fg = FactorGraphView::new(&cards, &factors, &potentials).unwrap();
let mut beliefs = [0.0; 4];
let mut scratch = [0.0; 18];
let report = sum_product_bp_into(&fg, &BpConfig::default(), &mut beliefs, &mut scratch).unwrap();
assert!(report.converged);
assert!((beliefs[0] - 0.7).abs() < 1e-12);
```

## 2. Developer Instructions

Verify `#![no_std]` compilation (with and without `alloc`), unit tests, zero-allocation message-passing sweeps, and `wasm32-unknown-unknown` compatibility:

```bash
cargo check -p engine_graph --no-default-features
cargo check -p engine_graph --no-default-features --features alloc
cargo check -p engine_graph --target wasm32-unknown-unknown
cargo test -p engine_graph
```
