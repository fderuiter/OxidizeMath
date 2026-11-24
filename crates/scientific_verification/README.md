# Scientific Verification

Host-only 6-tier scientific verification harness, solver convergence engine, and proof-certificate generator for the OxidizeMath workspace.

## 1. Overview & Architecture

The `scientific_verification` crate provides shared numerical certification harnesses executed by `unified_verification` and `xtask` across all eight mathematical crates (`math_commons`, `pure_math`, `domain_ai`, `domain_physics`, `domain_biology`, `domain_applied`, `domain_climate`, and `domain_epidemiology`) while staying strictly excluded from `wasm32` runtime bundles.

### Verification Pyramid Tiers

- **Tier 1 — Manufactured-Solution & Grid-Refinement Convergence (`mms`, `solvers`)**: Certifies all 48 differential, quadrature, optimization, and linear solvers using log-log ordinary least-squares convergence slopes and zero-heap-allocation hot-loop assertions.
- **Tier 2 — Deterministic Property & Conservation Sweeps (`proptest_sweeps`)**: Executes ChaCha-seeded deterministic `proptest` suites across copula monotonicity/bounds, unitary Crank-Nicolson quantum propagation, `LatticeBoltzmann` mass/momentum conservation, Kaplan-Meier survival probabilities, and Bessel Wronskians/recurrences.
- **Tier 3 — Offline High-Precision Golden Fixtures (`golden`)**: Validates special functions (`bessel_j`, `bessel_y`, `spherical_bessel_j` including asymptotic arguments $x = 40, 50, 100$, `legendre_p`, `hermite`, `associated_laguerre`, `gamma_function`, `beta_function`) and `GaussianCopula` against 150-digit offline fixtures within `< 1e-12`.
- **Tier 4 — Seeded Stochastic Moment & Confidence Bounds (`stochastic`)**: Verifies deterministic ChaCha-seeded Ornstein-Uhlenbeck stationary moments and Monte Carlo parlay pricing confidence intervals.
- **Tier 5 — IEEE-754 Outward-Rounded Interval Enclosures (`intervals`)**: Evaluates rigorous outward-rounded `Interval` enclosures for special-function and ODE kernels and records maximum enclosure widths.
- **Tier 6 — Mechanized Lean 4 Proof Certificates (`lean_proofs`)**: Verifies that every `.lean` module under `proofs/OxidizeMath/` is free of `sorry`, `admit`, and `axiom`, computes deterministic FIPS 180-4 SHA-256 digests, and runs `lake build` when a Lean 4 toolchain is installed.

---

## 2. Developer Instructions

### Outward-Rounded Interval Enclosures

Core interval enclosure primitives certified by Tier 5 live in `math_commons::primitives` (and are also re-exported through `math_explorer::prelude`):

```rust
use math_commons::primitives::{Interval, UnitIntervalEnclosure};

let half = Interval::point(0.5).unwrap();
let third = Interval::point(1.0)
    .unwrap()
    .checked_div(Interval::point(3.0).unwrap())
    .unwrap();
let sum = half + third;
assert!(sum.contains(5.0 / 6.0));
assert!(sum.width() < 1e-14);

let prob = UnitIntervalEnclosure::new(sum.lo(), sum.hi()).unwrap();
assert!(prob.interval().lo() >= 0.0 && prob.interval().hi() <= 1.0);
```

### Running Pyramid Verification Gates

Execute individual tiers or the complete 6-tier verification pyramid from the repository root:

```bash
# Run individual verification tiers
cargo run -p xtask -- verify-tier1
cargo run -p xtask -- verify-tier3
cargo run -p xtask -- verify-tier5
cargo run -p xtask -- verify-tier6

# Run the complete 6-tier scientific verification pyramid
cargo run -p xtask -- verify-pyramid
```

---

## 3. Ecosystem Integration

`scientific_verification` feeds live solver counts, interval enclosure bounds, per-tier check counts, and Lean 4 SHA-256 proof digests into `Verification_Certificate.json`, `.github/workflows/ci.yml`, and the interactive Traceability Portal in `math_explorer_gui`.
