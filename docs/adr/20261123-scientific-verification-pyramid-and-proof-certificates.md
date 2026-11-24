# ADR 20261123: Six-Tier Scientific Verification Pyramid, Outward-Rounded Interval Enclosures, and Proof Certificates

## Status

Accepted

## Context

OxidizeMath spans 51 differential, quadrature, optimization, stochastic, and domain PDE/ODE solvers across eight mathematical crates alongside special functions, stochastic processes, and academic traceability metadata. Previously:

1. AST classification recognized only two verification tiers (`Empirical` via `theory_verification!` and `Stochastic` via `stochastic_signature_verification!`) and lacked uniform Method of Manufactured Solutions (MMS) order-of-accuracy gates across all 51 solvers.
2. `bessel_j` and `bessel_y` truncated their power series at 50 terms, suffering catastrophic cancellation for large arguments such as $x = 40.0$ (`-12.3658` instead of `0.00736698...`).
3. `SIRModel::step`, `SEIRModel::step`, and `StandardSolverAdapter::step` passed hardcoded `t = 0.0` into time-dependent RHS evaluators, preventing non-autonomous MMS forcing without heap allocation.
4. Scalar domain wrappers in `math_commons::primitives` lacked companion IEEE-754 outward-rounded interval enclosures, and `Verification_Certificate.json` did not bind solver certificates, interval widths, or formal Lean 4 proof digests.

## Decision

1. **Host-Only `crates/scientific_verification` Crate**: Introduce a dedicated host-only workspace crate implementing all six tiers of the scientific verification pyramid while keeping `proptest` and heavy verification harnesses out of `wasm32` runtime bundles:
   - `Tier1_Convergence` (`mms_convergence_verification!`): Certifies all 51 numerical solvers via log-log OLS convergence order checks and zero-heap-allocation assertions.
   - `Tier2_Property` (`property_invariant_verification!`): Runs ChaCha-seeded deterministic `proptest` suites across copulas, Crank-Nicolson quantum propagators, `LatticeBoltzmann` D2Q9 grids, Kaplan-Meier survival curves, and special-function identities.
   - `Tier3_Golden` (`golden_fixture_verification!`): Evaluates special functions and copulas against offline 150-digit golden JSON fixtures (`< 1e-12`), supported by three-regime Maclaurin + Miller/Wronskian + Hankel asymptotic `bessel_j`/`bessel_y` evaluation and offline generator scripts under `scripts/fixtures/`.
   - `Tier4_Stochastic` (`stochastic_signature_verification!`): Validates deterministic ChaCha-seeded Ornstein-Uhlenbeck (Euler-Maruyama & exact Gaussian transition), Poisson arrival, $M/M/1$ CTMC steady-state, and Monte Carlo moment/confidence bounds.
   - `Tier5_Interval` (`interval_enclosure_verification!`): Certifies IEEE-754 outward-rounded `Interval`, `UnitIntervalEnclosure`, `PositiveInterval`, and `NonNegativeInterval` arithmetic in `math_commons::primitives`.
   - `Tier6_Formal` (`formal_proof_verification!`): Commits `sorry`-free Lean 4 proof modules under `proofs/OxidizeMath/` and verifies comment-stripped token hygiene and deterministic FIPS 180-4 SHA-256 digests (invoking `lake build` when Lean 4 is installed).
2. **Zero-Allocation Time Propagation**: Thread simulation time `t` through `TimeStepper` (`OdeModel`), `SIRModel`, `SEIRModel`, `ReactionKinetics::reaction_at_time`, `ReactionDiffusionSystem`, `TuringSystem`, and `StandardSolverAdapter` with zero heap allocations per step.
3. **AST, CLI, CI, and Portal Integration**: Extend `oxidize_core::ast_visitor::AstVisitor`, `TraceabilityReport`, `VerificationCertificate`, `unified_verification`, `xtask`, `.github/workflows/ci.yml`, and `math_explorer_gui`'s Traceability Portal to classify, gate, serialize, and display `tier_counts`, `certified_solvers`, `max_interval_width`, `interval_bounds`, and `lean4_proof_digests` while preserving ADR 0044 build-time manifest parity.

## Consequences

- Every numerical solver, special function, interval kernel, and Lean 4 proof certificate is continuously gated via `cargo run -p xtask -- verify-pyramid` and CI `deploy-pages` schema checks.
- Runtime `wasm32` bundle budgets and zero-allocation simulation stepping invariants remain intact.
