# ADR 20261123: Zero-Allocation Subatomic Physics Kernels and Interactive GUI Tab

## Status

Accepted (extends [ADR 20261123](20261123-checked-apis-and-typed-domain-errors.md) and [ADR 20261123](20261123-inventory-tabs-and-scoped-build-scripts.md))

## Context

`crates/domain_physics` previously implemented standalone scalar formulas across `physics::nuclear`, `physics::standard_model`, and `physics::high_energy`, but lacked:

1. Multi-step linear radioactive decay-chain evaluation (`BatemanChain`) and explicit Weizsäcker binding-energy term breakdowns (`LiquidDropBreakdown`) in `nuclear`.
2. Three-flavor PMNS lepton mixing and constant-density Wolfenstein-Mikheyev-Smirnov (MSW) matter propagation in `standard_model::neutrinos`.
3. Homogeneous and spacetime `FourVector` Lorentz boosts plus relativistic Doppler and superluminal apparent-motion factors in `high_energy::observer`.
4. Interactive GUI exposure of `nuclear`, `standard_model`, and special-relativistic `high_energy` models in `math_explorer_gui`.
5. Accurate `traceability.toml` specification links and physical `theory_verification!` assertions in place of placeholder `DUMMY = 1.0` constants and cross-domain paper citations.

## Decision

1. **Zero-Allocation `domain_physics` Kernels (`VerifiedAllocator`)**:
   - `BatemanChain<const N: usize>` (`N <= 8`) evaluates populations and activities entirely on stack arrays `[f64; N]`, combining an entrywise non-negative Jensen uniformization series for $\max_k(\lambda_k t) \le 2$ with a confluent partial-fraction solver guarding degenerate decay constants via `TOLERANCE_STANDARD`.
   - `LiquidDropModel::binding_energy_breakdown` exposes signed `volume`, `surface`, `coulomb`, `asymmetry`, `pairing`, `total`, and `per_nucleon` fields while preserving `BindingEnergyModel::binding_energy`.
   - `standard_model::neutrinos` reuses `flavor::construct_ckm` for `construct_pmns` and propagates constant-density MSW matter oscillations via trace-shifted scaling-and-squaring Taylor exponentiation on stack-allocated `Matrix3<Complex<f64>>`.
   - `high_energy::observer` implements `FourVector::lorentz_boost` using $\frac{\gamma - 1}{\beta^2} = \frac{\gamma^2}{\gamma + 1}$ to eliminate cancellation at $\vec{\beta} = \mathbf{0}$, alongside `relativistic_doppler_factor`, `apparent_transverse_beta`, and `doppler_beaming_factor`.
2. **Static `inventory` `SubatomicPhysicsTab` (`#tab=subatomic`)**:
   - Implemented under [`math_explorer_gui/src/tabs/subatomic/`](../../math_explorer_gui/src/tabs/subatomic/mod.rs), feature-gated via `#[cfg(feature = "physics")] pub mod subatomic;` in [`math_explorer_gui/src/tabs/mod.rs`](../../math_explorer_gui/src/tabs/mod.rs), and registered statically via `inventory::submit! { TabDescriptor { id: "subatomic", order: 19, ... } }` (`academic_domain() -> "Physics"`, `url_slug() -> "subatomic"`).
   - Registers three `SimulationFramework` tools (`NuclearStabilityTool`, `StandardModelTool`, `RelativisticJetTool`) implementing `FiniteStateGuard` and enforcing `URL_JSON_BOUNDS` (4 KiB), with `SemanticPalette` + high-contrast colormap support, AccessKit keyboard-navigable `(Z, N)` chart canvas, snapshot `rollback_step_back` NaN recovery, four `StarterPresetMetadata` cards, and CSV/JSON/PNG/SVG plot exports.
3. **Traceability & Verification Integrity**:
   - Repaired `traceability.toml` entries for `nuclear`, `standard_model`, and `high_energy` to `spec:*` targets while retaining all `.tex` paper links on their true modules so zero papers become orphaned.
   - Replaced `DUMMY = 1.0` in `nuclear`, `standard_model`, and `high_energy` `theory_verification!` blocks with physical constants and exact invariant checks.

## Consequences

- All new kernel functions execute with zero heap allocations across 1,000-step hot loops and new stack-only error paths under `VerifiedAllocator` (`tests/test_subatomic_zero_alloc.rs`).
- Users can interactively explore nuclear stability and decay chains, Standard Model electroweak/QCD/MSW neutrino physics, and relativistic blazar jets via shareable `#tab=subatomic` URLs within the 3 MB compressed WebAssembly size budget.
