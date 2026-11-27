# ADR 20261127: Enforce Strict Five-Tier DAG via Kernel Promotion and Single-Domain Feature Isolation

## Status

Accepted

## Context

Prior to this decision, three classes of architectural coupling prevented isolated single-domain compilation and obscured the five-tier runtime hierarchy (`04_bindings -> 03_domains -> 02_engines -> 01_pure -> 00_foundation`):

1. **Sideways Peer-Domain Dependencies (`03_domains -> 03_domains`)**: `domain_physics` and `domain_biology` depended sideways on `domain_applied` for evolutionary game-theory kernels (`FitnessStrategy`, `MatrixPayoff`, `ReplicatorDynamics`, `GameTheoryError`), discrete-time Kalman state estimation (`KalmanModel`, `KalmanSystem`, `KalmanFilter`, `KalmanFilterBuilder`, `KalmanError`), and CSR spatial cortical surface distance primitives (`Surface`, `FlatSpatialGrid`, `SpatialGrid`, `ThicknessWorkspace`, `cortical_thickness*`).
2. **Upward Engine Dependencies from `pure_math` (`01_pure -> 02_engines`)**: `pure_math` depended upward on `engine_linalg`, `engine_diffeq`, and `engine_pde` for `DMatrix`/`DVector` numerical linear algebra wrappers, 2D upwind advection and fused PDE steppers, Gillespie event sampling (`ssa_sample_event`), scalar Ornstein-Uhlenbeck Euler-Maruyama stepping, and Langevin BAOAB integration.
3. **Transitive Cross-Domain Feature Chaining & Missing DAG Gate**: `oxidizemath`, `math_explorer`, and `math_explorer_gui` chained peer domain features (such as `climate -> ai` and cross-wired GUI domain features), causing `--no-default-features --features <domain>` builds to compile unrelated `domain_*` crates, while `unified_verification` (`batch-static` / `batch-staged`) and `xtask` lacked a fail-closed gate rejecting peer-domain, upward-tier, runtime-to-tooling, and multi-domain feature closure edges.

## Decision

1. **Kernel Promotions Across Tiers 0, 1, and 2 (Without Moving Crate Directories)**:
   - Promote `#![no_std]` `LinalgError`, `LinearAlgebraError`, and `mat_idx` to Tier 0 [`math_scalars`](../../crates/00_foundation/math_scalars/README.md).
   - Promote triangulated `Surface`, CSR uniform grids (`FlatSpatialGrid`, `SpatialGrid`, `ThicknessWorkspace`), and `cortical_thickness*` kernels to [`oxidize_core::mesh`](../../crates/01_core/oxidize_core/README.md), preserving `VerifiedAllocator` zero-allocation invariants and re-exporting them in `domain_applied::applied::freesurfer`.
   - Promote `#![no_std]` `ssa_sample_event` to `math_commons::math_kernel::types` (re-exported in `engine_diffeq::jump`) and promote `FitnessStrategy`, `MatrixPayoff`, `ReplicatorDynamics`, and `GameTheoryError` to `pure_math` (`pure_math::game_theory::evolutionary` and `pure_math::error`), re-exporting all items from `domain_applied` for backward compatibility.
   - Promote `KalmanModel`, `KalmanSystem`, `KalmanFilter`, `KalmanFilterBuilder` (under optional `nalgebra`), and `#![no_std]` `KalmanError` to Tier 2 [`engine_diffeq`](../../crates/02_engines/engine_diffeq/README.md) with re-exports in `domain_applied`, removing `domain_applied` completely from `domain_physics` and `domain_biology`.
   - Relocate `nalgebra_ops::{numerical, decomposition}` to Tier 2 [`engine_linalg`](../../crates/02_engines/engine_linalg/README.md) (under optional `nalgebra`) and `#![no_std]` `advection_2d` to Tier 2 [`engine_pde`](../../crates/02_engines/engine_pde/README.md), dropping `engine_linalg`, `engine_diffeq`, and `engine_pde` from `pure_math` while composing full `oxidizemath::pure_math` and `math_explorer::pure_math` re-exports at Tier 4 (`oxidizemath/src/pure_math_facade.rs`).

2. **Independent Single-Domain Cargo Feature Closures**:
   - Untangle domain feature tables in [`oxidizemath`](../../crates/04_bindings/oxidizemath/Cargo.toml), [`math_explorer`](../../crates/04_bindings/math_explorer/Cargo.toml), and [`math_explorer_gui`](../../apps/math_explorer_gui/Cargo.toml) so each single-domain feature (`ai`, `applied`, `biology`, `climate`, `epidemiology`, `physics`, `pure_math`) activates at most one `domain_*` crate.
   - Guard cross-domain GUI workbenches, presets, audio bridges, and tests in `math_explorer_gui` with explicit `#[cfg(all(feature = "...", feature = "..."))]` attributes, and declare multi-domain `required-features` on `oxidize-cli` examples where needed.

3. **Fail-Closed `check-dag` Verification Gate & Topology Sync**:
   - Add `check-dag` (`apps/unified_verification/src/dag_gate.rs`) to `unified_verification` (`check-dag`, `batch-static`, `batch-staged`) and `xtask` (`check-dag`, `lint-staged`, `lint-all`), enforcing `apps -> 04_bindings -> 03_domains -> 02_engines -> 01_pure -> 00_foundation`, zero peer `domain_*` edges, zero runtime-to-tooling edges (exempting `xtask`, `unified_verification`, and `markdown_tests` as sources), and `<= 1` `domain_*` package per single-domain feature closure.
   - Update [`docs/architecture/system_overview.md`](../architecture/system_overview.md) and regenerate [`apps/unified_verification/api_baseline.json`](../../apps/unified_verification/api_baseline.json).

## Consequences

- **Positive**: Downstream library consumers and contributors building `--no-default-features --features <domain>` compile only the requested `domain_*` crate and downward engines/foundations, while keeping `#![no_std]` defaults intact across Tier 0 and Tier 2 engines.
- **Positive**: Pre-commit (`xtask lint-staged`) and CI (`unified_verification batch-static`) immediately block any future sideways peer-domain edge, upward tier edge, runtime-to-tooling edge, or cross-domain feature chain with source/target tier diagnostics and remediation hints.
- **Trade-offs**: Cross-domain GUI tools and CLI examples that combine multiple scientific domains must explicitly declare dual `#[cfg(all(feature = "...", feature = "..."))]` guards or `required-features` lists.
