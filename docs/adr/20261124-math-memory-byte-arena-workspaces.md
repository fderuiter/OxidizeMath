# ADR 20261124: Tier 0 `math_memory` ByteArena, Optional-`alloc` Engines, and Arena-Backed Workspaces

## Status

Accepted

* **Date:** 2026-11-24

## Context

Across OxidizeMath's linear algebra, differential equation, PDE, stochastic, statistical, AI, applied, and physics engines, scratch memory management suffered from four structural bottlenecks:

1. **Fragmented Multi-`Vec` Workspaces**: Domain and pure-math `*Workspace*` structs held 98 separate `Vec` fields (including nested `Vec<Vec<...>>` adjacency and $\mathbb{Z}_2$ basis tables in `TdaFiltrationWorkspace`), causing dozens of independent heap allocations during initialization and preventing uniform scratch buffer sharing.
2. **Duplicate `SolverWorkspace` Types and Unused `*OwnedWorkspace` Wrappers**: Four distinct structs named `SolverWorkspace` (`engine_linalg`, `math_commons::math_kernel::types`, `math_commons::stochastic`, and `domain_physics::quantum::tridiagonal`) collided across imports, while three `*OwnedWorkspace` wrappers (`engine_linalg::OwnedWorkspace`, `engine_diffeq::OwnedWorkspace`, and `engine_pde::PdeOwnedWorkspace`) had zero external callers.
3. **Mandatory `alloc` and Upward Tier Coupling in `math_commons`**: `math_commons` unconditionally imported `extern crate alloc` and depended upward on Tier 2 `engine_linalg` solely to solve bounded (`k <= 8`) normal equations in `fit/lm.rs` and `fit/poly.rs`, blocking pure `core`-only `--no-default-features` builds of `math_commons`, `engine_diffeq`, and `engine_pde`.
4. **Per-Call Scratch Allocations in Verified Linear-Algebra Callers**: Twelve verified entry points across `pure_math` (`numerical.rs`, `decomposition.rs`, `copula/core.rs`), `domain_applied` (`freesurfer/glm.rs`), and `domain_physics` (`radar_gating/surface.rs`) allocated fresh temporary scratch `Vec`s on each call rather than accepting caller-managed arena scratch buffers.

## Decision

1. **Introduce Tier 0 `crates/00_foundation/math_memory`**:
   - Create `#![no_std]` `#![forbid(unsafe_code)]` crate `math_memory` (`default = []`, optional `alloc`, optional `std`) built on `bytemuck` (`Pod + Zeroable`).
   - Implement fallible `ByteArena<'a>` (`alloc_slice`, `retain_slice`, `alloc_flags`, `alloc_bits`, `checkpoint`/`rewind`, and scoped RAII `reborrow` via `ArenaScope`) using checked integer arithmetic (`checked_mul`, `checked_add`) returning `Result<_, ArenaError>` without panicking on overflow or capacity exhaustion.
   - Provide safe boolean scratch views (`FlagSlice<'a>` over `&'a mut [u8]` and packed `BitSlice<'a>` over `&'a mut [u64]`) so raw bytes are never cast to `&mut [bool]`.
   - Define the unified `SolverWorkspace` trait (`as_byte_arena`, `reborrow`, `byte_capacity`, `reset`) and `alloc`-gated `OwnedByteArena` (`ArenaLayout` + typed `ArenaRegion<T>` handles over an 8-byte-aligned `Vec<u64>` backing store).

2. **Decouple `math_commons` from `engine_linalg` and Propagate Optional `alloc`**:
   - Gate `alloc`-dependent modules in `math_commons` (`ast`, `fit` owned convenience APIs, `poi` owned convenience APIs, `registry`, `series` `Vec` impls, `stochastic`, and `String`-backed errors/colormaps) behind `#[cfg(feature = "alloc")]`, solving bounded $k \le 8$ curve-fitting systems on the stack via streaming Givens QR and partial-pivot Gaussian elimination in `fit/poly.rs` and `fit/lm.rs` and removing `engine_linalg` from `math_commons/Cargo.toml`.
   - Propagate optional `alloc` across `engine_diffeq` and `engine_pde` so `math_memory`, `math_commons`, `engine_linalg`, `engine_diffeq`, and `engine_pde` all compile cleanly with `--no-default-features`.

3. **Retire `*OwnedWorkspace`, Disambiguate `SolverWorkspace`, and Re-Back Multi-`Vec` Workspaces**:
   - Delete `engine_linalg::OwnedWorkspace`, `engine_diffeq::OwnedWorkspace`, and `engine_pde::PdeOwnedWorkspace`.
   - Rename `math_commons::stochastic::SolverWorkspace` to `StochasticWorkspace` and `domain_physics::physics::quantum::tridiagonal::SolverWorkspace` to `TridiagonalWorkspace`, retiring `math_commons::math_kernel::types::SolverWorkspace` and `engine_linalg`'s struct wrapper in favor of `math_memory::{ByteArena, SolverWorkspace}`.
   - Re-back all 18 `Vec`-bearing `*Workspace*` structs across `crates/` (eliminating all 98 `Vec` fields and flattening all nested `Vec<Vec<...>>` tables in `TdaFiltrationWorkspace`) onto `OwnedByteArena` and `ArenaScope::retain_slice` split-borrows.

4. **Add Zero-Allocation `*_with_workspace` Companion Entry Points**:
   - Add `*_with_workspace` entry points taking `&mut impl SolverWorkspace` (plus `*_workspace_bytes` sizing helpers where needed) across all 12 verified callers in `pure_math::pure_math::algebra::linear_algebra::{numerical, decomposition}`, `pure_math::pure_math::statistics::copula::core`, `domain_applied::applied::freesurfer::glm`, and `domain_physics::physics::medical::radar_gating::surface`, verified to record 0 heap allocations under `VerifiedEngine::run_verified`.

## Consequences

* **Positive:** `math_memory`, `math_commons`, `engine_linalg`, `engine_diffeq`, and `engine_pde` compile cleanly under `--no-default-features` with zero `unsafe` blocks in `math_memory` and zero upward Tier 1 -> Tier 2 dependencies.
* **Positive:** All `*Workspace*` structs in `crates/` allocate a single 8-byte-aligned `OwnedByteArena` buffer at construction time and execute hot solver loops with 0 dynamic heap allocations under `VerifiedAllocator`.
* **Trade-offs:** Public API additions (`math_memory`, renamed workspace types, and `*_with_workspace` companions) require regenerating `crates/unified_verification/api_baseline.json` and updating workspace crate inventory counts from 23 to 24.
