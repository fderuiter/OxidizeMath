# ADR 20261126: Tier 0 `math_memory` ByteArena, Optional `alloc` Engines, and Arena-Backed Solver Workspaces

## Status
Accepted (2026-11-26)

## Context

Prior to this decision, zero-allocation memory management across OxidizeMath had four structural limitations:

1. **Missing `#![no_std]` Tier 0 memory substrate:** Workspace crates lacked a shared `#![no_std]`, `#![forbid(unsafe_code)]` byte arena capable of carving aligned heterogeneous slices (`usize`, `u64`, `f32`, `f64`, `Complex<T>`, and boolean flags/bitsets) from a single stack `&mut [u8]` slice or heap buffer.
2. **Mandatory `alloc` and upward Tier 2 coupling in `math_commons`:** `crates/02_engines/math_commons` linked `alloc` unconditionally and depended upward on `engine_linalg` for small curve-fitting normal systems (`k <= 8`) in `fit/lm.rs` and `fit/poly.rs`, trapping `engine_diffeq` and `engine_pde` into transitive `alloc` builds even under `--no-default-features`.
3. **Colliding homogeneous workspace types and multi-`Vec` fragmentation:** Four distinct structs shared the unqualified name `SolverWorkspace`, three `*OwnedWorkspace` wrappers (`engine_linalg::OwnedWorkspace`, `engine_diffeq::OwnedWorkspace`, `engine_pde::PdeOwnedWorkspace`) had zero production callers, and domain/engine `*Workspace*` structs held dozens of separate `Vec` allocations.
4. **Per-call `vec!` scratch in verified solvers:** Twelve verified linear algebra, Gaussian copula, neuroimaging GLM, and radar surface-fitting entry points allocated temporary `vec!` scratch buffers on every invocation instead of accepting a reusable `SolverWorkspace`.

## Decision

1. **Create Tier 0 `#![no_std]` crate `crates/00_foundation/math_memory`:**
   - Compile with `#![no_std]`, `#![forbid(unsafe_code)]`, `default = []`, and optional `alloc`, `std`, and `serde` features, depending only on `bytemuck`, `num-complex`, and Tier 0 `math_diagnostics`.
   - Provide `ByteArena<'a>`, `ByteCursor<'a>`, and `ArenaCheckpoint` with checked arithmetic (`checked_mul` / `checked_add`), power-of-two alignment rounding up to 16 bytes, scoped `reborrow` cursor restoration, and typed `ArenaError` diagnostics (`MEM-001`..`MEM-007`).
   - Provide safe boolean views `FlagSlice<'a>` (`&mut [u8]`) and `BitSlice<'a>` (`&mut [u64]`) without raw `&mut [bool]` byte reinterpretation, plus `ArenaLayout` / `ArenaRegion<T>` / `RawByteCursor<'a>` for persistent multi-region workspaces.
   - Define the `SolverWorkspace` trait (`byte_capacity`, `remaining_bytes`, `reset`, `reborrow`, `cursor`, `alloc_slice`) and `alloc`-gated `u128`-aligned `OwnedByteArena`.

2. **Decouple `math_commons` from `engine_linalg` and make `alloc` optional across solver engines:**
   - Solve bounded curve-fit linear systems (`k <= 8`) directly on the stack in `math_commons::fit::lm` (damped Cholesky $LDL^\top$ / Gaussian elimination) and `math_commons::fit::poly` (pivoted stack Householder QR) and remove `engine_linalg` from `math_commons/Cargo.toml`.
   - Gate `alloc`-requiring modules in `math_commons` and propagate optional `alloc` features from `engine_diffeq` and `engine_pde` to `math_commons` and `math_memory` so `math_memory`, `math_commons`, `engine_linalg`, `engine_diffeq`, and `engine_pde` compile cleanly with `--no-default-features` and zero `alloc` dependencies.

3. **Unify workspace naming, re-back `*Workspace*` structs onto `OwnedByteArena`, and add `*_with_workspace` solvers:**
   - Retire the 3 unused `*OwnedWorkspace` wrappers, disambiguate the 4 former `SolverWorkspace` structs as `LinalgWorkspace<'a, T>`, `ScalarWorkspace<'a, T>`, `StochasticWorkspace`, and `TridiagonalWorkspace`, and re-back multi-buffer `*Workspace*` structs across `math_commons`, `pure_math`, `domain_applied`, `domain_ai`, and `domain_physics` onto single `OwnedByteArena` allocations.
   - Add zero-allocation `*_with_workspace` entry points across `pure_math::algebra::linear_algebra::{numerical, decomposition}`, `pure_math::statistics::copula::CorrelationMatrix::new_with_workspace`, `domain_applied::applied::freesurfer::glm::{estimate_beta_with_workspace, t_statistic_with_workspace}`, and `domain_physics::physics::medical::radar_gating::surface::BiQuadraticSurface::fit_with_workspace`.

## Consequences

- **Positive:** Firmware and embedded callers can build `math_memory`, `math_commons`, `engine_linalg`, `engine_diffeq`, and `engine_pde` on `#![no_std]` targets without a heap allocator, wrapping stack `&mut [u8]` buffers in `ByteArena`.
- **Positive:** Domain workspaces consolidate dozens of separate `Vec` heap allocations into single cache-contiguous `OwnedByteArena` buffers, and all 12 upgraded solvers execute with `0` dynamic heap allocations inside `VerifiedEngine::run_verified` under `VerifiedAllocator`.
- **Neutral:** Workspace member count increases from `28` to `29` crates, reflected across `Cargo.toml`, `verification_whitelist.toml`, `traceability.toml`, `xtask`, architecture guides, and `api_baseline.json`.
