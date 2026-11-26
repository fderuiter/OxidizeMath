# ADR 20261124: Promote `GrayScottKinetics` and 2D CFL Stepping Helpers in `domain_biology`

## Status
Accepted (extends [ADR 0013](0013-scientific-metadata-documentation-standard.md), [ADR 0021](0021-precompute-stencil-reciprocals.md), and [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md))

## Context
1. **Verified `GrayScottKinetics` & Shared `ReactionModel` Helpers**: [`GrayScottKinetics`](../../crates/03_domains/domain_biology/src/biology/morphogenesis/reaction.rs) and [`SchnakenbergKinetics`](../../crates/03_domains/domain_biology/src/biology/morphogenesis/reaction.rs) share identical 2-species slice and batch [`ReactionModel`](../../crates/03_domains/domain_biology/src/biology/reaction_diffusion/model.rs) loops in [`crates/domain_biology/src/biology/morphogenesis/reaction.rs`](../../crates/03_domains/domain_biology/src/biology/morphogenesis/reaction.rs) and are publicly re-exported through [`math_explorer::biology::morphogenesis`](../../crates/04_bindings/math_explorer/src/lib.rs).
2. **Zero-Allocation Dual-Kinetics Stepping**: [`TuringSystem`](../../crates/03_domains/domain_biology/src/biology/morphogenesis/mod.rs) pre-allocates double-buffered [`TuringState`](../../crates/03_domains/domain_biology/src/biology/morphogenesis/state.rs) concentration vectors (`state` and `next_state`). Stepping a pre-allocated `TuringSystem` with an alternate stack-inlined `ReactionKinetics<N>` value without dynamic dispatch (`dyn ReactionKinetics<2>`) or reallocating `TuringState` buffers under [`VerifiedAllocator`](../../crates/00_foundation/verified_engine/src/allocator.rs) requires an in-place generic step entry point.
3. **Explicit 2D Forward Euler CFL Bound**: [`FusedEulerSolver`](../../crates/03_domains/domain_biology/src/biology/morphogenesis/solvers.rs) fuses 5-point 2D Laplacian stencils with local reaction kinetics via Forward Euler integration, which requires $\Delta t \le \frac{\Delta x^2}{4 \max(D_u, D_v)}$ for numerical stability on isotropic 2D grids. Callers lacked a verified helper to compute the $0.8$-damped 2D CFL bound and multi-substep count.

## Decision
1. **Deduplicate `ReactionModel` 2D Batching & Harden `GrayScottKinetics` Verification**:
   - Factor shared `reaction_2d_slice` and `add_reaction_batch_2d` helpers in [`crates/domain_biology/src/biology/morphogenesis/reaction.rs`](../../crates/03_domains/domain_biology/src/biology/morphogenesis/reaction.rs) and verify [`GrayScottKinetics`](../../crates/03_domains/domain_biology/src/biology/morphogenesis/reaction.rs) equilibrium, autocatalysis, and [`math_explorer::biology::morphogenesis`](../../crates/04_bindings/oxidizemath/tests/facade_exports.rs) facade reachability.
2. **Zero-Allocation `TuringSystem::step_with_kinetics`**:
   - Add `TuringSystem::step_with_kinetics<KAlt: ReactionKinetics<N>>(&mut self, kinetics: &KAlt, dt: f64)` sharing `step_bufs` and pre-allocated `state`/`next_state` buffers with `TuringSystem::step`, preserving monomorphized stencil fusion and zero heap allocations under `VerifiedAllocator`.
3. **2D CFL Helpers on `FusedEulerSolver`**:
   - Add `FusedEulerSolver::cfl_dt_2d(dx, d_u, d_v)` (`0.8 * dx^2 / (4.0 * max(|d_u|, |d_v|))`) and `FusedEulerSolver::cfl_substeps_2d(target_dt, dx, d_u, d_v, max_substeps)` to [`crates/domain_biology/src/biology/morphogenesis/solvers.rs`](../../crates/03_domains/domain_biology/src/biology/morphogenesis/solvers.rs), and drive optional automatic CFL substepping in [`MorphogenesisTool`](../../apps/math_explorer_gui/src/tabs/morphogenesis/mod.rs).
   - Regenerate [`crates/unified_verification/api_baseline.json`](../../apps/unified_verification/api_baseline.json) to record the expanded public API surface.

## Consequences
- **Positive**: Callers can step any pre-allocated `TuringSystem` with alternate `ReactionKinetics` strategies in zero allocations, while `MorphogenesisTool` pairs a fast analytical sidebar 2D stability minimap with progressive parameter atlases, coupled LBM wind-tunnel advection, and Gray-Scott Mitosis, Coral, and Soliton presets.
- **Positive**: `FusedEulerSolver::cfl_substeps_2d` prevents accidental Forward Euler blowups under high inhibitor diffusivity (`D_v = 100`) while keeping manual unstable `dt` overrides (`auto_dt = false`) available for divergence fault-recovery testing.
- **Neutral**: Existing `TuringSystem::new` and `SchnakenbergKinetics` signatures and numerical outputs remain bit-for-bit unchanged.
