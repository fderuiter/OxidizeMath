# ADR 0043: Promote `GrayScottKinetics` and 2D CFL Stepping Helpers in `domain_biology`

## Status
Accepted (extends [ADR 0013](0013-scientific-metadata-documentation-standard.md), [ADR 0021](0021-precompute-stencil-reciprocals.md), and [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md))

## Context
1. **Unexported `GrayScottKinetics` Strategy**: [`crates/domain_biology/README.md`](../../crates/domain_biology/README.md) documents Gray-Scott reaction-diffusion pattern generation alongside Turing morphogenesis, while [`GrayScottKinetics`](../../crates/domain_biology/src/biology/morphogenesis/reaction.rs) previously existed only as a private struct inside [`crates/domain_biology/tests/test_turing_regression.rs`](../../crates/domain_biology/tests/test_turing_regression.rs). Downstream consumers of [`math_explorer::biology::morphogenesis`](../../math_explorer/src/lib.rs) and [`math_explorer_gui`](../../math_explorer_gui/src/tabs/morphogenesis/mod.rs) could only access [`SchnakenbergKinetics`](../../crates/domain_biology/src/biology/morphogenesis/reaction.rs).
2. **Zero-Allocation Dual-Kinetics Stepping**: [`TuringSystem`](../../crates/domain_biology/src/biology/morphogenesis/mod.rs) pre-allocates double-buffered [`TuringState`](../../crates/domain_biology/src/biology/morphogenesis/state.rs) concentration vectors (`state` and `next_state`). Switching between `SchnakenbergKinetics` and `GrayScottKinetics` at runtime without trait-object dynamic dispatch (`dyn ReactionKinetics<2>`) or reallocating `TuringState` buffers under [`VerifiedAllocator`](../../crates/verified_engine/src/allocator.rs) requires an in-place step entry point parameterized over an alternate stack-inlined `ReactionKinetics<N>` value.
3. **Explicit 2D Forward Euler CFL Bound**: [`FusedEulerSolver`](../../crates/domain_biology/src/biology/morphogenesis/solvers.rs) fuses 5-point 2D Laplacian stencils with local reaction kinetics via Forward Euler integration, which requires $\Delta t \le \frac{\Delta x^2}{4 \max(D_u, D_v)}$ for numerical stability on isotropic 2D grids. Callers lacked a verified helper to compute the $0.8$-damped 2D CFL bound and multi-substep count.

## Decision
1. **Promote `GrayScottKinetics` to Public `domain_biology` API**:
   - Promote [`GrayScottKinetics`](../../crates/domain_biology/src/biology/morphogenesis/reaction.rs) in [`crates/domain_biology/src/biology/morphogenesis/reaction.rs`](../../crates/domain_biology/src/biology/morphogenesis/reaction.rs) defining `GrayScottKinetics { f, k }` deriving `Theory` (citing Pearson 1993) and implementing `ReactionKinetics<2>` and `ReactionModel`, re-exported from [`crates/domain_biology/src/biology/morphogenesis/mod.rs`](../../crates/domain_biology/src/biology/morphogenesis/mod.rs) and reachable via `math_explorer::biology::morphogenesis::GrayScottKinetics`.
2. **Zero-Allocation `TuringSystem::step_with_kinetics`**:
   - Add `TuringSystem::step_with_kinetics<K2: ReactionKinetics<N>>(&mut self, kinetics: &K2, dt: f64)` sharing the pre-allocated `state` and `next_state` buffers with `TuringSystem::step`, preserving monomorphized stencil fusion and zero heap allocations under `VerifiedAllocator`.
3. **2D CFL Helpers on `FusedEulerSolver`**:
   - Add `FusedEulerSolver::cfl_dt_2d(dx, d_u, d_v)` (`0.8 * dx^2 / (4.0 * max(d_u, d_v))`) and `FusedEulerSolver::cfl_substeps_2d(target_dt, dx, d_u, d_v, max_substeps)` to [`crates/domain_biology/src/biology/morphogenesis/solvers.rs`](../../crates/domain_biology/src/biology/morphogenesis/solvers.rs).
   - Regenerate [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json) to record the expanded public API surface.

## Consequences
- **Positive**: `GrayScottKinetics` is a first-class verified strategy across `domain_biology`, `math_explorer`, and `MorphogenesisTool`, powering Mitosis, Coral, and Soliton regimes without AST evaluation overhead or heap allocations.
- **Positive**: `FusedEulerSolver::cfl_substeps_2d` prevents accidental Forward Euler blowups under high inhibitor diffusivity (`D_v = 100`) while keeping manual unstable `dt` overrides available for divergence fault-recovery testing.
- **Neutral**: Existing `TuringSystem::new` and `SchnakenbergKinetics` signatures and numerical outputs remain bit-for-bit unchanged.
