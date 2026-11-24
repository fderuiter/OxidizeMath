# ADR 20261123: `#![no_std]` Tier 2 `engine_diffeq` & `engine_pde` Solvers and Continuum Stat-Mech SPDEs

## Status

Accepted

## Context

Prior to this decision, 0D stochastic differential equation (SDE) integrators, underdamped Langevin splittings, finite-difference spatial stencils, and continuum stochastic partial differential equation (SPDE) kernels were fragmented across multiple Tier 3/4 domain crates (`pure_math`, `domain_biology`, `domain_applied`, `domain_ai`, and `domain_physics`):

1. **Duplicated 0D Stochastic Integrators**: Ornstein-Uhlenbeck Euler-Maruyama stepping (`pure_math::statistics::ou_process`), 2D BAOAB Langevin momentum dynamics (`pure_math::analysis::momentum_dynamics`), Gillespie jump sampling (`pure_math::analysis::stochastic`), and score-distillation Gaussian diffusion (`domain_ai::ai::sds::diffusion`) each maintained ad-hoc Gaussian sampling and time-stepping loops.
2. **Siloed PDE Stencils and Private Biharmonic Operators**: 1D/2D/3D finite-difference operators lived inside `pure_math::analysis::pde::fused_stepper`, 2D Biharmonic ($\nabla^4 u$) evaluation was private to Kirchhoff-Love plate bending in `wave_2d.rs`, and semi-Lagrangian characteristic upwind advection was bespoke to `domain_applied::applied::cannibalism::death_rate`.
3. **Missing `#![no_std]` Tier 0/2 Foundations and Continuum Stat-Mech SPDEs**: `verified_engine` and `math_commons` assumed `std` unconditionally, preventing bare-metal `#![no_std]` compilation of core numerical steppers, while `domain_physics::physics::stat_mech` lacked verified continuum SPDE solvers for Edwards-Wilkinson, Kardar-Parisi-Zhang (KPZ), Allen-Cahn, and conserved Cahn-Hilliard dynamics.

## Decision

1. **`#![no_std]` Tier 0 Kernel Primitives ([`math_commons`](../../crates/math_commons/src/math_kernel/types.rs) & [`verified_engine`](../../crates/verified_engine/src/metrics.rs))**:
   - Gate thread-local runtime metrics and `VerifiedAllocator` in `verified_engine` behind `feature = "std"` with zero-cost `#![no_std]` no-op shims when `std` is disabled.
   - Make `math_commons::math_kernel::types` compile under `#![no_std]` via `libm` and `rand_core` (`default-features = false`), introducing the [`Scalar`](../../crates/math_commons/src/math_kernel/types.rs) trait (`f32`, `f64`), zero-allocation borrowed [`SolverWorkspace`](../../crates/math_commons/src/math_kernel/types.rs) bump-slice arenas, and deterministic Box-Muller Gaussian sampling (`sample_open_unit`, `sample_standard_normal_pair`, `fill_standard_normal`) over `RngCore`.

2. **Tier 2 `#![no_std]` Differential Equation Engine ([`crates/engine_diffeq`](../../crates/engine_diffeq/README.md))**:
   - Provide multivariate Itô and Stratonovich integrators over `Scalar`, `RngCore`, and `SolverWorkspace`: `EulerMaruyamaSolver` (strong order $0.5$ / Euler-Heun), `MilsteinSolver` (strong order $1.0$ with optional truncated Karhunen-Loève Lévy area), derivative-free Rössler `Sriw1Solver` (strong order $1.5$), symmetric Leimkuhler-Matthews `BaoabLangevinSolver`, and discrete-event Gillespie `ssa_sample_event`.

3. **Tier 2 `#![no_std]` Continuum PDE & SPDE Engine ([`crates/engine_pde`](../../crates/engine_pde/README.md))**:
   - Consolidate `BoundaryCondition` (`Periodic`, `Neumann`, `SymmetricNeumann`, `Dirichlet`) and `StencilOperators` (1D/2D/3D Laplacians, central gradients, $|\nabla u|^2$, directional and characteristic upwind advection, and public 2D Biharmonic `eval_biharmonic_2d_from_laplacian` / `eval_biharmonic_2d`).
   - Implement `ExplicitSpdeStencilStepper` and Strang-split `SplitStepSpdeStencilStepper` with additive white noise and staggered-face conserved-flux noise ($\nabla \cdot \vec{\eta}$) preserving spatial mean to machine precision.
   - Implement `SemiImplicitFourierStepper` pairing a zero-allocation `#![no_std]` radix-2 1D/2D Cooley-Tukey FFT with Orszag 2/3-rule dealiasing for Cahn-Hilliard, KPZ, and Edwards-Wilkinson fields.

4. **Domain Delegations & Verified Continuum Statistical Mechanics ([`crates/domain_physics`](../../crates/domain_physics/src/physics/stat_mech/mod.rs))**:
   - Delegate existing domain callers in `pure_math` (`fused_stepper`, `wave_2d`, `ou_process`, `momentum_dynamics`, `stochastic`), `domain_biology` (`morphogenesis::solvers::FusedEulerEvolution` forwarding `rng`), `domain_applied` (`cannibalism::{mckendrick_von_foerster, death_rate}`), and `domain_ai` (`sds::diffusion`) to `engine_diffeq` and `engine_pde` without breaking public facade signatures.
   - Implement verified zero-allocation `EdwardsWilkinson1D`/`EdwardsWilkinson2D`, `KpzModel1D`/`KpzModel2D`, `AllenCahn2D`, and `CahnHilliard2D` solvers in `domain_physics::physics::stat_mech` with real `theory_verification!` checks and updated `traceability.toml` specification links.

## Consequences

- **Positive**: `engine_diffeq` and `engine_pde` compile cleanly under `--no-default-features` (`#![no_std]`), `--no-default-features --features alloc`, native `std`, and `wasm32-unknown-unknown` while executing stepping loops with zero heap allocations under `VerifiedAllocator`.
- **Positive**: Both Kirchhoff-Love plate bending (`wave_2d.rs`) and conserved 2D Cahn-Hilliard solvers share a single verified Biharmonic operator on `StencilOperators`, and staggered-face conserved-flux noise guarantees discrete mass conservation.
- **Trade-offs**: Workspace member count increases from 20 to 22 crates, reflected in [`system_overview.md`](../architecture/system_overview.md) and [`api_baseline.json`](../../crates/unified_verification/api_baseline.json).
