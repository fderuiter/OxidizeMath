# `engine_pde` — `#![no_std]` Continuum PDE & Stochastic PDE Solver Engine

## 1. Overview & Architecture

`engine_pde` is OxidizeMath's Tier 2 `#![no_std]` continuum partial differential equation (PDE) and stochastic partial differential equation (SPDE) solver crate. It unifies 1D, 2D, and 3D finite-difference stencils, boundary-value operators, conserved-flux stochastic noise, and radix-2 pseudo-spectral Fourier steppers over [`Scalar`](math_commons::math_kernel::types::Scalar), [`SolverWorkspace`](math_commons::math_kernel::types::SolverWorkspace), and `rand_core::RngCore`.

Core capabilities:
- **Unified Finite-Difference Stencils**: [`StencilOperators`] and [`FusedStencilStepper`] expose 1D, 2D, and 3D central Laplacians, coupled multi-species Neumann and toroidal periodic 2D stencil passes (`step_2d_coupled_neumann`, `step_2d_coupled_periodic`), central gradients, squared gradients ($|\nabla h|^2$), first-order and semi-Lagrangian characteristic upwind operators, and public 2D Biharmonic ($\nabla^4 u$) evaluation across [`BoundaryCondition`] (`Periodic`, `Neumann`, `SymmetricNeumann`, and `Dirichlet`).
- **Explicit & Operator-Split SPDE Steppers**: [`ExplicitSpdeStencilStepper`] and [`SplitStepSpdeStencilStepper`] evolve reaction-diffusion, Allen-Cahn, Edwards-Wilkinson, KPZ, and Cahn-Hilliard fields with additive white noise or staggered-face conserved-flux noise ($\nabla \cdot \vec{\eta}$).
- **Radix-2 Dealiased Spectral Stepper**: [`SemiImplicitFourierStepper`] pairs a zero-allocation `#![no_std]` Cooley-Tukey 1D/2D FFT with Orszag 2/3-rule dealiasing for stiff fourth-order Cahn-Hilliard and nonlinear Kardar-Parisi-Zhang SPDEs.

```rust
use engine_pde::{BoundaryCondition, StencilOperators};
use math_commons::math_kernel::types::StepSize;

let ops = StencilOperators::new_2d(StepSize(0.25), StepSize(0.25));
let field = [0.0_f64; 16];
let biharm = ops.eval_biharmonic_2d(&field, 4, 4, 1, 1, BoundaryCondition::Periodic);
assert_eq!(biharm, 0.0);
```

## 2. Developer Instructions

Verify `#![no_std]`, `alloc`, native `std`, and WebAssembly builds from the workspace root:

```bash
cargo check -p engine_pde --no-default-features
cargo check -p engine_pde --no-default-features --features alloc
cargo check -p engine_pde --target wasm32-unknown-unknown --all-features
cargo test -p engine_pde
```
