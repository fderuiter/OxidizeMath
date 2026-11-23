# `engine_diffeq` — Zero-Allocation `#![no_std]` Differential Equations & Quadrature Engine

## 1. Overview & Architecture

`engine_diffeq` is OxidizeMath's foundational `#![no_std]` numerical engine for ordinary differential equation (ODE) integration, composite and adaptive numerical quadrature, and safeguarded cubic Hermite root-bracketing. It is designed in strict compliance with Rule 3 of the NASA Power of 10 (`docs/architecture/zero_allocation_runtime.md`): every solver step, Newton/LU iteration, quadrature refinement, and root-bracketing loop executes with zero dynamic heap allocations.

Key capabilities include:

- **Caller-Owned & Stack Workspaces (`workspace`)**: `SolverWorkspace` and `StackWorkspace` provide zero-allocation scratch storage for explicit stage vectors, dense finite-difference Jacobians, partial-pivoting LU factorizations, and adaptive quadrature subinterval buffers.
- **Explicit, Adaptive & Stiff ODE Solvers (`ode`)**:
  - Explicit fixed-step solvers: `Euler`, `Heun` (RK2), and `RungeKutta4` (classic RK4).
  - Embedded adaptive solver: `DormandPrince45` (`RK45` 7-stage FSAL pair with proportional-integral step-size control).
  - Stiff implicit solvers: L-stable `Rosenbrock23` (`ode23s` Rosenbrock-Wanner method with non-autonomous $\partial f / \partial t$ coupling) and variable-step two-step `Bdf2` (bootstrapped via L-stable second-order TR-BDF2 and damped Newton-LU iterations).
  - Unified `TimeStepper` (with defaulted `get_time` / `set_time` accessors) and `OdeModel` tracking elapsed time `t` accurately across non-autonomous systems.
- **Composite & Adaptive Quadrature (`quadrature`)**:
  - `Trapezoidal` with midpoint-reusing Richardson error bounds, optional `eps > 0` refinement, and shared uniform/non-uniform discrete grid kernels.
  - `Simpson` composite 1/3 rule with fourth-order Richardson error estimation and `eps > 0` refinement.
  - `GaussKronrod15` (`G7-K15`) adaptive quadrature operating over a fixed-capacity stack or slice subinterval priority buffer.
- **Safeguarded Cubic Hermite Root-Bracketing (`hermite`)**:
  - Shared `cubic_hermite_scalar`, `cubic_hermite_slope`, Fritsch-Carlson monotone slope clamping, and safeguarded Newton-bisection root-finding over unit brackets `[0, 1]`.

## 2. Developer Instructions

Build and verify the crate in pure `#![no_std]` mode and on WebAssembly:

```sh
cargo check -p engine_diffeq --no-default-features
cargo check -p engine_diffeq --no-default-features --target wasm32-unknown-unknown
cargo test -p engine_diffeq
```

Downstream crates that use `math_explorer` or `pure_math` access all `engine_diffeq` solvers through `math_explorer::diffeq` and `math_explorer::pure_math::analysis`. Direct `#![no_std]` consumers can depend on `engine_diffeq` with `default-features = false` and supply stack or slice `SolverWorkspace` buffers directly.
