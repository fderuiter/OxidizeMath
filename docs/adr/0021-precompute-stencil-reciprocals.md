# 21. Precompute Reciprocal Grid Spacings in StencilOperators

## Status
Accepted

## Context
In `crates/pure_math/src/pure_math/analysis/pde/fused_stepper.rs`, the spatial stencil operators (`central_diff_1st`, `central_diff_2nd`, `central_diff_2nd_2d`, and `upwind_flux`) previously computed floating-point divisions on every single invocation.
During 2D reaction-diffusion simulations in `crates/domain_biology/src/biology/diffusion/fd2d/mod.rs`, the fused spatial loops over grid nodes evaluate spatial derivatives hundreds of thousands of times per time step (e.g. over 360,000 divisions per step on a $300 \times 300$ grid with coupled reacting species).
Repeated floating-point divisions inside hot inner stencil loops degrade stepping throughput and benchmark execution time.

## Decision
1. Precomputed inverse grid spacings in `StencilOperators` during struct initialization:
   - `inv_dx = 1.0 / *dx`
   - `inv_two_dx = 1.0 / (2.0 * *dx)`
   - `inv_dx_sq = 1.0 / (*dx * *dx)`
   - `inv_dy_sq = 1.0 / (*dy * *dy)`
2. Validated in `new` and `new_2d` that step sizes `dx` and `dy` are strictly positive to prevent division by zero or invalid reciprocal factors.
3. Updated stencil methods to multiply by the precomputed reciprocals instead of dividing:
   - `central_diff_1st` multiplies by `self.inv_two_dx`.
   - `central_diff_2nd` multiplies by `self.inv_dx_sq`.
   - `central_diff_2nd_2d` multiplies by `self.inv_dx_sq` and `self.inv_dy_sq`.
   - `upwind_flux` multiplies by `self.inv_dx`.
4. Maintained identical public API signatures, mathematical behavior, and numerical accuracy across all callers in `pure_math`, `domain_biology`, and `domain_applied`.
5. Added comprehensive unit tests in `fused_stepper.rs` covering reciprocal precomputation, edge-case assertions, and numerical equivalence.

## Consequences
- Eliminates per-node floating-point divisions across all 1D and 2D fused stencil evaluations, replacing them with high-throughput floating-point multiplications.
- Significantly accelerates multi-pass grid updates, including Turing pattern and reaction-diffusion simulations.
- Full numerical equivalence and stability are preserved across all existing mathematical PDE callers.
