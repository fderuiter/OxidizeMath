# ADR 0007: Gaussian Opacity Evaluation Optimization

## Context
In `crates/domain_ai/src/ai/gaussian_splatting/rendering.rs`, `evaluate_gaussian_opacity` is invoked frequently during rasterization iterations. Calculating exponentiation (`power.exp()`) when `power` falls below active opacity limits (or is positive) introduces unnecessary floating-point exponentiation overhead.

## Decision
We optimized `evaluate_gaussian_opacity` by:
1. Adding an early-exit guard clause (`if power > 0.0 || power < -16.0 { return 0.0; }`) to bypass `power.exp()` computation when power is out of active opacity bounds.
2. Annotating `evaluate_gaussian_opacity` with `#[inline]` to eliminate function call overhead during rasterization iterations.

## Consequences
- **Performance**: Significant speedup during 2D Gaussian splatting rasterization.
- **Maintainability**: Pure function behavior, signature, and safety attributes (`#[verified_engine::verified]`) remain preserved.
