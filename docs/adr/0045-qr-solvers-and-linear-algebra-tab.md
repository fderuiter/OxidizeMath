# ADR 0045: Householder QR / Thin-SVD Linear Solvers and Linear Algebra Conditioning Workbench

## Status

Accepted (2026-11-23)

## Context

Previously, [`crates/pure_math/src/pure_math/algebra/linear_algebra/numerical.rs`](../../crates/pure_math/src/pure_math/algebra/linear_algebra/numerical.rs) and [`crates/domain_applied/src/applied/freesurfer/glm.rs`](../../crates/domain_applied/src/applied/freesurfer/glm.rs) solved ordinary least-squares and neuroimaging general linear models by explicitly forming the normal-equations Gram matrix $A^T A$ (or $X^T X$), squaring the condition number $\kappa_2(A)^2$ and losing precision or falsely failing on ill-conditioned Läuchli and high-degree Vandermonde designs with $\kappa_2(A) \in [10^8, 10^{10}]$. In addition, [`moore_penrose_pseudoinverse`](../../crates/pure_math/src/pure_math/algebra/linear_algebra/numerical.rs) and [`low_rank_approximation`](../../crates/pure_math/src/pure_math/algebra/linear_algebra/decomposition.rs) called unbounded SVD routines (`SVD::new`) or `.expect()` on singular-value factors, and `math_explorer_gui` lacked a dedicated Pure Math Linear Algebra tab exposing matrix conditioning, factorizations, and constrained solvers.

## Decision

1. **QR-First Least Squares, Bounded Thin-SVD Fallbacks, and Structured Errors in `pure_math`:**
   - Extended `LinearAlgebraError` in [`numerical.rs`](../../crates/pure_math/src/pure_math/algebra/linear_algebra/numerical.rs) with `NonFiniteInput`, `ConvergenceFailed`, and `NotPositiveDefinite`, validating finite and non-empty inputs across linear-algebra entry points.
   - Rewrote `solve_normal_equation` to factor $A = Q R$ via Householder QR when $m \ge n$ and $|R_{ii}| > \max(m, n) \cdot \varepsilon \cdot \max_j |R_{jj}|$, falling back to bounded thin SVD (`SVD::try_new` with `100 * max(m, n)` iterations) for rank-deficient or underdetermined ($m < n$) systems while preserving its public signature and downstream [`multivariate_linear_regression`](../../crates/pure_math/src/pure_math/statistics/regression.rs) callers.
   - Updated `moore_penrose_pseudoinverse` and `low_rank_approximation` to return `Result<DMatrix<f64>, LinearAlgebraError>` without panicking `.expect()` calls or unbounded SVD loops.
   - Added `solve_ridge_regression` (augmented Householder QR on $[A; \sqrt{\lambda} I_n]$ avoiding $A^T A + \lambda I$) and `solve_nnls` (Lawson-Hanson active-set non-negative least squares with bounded outer iterations).

2. **Factorizations, Norms, Conditioning Metrics, and QR-Backed FreeSurfer GLM:**
   - Added `MatrixNorm` (`One`, `Infinity`, `Frobenius`, `Spectral`), `matrix_norm`, `qr_decomposition`, `cholesky_decomposition`, `svd_decomposition`, `condition_number`, `QrFactors`, and `SvdFactors` in [`decomposition.rs`](../../crates/pure_math/src/pure_math/algebra/linear_algebra/decomposition.rs) and re-exported them from [`linear_algebra/mod.rs`](../../crates/pure_math/src/pure_math/algebra/linear_algebra/mod.rs).
   - Rewrote `estimate_beta` and `t_statistic` in [`glm.rs`](../../crates/domain_applied/src/applied/freesurfer/glm.rs) to solve $R \hat{\beta} = Q^T Y$ via Householder QR and evaluate contrast variance scaling $c^T (X^T X)^{-1} c = \|R^{-T} c\|_2^2$ via forward substitution on $R^T w = c$ without forming $X^T X$.

3. **Auto-Discovered `LinearAlgebraTab` and `ConditioningWorkbenchTool`:**
   - Added [`math_explorer_gui/src/tabs/linear_algebra/mod.rs`](../../math_explorer_gui/src/tabs/linear_algebra/mod.rs) (`// @explorer_feature = "pure_math"`, `#tab=linear_algebra` slug, `"Pure Math"` academic domain) registering `ConditioningWorkbenchTool` (`"LinearAlgebraConditioningTool"`) and four starter gallery presets covering Läuchli catastrophic cancellation, Vandermonde polynomial regression, collinear Ridge stabilization, and non-negative spectral unmixing.

## Consequences

- Ordinary least-squares, multivariate regression, and FreeSurfer GLM contrasts remain accurate on ill-conditioned systems up to $\kappa_2(A) \approx 10^8$ where explicit $A^T A$ fails in IEEE-754 `f64`.
- All fallible SVD, pseudoinverse, Cholesky, Ridge, and NNLS routines return typed `LinearAlgebraError` values rather than panicking or spinning indefinitely on non-finite or non-convergent inputs.
- Users can interactively compare Normal Equations ($A^T A$), Householder QR, Thin SVD, Augmented-QR Ridge, and Active-Set NNLS alongside $\log_{10}(\sigma_i)$ singular-value decay on desktop and `wasm32-unknown-unknown`.
