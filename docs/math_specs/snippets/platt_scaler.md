Logistic Platt post-hoc probability recalibrator
$$\hat{p}(y = 1 \mid z) = \sigma(a z + b) = \frac{1}{1 + \exp(-(a z + b))},$$
fitting slope $a = \text{scale}$ and intercept $b = \text{bias}$ by minimizing $L_2$-regularized held-out binary cross-entropy via zero-allocation damped $2 \times 2$ Newton-Raphson with backtracking line search.

Reference: Platt (1999), *Advances in Large Margin Classifiers*, MIT Press;
Lin, Lin, & Weng (2007), *Mach. Learn.* 68(3), DOI: 10.1007/s10994-007-5018-6;
[`docs/math_specs/pure_math.md`](../pure_math.md).

# Stability
Evaluates sigmoid probabilities via branch-split exponentials and softplus cross-entropy without overflow, floors per-sample Fisher weights at $10^{-12}$, and applies adaptive Levenberg-Marquardt Hessian damping with determinant guard $\det(H + \lambda I) > 10^{-18}$ and Armijo backtracking.

# Domain of Validity
Requires non-empty matching `logits` and `outcomes` slices containing finite logits and at least one positive and one negative outcome, `max_iter >= 1`, `tol > 0.0`, and `l2_reg >= 0.0` (`MathError`).
