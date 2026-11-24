Empirical risk minimization interface $\min_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}; X, \mathbf{y})$ coupling models with gradient optimizers.

Reference: Robbins & Monro (1951), DOI: 10.1214/aoms/1177729586;
[`docs/math_specs/pure_math.md`](../pure_math.md).

# Stability
Convergence depends on step-size scheduling and Lipschitz continuity of the model loss gradient $\nabla_{\boldsymbol{\theta}} \mathcal{L}$.

# Domain of Validity
Valid for differentiable parametric models evaluated on non-empty finite feature-target datasets.
