Post-hoc temperature probability recalibrator
$$\hat{\mathbf{p}}(\mathbf{z}) = \operatorname{softmax}(\mathbf{z} / T), \qquad \hat{p}_{\text{bin}}(z) = \sigma(z / T), \qquad T > 0,$$
optimizing inverse temperature $\beta = 1 / T$ over strictly convex held-out negative log-likelihood while preserving top-1 classification rankings.

Reference: Guo, Pleiss, Sun, & Weinberger (2017), *Proc. 34th ICML* (PMLR 70), arXiv:1706.04599;
[`docs/math_specs/pure_math.md`](../pure_math.md).

# Stability
Reparameterizes optimization in $\beta = 1 / T \in \[10^{-3}, 10^3\]$ where binary and multiclass cross-entropy are globally convex with variance Hessian $\operatorname{Var}_p(z) \ge 0$, using max-shifted log-sum-exp softmax evaluation and damped Newton backtracking.

# Domain of Validity
Requires strictly positive `temperature` (`PositiveFloat`), finite non-empty logits, both positive and negative outcomes for `fit_binary`, and `num_classes >= 2` with `logits_flat.len() == targets.len() * num_classes` and in-bounds class indices for `fit_multiclass`.
