$L_1$-regularized linear least-squares (LASSO) model $\min_{\mathbf{w}} \frac{1}{2n}\|X\mathbf{w} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{w}\|_1$.

Reference: Tibshirani (1996), DOI: 10.1111/j.2517-6161.1996.tb02080.x; Beck & Teboulle (2009), DOI: 10.1137/080716542;
[`docs/math_specs/pure_math.md`](../pure_math.md).

# Stability
Proximal soft-thresholding $S_{\eta \lambda}(z) = \operatorname{sign}(z)\max(|z| - \eta \lambda, 0)$ is non-expansive and converges monotonically for step sizes $\eta \le 1 / L$ where $L = \|X^\top X\|_2 / n$.

# Domain of Validity
Valid for finite design matrices $X \in \mathbb{R}^{n \times p}$, response vectors $\mathbf{y} \in \mathbb{R}^n$ ($n \ge 1$), and regularization penalty $\lambda \ge 0$.
