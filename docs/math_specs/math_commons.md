# Canonical Mathematical Specification: `math_commons`

Canonical governing equations, bibliography references, stability bounds, and validity domains for [`crates/math_commons`](../../crates/02_engines/math_commons/README.md), linked to [`papers/references.bib`](../../papers/references.bib) and [`papers/essay.tex`](../../papers/essay.tex).

## 1. Polynomial & Nonlinear Curve Fitting (`fit`)

### Governing Equations

Scaled polynomial ordinary least squares maps sample abscissae $x_i$ to centered normalized coordinates $u_i = (x_i - \mu_x)/\sigma_x$ and solves the normal system $(V^T V) c = V^T y$ for degree $1 \le d \le 4$:

$$
\hat{y}(x) = \sum_{k=0}^{d} c_k x^k, \qquad \min_{\theta \in \mathbb{R}^p} \sum_{i=1}^{N} \bigl(y_i - f(x_i; \theta)\bigr)^2
$$

Nonlinear [`FitModel`](../../crates/02_engines/math_commons/src/fit/mod.rs) presets (`Exponential`, `PowerLaw`, `Logarithmic`, `Logistic`, `Sinusoidal`) and arbitrary scalar callbacks iterate damped Levenberg-Marquardt normal equations on fixed stack buffers (`k <= 6`):

$$
\bigl(J(\theta_t)^T J(\theta_t) + \lambda_t \operatorname{diag}(J^T J + I)\bigr)\,\Delta\theta_t = J(\theta_t)^T r(\theta_t)
$$

### Citations

- BibTeX keys: [`nalgebra`](../../papers/references.bib#nalgebra), [`boisvert1997`](../../papers/references.bib#boisvert1997), [`kammler2007first`](../../papers/references.bib#kammler2007first), [`Yuval2021UsePrecision`](../../papers/references.bib#Yuval2021UsePrecision) (DOI: `10.1029/2020GL091363`).
- Manuscript: [`papers/essay.tex`](../../papers/essay.tex).

### Stability

Standardizing abscissae before forming Vandermonde moments bounds polynomial condition numbers for $d \le 4$, while Marquardt multiplicative diagonal damping ($\lambda_t \in [10^{-12}, 10^{12}]$) with partial-pivoted Gaussian elimination (`solve_dense_small`) prevents singular Jacobian steps without heap allocation.

### Domain of Validity

Requires at least $p$ finite $(x_i, y_i)$ observations ($p = d + 1$ for degree-$d$ polynomials, $1 \le k \le 6$ for `levenberg_marquardt_numeric`), non-zero abscissa spread ($\sigma_x > 0$), $x_i > 0$ for logarithmic and power-law models, and $y_i > 0$ for exponential/power-law log-linear seeding.

```rust
use math_explorer::fit::{FitModel, fit_model};

let pts = [[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 7.0]];
let fit = fit_model(&pts[..], FitModel::Linear, false).unwrap();
assert!((fit.r_squared - 1.0).abs() < 1e-12);
assert!((fit.eval(4.0) - 9.0).abs() < 1e-9);
```

## 2. Wigner Coupling & Stack AST Kernels (`wigner`, `ast`, `poi`)

### Governing Equations

Clebsch-Gordan coefficients $\langle j_1 m_1 j_2 m_2 \mid J M \rangle$ satisfy the Racah alternating factorial sum subject to selection rules $m_1 + m_2 = M$ and $|j_1 - j_2| \le J \le j_1 + j_2$:

$$
\langle j_1 m_1 j_2 m_2 \mid J M \rangle = \delta_{m_1 + m_2, M} \sqrt{2J + 1}\,\Delta(j_1, j_2, J) \sum_{k} \frac{(-1)^k}{k!\,\Gamma_k(j_1, j_2, J, m_1, m_2)}
$$

### Citations

- BibTeX keys: [`kammler2007first`](../../papers/references.bib#kammler2007first), [`pushpa2022eisenstein`](../../papers/references.bib#pushpa2022eisenstein) (DOI: `10.1007/s40065-022-00360-6`).
- Manuscript: [`papers/quantum_mechanics.tex`](../../papers/quantum_mechanics.tex).

### Stability

Log-factorial/exact integer ratio evaluation bounds alternating Racah cancellation for low-to-moderate half-integer angular momenta, and compiled RPN bytecode (`CompiledExpr`) evaluates on a fixed-depth stack without recursion.

### Domain of Validity

Angular momentum quantum numbers $2j_1, 2j_2, 2J \in \mathbb{Z}_{\ge 0}$ must satisfy triangle parity $(j_1 + j_2 - J) \in \mathbb{Z}_{\ge 0}$ and $|m_i| \le j_i$; out-of-triangle configurations evaluate identically to $0.0$.
