# Canonical Mathematical Specification: `engine_linalg`

Canonical governing equations, bibliography references, stability bounds, and validity domains for [`crates/02_engines/engine_linalg`](../../crates/02_engines/engine_linalg/README.md), linked to [`papers/references.bib`](../../papers/references.bib) and [`papers/advanced_linear_algebra.tex`](../../papers/advanced_linear_algebra.tex).

## 1. Direct & Orthogonal Factorizations (`lu`, `cholesky`, `qr`)

### Governing Equations

Pivoted LU ($P A = L U$), symmetric positive-definite Cholesky ($A = L L^T$), and rank-revealing Householder QR with column pivoting ($A P = Q R$) solve dense linear systems and least-squares objectives in caller-owned `SolverWorkspace` buffers without forming $A^T A$:

$$
P A = L U \implies A x = b, \qquad A = L L^T \succ 0 \implies L(L^T x) = b, \qquad \min_{x \in \mathbb{R}^n} \|A x - b\|_2 = \min_{z \in \mathbb{R}^n} \|R z - Q^T b\|_2,\ x = P z
$$

Contrast variance forms evaluate $c^T (A^T A)^{-1} c = \|R^{-T}(P^T c)\|_2^2$ directly via triangular forward substitution (`qr_solve_rt_permuted`).

### Citations

- BibTeX keys: [`boisvert1997`](../../papers/references.bib#boisvert1997), [`nalgebra`](../../papers/references.bib#nalgebra).
- Literature: Golub & Van Loan (2013), *Matrix Computations* (4th ed.), DOI: `10.56021/9781421407944`; Higham (2002), *Accuracy and Stability of Numerical Algorithms* (2nd ed.), DOI: `10.1137/1.9780898718027`.
- Manuscript: [`papers/advanced_linear_algebra.tex`](../../papers/advanced_linear_algebra.tex).

### Stability

Partial row pivoting bounds LU multipliers by $|L_{ij}| \le 1$; Cholesky requires zero pivoting and achieves backward error $O(n\,\varepsilon_{\text{mach}})$ on SPD inputs; Householder QR with column pivoting is unconditionally backward-stable with relative backward error $O(m n\,\varepsilon_{\text{mach}})$ and avoids squaring $\kappa_2(A)$.

### Domain of Validity

Requires finite IEEE-754 inputs (`check_all_finite`), matching row-major slice lengths (`m * n`), sufficient `SolverWorkspace` capacity, nonsingular pivots (`LinalgError::SingularMatrix`), and symmetry plus positive definiteness (`LinalgError::NotPositiveDefinite`) for Cholesky.

```rust
use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::pure_math::algebra::linear_algebra::solve_normal_equation;

let a = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 2.0]);
let b = DVector::from_vec(vec![1.0, 3.0, 5.0]);
let x = solve_normal_equation(&a, &b).unwrap();
assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 2.0).abs() < 1e-12);
```

## 2. SVD Pseudoinverse, Tikhonov Regularization & Sparse Krylov Solvers (`svd_ops`, `tikhonov`, `krylov`)

### Governing Equations

Golub-Kahan bidiagonal SVD (`svd_solve_least_squares`, `qr_or_svd_solve`) and Tikhonov spectral filtering (`tikhonov_solve`, `solve_damped_normal`) compute minimum-norm and ridge-regularized solutions, while sparse CSR Conjugate Gradient (`cg_solve`) and restarted $\text{GMRES}(m)$ (`gmres_solve`) minimize energy or Krylov residual norms over $\mathcal{K}_k(A, r_0)$:

$$
x_{\text{SVD}} = \sum_{\sigma_i > \tau} \frac{u_i^T b}{\sigma_i} v_i, \qquad x_\lambda = \sum_{i=1}^{\min(m,n)} \frac{\sigma_i}{\sigma_i^2 + \lambda^2} (u_i^T b) v_i, \qquad \min_{x \in x_0 + \mathcal{K}_m(A, r_0)} \|b - A x\|_2
$$

### Citations

- BibTeX keys: [`boisvert1997`](../../papers/references.bib#boisvert1997), [`nalgebra`](../../papers/references.bib#nalgebra).
- Literature: Hestenes & Stiefel (1952), DOI: `10.6028/jres.049.044`; Saad & Schultz (1986), DOI: `10.1137/0907058`; Tikhonov & Arsenin (1977), *Solutions of Ill-Posed Problems*.
- Manuscript: [`papers/advanced_linear_algebra.tex`](../../papers/advanced_linear_algebra.tex).

### Stability

SVD spectral truncation discards singular values $\sigma_i \le \max(\tau, \max(m,n)\varepsilon_{\text{mach}}\sigma_1)$ to bound noise amplification; Tikhonov filter factors $\sigma_i / (\sigma_i^2 + \lambda^2) \le 1/(2\lambda)$ bound sensitivity for $\lambda > 0$; `cg_solve` checks $p_k^T A p_k > 0$ at every step and `gmres_solve` uses Modified Gram-Schmidt with Givens rotations.

### Domain of Validity

Requires finite entries, regularization $\lambda \ge 0$ (`tikhonov_solve`, `solve_damped_normal`), SPD sparse operator $A \succ 0$ and tolerance $\text{tol} > 0$ for `cg_solve`, and restart dimension $m \ge 1$ with $\text{tol} > 0$ for `gmres_solve` (`LinalgError::NonConvergence` if budget is exhausted).

```rust
use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::pure_math::algebra::linear_algebra::solve_tikhonov_system;

let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
let b = DVector::from_vec(vec![2.0, -4.0]);
let x = solve_tikhonov_system(&a, &b, 1.0).unwrap();
assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] + 2.0).abs() < 1e-12);
```
