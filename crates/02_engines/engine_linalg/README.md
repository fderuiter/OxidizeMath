# `engine_linalg`

## 1. Overview & Architecture

`engine_linalg` is the foundational `#![no_std]` numerical linear algebra engine for the OxidizeMath workspace (`crates/02_engines/engine_linalg`). It executes dense factorizations, singular value decompositions, sparse CSR Krylov iterations, matrix norm and condition-number estimators, and Tikhonov regularization over generic IEEE-754 [`Scalar`] types (`f32`, `f64`) using caller-owned [`SolverWorkspace`] scratch slices with zero heap allocations.

### Key Capabilities

- **Caller-Owned Scratch Memory**: [`SolverWorkspace`] partitions preallocated stack or arena buffers deterministically without `std` or a global allocator (`alloc` is strictly optional for [`OwnedWorkspace`] convenience wrappers).
- **Direct Factorizations**:
  - **Householder QR with Column Pivoting** (`qr_factor_pivoted`, `qr_solve_least_squares`, `qr_solve_rt_permuted`, `StreamingQr`) for rank-revealing orthogonal least squares without squaring condition numbers in $A^T A$.
  - **LU with Partial Pivoting** (`lu_factor_in_place`, `lu_solve`, `lu_inverse`, `lu_determinant`) for square nonsingular linear systems.
  - **Strict & Semidefinite Cholesky** (`cholesky_factor`, `cholesky_solve`, `cholesky_semidefinite`) for symmetric positive-definite and correlation/covariance PSD factorizations.
- **Singular Value Decompositions** (`papers/advanced_linear_algebra.tex`):
  - **One-Sided Hestenes-Jacobi SVD** (`svd_jacobi`) and symmetric Jacobi eigensolver (`symmetric_eigen_jacobi`).
  - **Golub-Kahan Bidiagonal QR SVD** (`svd_bidiagonal`, `svd_pseudoinverse`, `svd_low_rank_approx`, `svd_solve_least_squares`, `qr_or_svd_solve`) with explicit [`LinalgError`] diagnostics on `NaN` or non-convergence.
- **Sparse Krylov Solvers**:
  - Zero-copy [`CsrMatrixView`] sparse matrix views with preallocated **Conjugate Gradient** (`cg_solve`) and **restarted GMRES** (`gmres_solve`).
- **Conditioning & Regularization**:
  - Matrix norms (`norm_l1`, `norm_linf`, `norm_frobenius`, `norm_spectral`), Hager-Higham 1-norm and spectral condition estimators (`cond_l1`, `cond_spectral`), and SVD/LU Tikhonov ridge solvers (`tikhonov_solve`, `solve_damped_normal`).

### Example

```rust
use engine_linalg::{SolverWorkspace, qr_least_squares_workspace_len, qr_solve_least_squares};

// Overdetermined 3x2 linear least-squares problem A x ≈ b
let a = [1.0_f64, 0.0, 1.0, 1.0, 1.0, 2.0];
let b = [1.0_f64, 3.0, 5.0];
let mut x = [0.0_f64; 2];
let mut perm = [0_usize; 2];
let mut scratch = [0.0_f64; 32];
assert!(scratch.len() >= qr_least_squares_workspace_len(3, 2));
let mut ws = SolverWorkspace::new(&mut scratch);

let rank = qr_solve_least_squares(&a, 3, 2, &b, &mut x, &mut perm, &mut ws).unwrap();
assert_eq!(rank, 2);
assert!((x[0] - 1.0).abs() < 1e-12);
assert!((x[1] - 2.0).abs() < 1e-12);
```

## 2. Developer Instructions

Verify `#![no_std]` WebAssembly compilation and run the unit, conditioning, and zero-allocation test suite from the workspace root:

```bash
cargo check -p engine_linalg --target wasm32-unknown-unknown --no-default-features
cargo test -p engine_linalg
```
