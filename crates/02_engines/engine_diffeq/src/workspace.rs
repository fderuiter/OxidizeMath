//! Zero-allocation caller-owned slice and stack `SolverWorkspace` buffers plus in-place dense LU routines.
// [cite:engine_diffeq]

use crate::error::DiffeqError;

/// Caller-owned mutable scratch buffer enabling zero-allocation ODE stepping, stiff Newton/LU solves,
/// and adaptive quadrature without `alloc` or `std`.
#[derive(Debug)]
pub struct SolverWorkspace<'a> {
    buf: &'a mut [f64],
}

impl<'a> SolverWorkspace<'a> {
    /// Wraps a caller-owned mutable `f64` slice as a [`SolverWorkspace`].
    #[must_use]
    #[verified_engine::verified]
    pub fn from_slice(buf: &'a mut [f64]) -> Self {
        Self { buf }
    }

    /// Returns the total `f64` capacity of this workspace.
    #[must_use]
    #[verified_engine::verified]
    pub const fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Borrows the underlying scratch slice mutably, verifying at least `required` elements exist.
    #[verified_engine::verified]
    pub fn try_slice_mut(&mut self, required: usize) -> Result<&mut [f64], DiffeqError> {
        let available = self.buf.len();
        if available < required {
            return Err(DiffeqError::InsufficientWorkspace {
                required,
                available,
            });
        }
        Ok(&mut self.buf[..required])
    }

    /// Computes the minimum `f64` scratch elements required for a stiff `Rosenbrock23` or `Bdf2` step
    /// of dimension `dim` (`dim * dim + 8 * dim`).
    #[must_use]
    #[verified_engine::verified]
    pub const fn stiff_scratch_len(dim: usize) -> usize {
        dim * dim + 8 * dim
    }

    /// Computes the minimum `f64` scratch elements required for explicit `DormandPrince45` stepping
    /// of a slice state of dimension `dim` (`9 * dim`).
    #[must_use]
    #[verified_engine::verified]
    pub const fn rk45_scratch_len(dim: usize) -> usize {
        9 * dim
    }
}

/// Fixed-capacity stack buffer that lends out zero-allocation [`SolverWorkspace`] views.
#[derive(Debug, Clone)]
pub struct StackWorkspace<const CAP: usize> {
    storage: [f64; CAP],
}

impl<const CAP: usize> Default for StackWorkspace<CAP> {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new()
    }
}

impl<const CAP: usize> StackWorkspace<CAP> {
    /// Constructs a zero-initialized stack workspace of `CAP` `f64` elements.
    #[must_use]
    #[verified_engine::verified]
    pub const fn new() -> Self {
        Self {
            storage: [0.0; CAP],
        }
    }

    /// Borrows this stack buffer as a [`SolverWorkspace`].
    #[must_use]
    #[verified_engine::verified]
    pub fn as_workspace(&mut self) -> SolverWorkspace<'_> {
        SolverWorkspace::from_slice(&mut self.storage)
    }
}

/// Maps 2D `(row, col)` coordinates in a row-major `width`-strided matrix to a linear index.
#[inline]
pub(crate) const fn row_major_index(row: usize, width: usize, col: usize) -> usize {
    row * width + col
}

/// Factorizes an `n x n` row-major dense matrix `mat` in-place via Doolittle LU with partial pivoting,
/// recording row swaps in `pivots[..n]`.
#[verified_engine::verified]
pub fn lu_factor_in_place(
    mat: &mut [f64],
    pivots: &mut [usize],
    n: usize,
) -> Result<(), DiffeqError> {
    for (idx, p) in pivots.iter_mut().take(n).enumerate() {
        *p = idx;
    }
    for k in 0..n {
        let mut pivot_row = k;
        let mut max_abs = mat[row_major_index(k, n, k)].abs();
        for i in (k + 1)..n {
            let cand = mat[row_major_index(i, n, k)].abs();
            if cand > max_abs {
                max_abs = cand;
                pivot_row = i;
            }
        }
        if !max_abs.is_finite() || max_abs <= 1e-30 {
            return Err(DiffeqError::SingularJacobian { pivot_index: k });
        }
        if pivot_row != k {
            pivots.swap(k, pivot_row);
            for col in 0..n {
                mat.swap(
                    row_major_index(k, n, col),
                    row_major_index(pivot_row, n, col),
                );
            }
        }
        let inv_diag = 1.0 / mat[row_major_index(k, n, k)];
        for i in (k + 1)..n {
            let ik = row_major_index(i, n, k);
            let factor = mat[ik] * inv_diag;
            mat[ik] = factor;
            for j in (k + 1)..n {
                let ij = row_major_index(i, n, j);
                let kj = row_major_index(k, n, j);
                mat[ij] -= factor * mat[kj];
            }
        }
    }
    Ok(())
}

/// Solves $A x = b$ using an in-place LU factorization produced by [`lu_factor_in_place`],
/// reading right-hand side `rhs[..n]` and writing the solution into `sol[..n]`.
#[verified_engine::verified]
pub fn lu_solve_in_place(lu: &[f64], pivots: &[usize], rhs: &[f64], sol: &mut [f64], n: usize) {
    for i in 0..n {
        let mut sum = rhs[pivots[i]];
        for (j, &s_j) in sol.iter().enumerate().take(i) {
            sum -= lu[row_major_index(i, n, j)] * s_j;
        }
        sol[i] = sum;
    }
    for idx in 0..n {
        let i = n - 1 - idx;
        let mut sum = sol[i];
        for (j, &s_j) in sol.iter().enumerate().take(n).skip(i + 1) {
            sum -= lu[row_major_index(i, n, j)] * s_j;
        }
        sol[i] = sum / lu[row_major_index(i, n, i)];
    }
}
