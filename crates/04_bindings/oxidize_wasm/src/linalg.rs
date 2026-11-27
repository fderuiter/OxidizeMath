//! Zero-allocation `#[wasm_bindgen]` linear algebra workspace wrapping `engine_linalg`.
// [cite:engine_linalg]
// [cite:advanced_linear_algebra]

use crate::error::WasmSdkError;
use engine_linalg::{
    CsrMatrixView, LinalgError, OwnedByteArena, cg_solve, cg_workspace_len, cholesky_solve,
    cholesky_workspace_len, lu_determinant, lu_factor_in_place, lu_inverse_workspace_len, lu_solve,
    qr_least_squares_workspace_len, qr_solve_least_squares, svd_jacobi, svd_jacobi_workspace_len,
    tikhonov_solve, tikhonov_workspace_len,
};
use wasm_bindgen::prelude::*;

/// Maximum matrix dimension `max_dim` per axis for `WasmLinalgWorkspace` (`2,048 x 2,048`).
pub const MAX_LINALG_DIM: usize = 2_048;

fn required_linalg_scratch(max_dim: usize) -> usize {
    let lu_cap = lu_inverse_workspace_len(max_dim);
    let chol_cap = cholesky_workspace_len(max_dim);
    let qr_cap = qr_least_squares_workspace_len(max_dim, max_dim);
    let svd_cap = svd_jacobi_workspace_len(max_dim, max_dim);
    let tikh_cap = tikhonov_workspace_len(max_dim, max_dim);
    let cg_cap = cg_workspace_len(max_dim);
    lu_cap
        .max(chol_cap)
        .max(qr_cap)
        .max(svd_cap)
        .max(tikh_cap)
        .max(cg_cap)
}

/// Reusable zero-allocation linear algebra workspace for LU, Cholesky, QR, SVD, Tikhonov, and CG solves.
///
/// Pre-allocates scratch buffers for matrices up to `max_dim x max_dim` once at construction.
/// Callers can pass `Float64Array` slices directly (`&[f64]` / `&mut [f64]`) or populate the
/// internal linear-memory matrix/vector buffers via [`Self::matrix_ptr`] / [`Self::rhs_ptr`] and
/// read [`Self::solution_ptr`] with zero copies.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WasmLinalgWorkspace {
    max_dim: usize,
    scratch: OwnedByteArena,
    pivots: Vec<usize>,
    mat_buf: Vec<f64>,
    rhs_buf: Vec<f64>,
    sol_buf: Vec<f64>,
    u_buf: Vec<f64>,
    vt_buf: Vec<f64>,
}

#[wasm_bindgen]
impl WasmLinalgWorkspace {
    /// Pre-allocates a reusable workspace supporting dense and sparse systems up to `max_dim`.
    #[wasm_bindgen(js_name = tryNew)]
    pub fn try_new(max_dim: usize) -> Result<Self, WasmSdkError> {
        if max_dim == 0 || max_dim > MAX_LINALG_DIM {
            return Err(WasmSdkError::invalid_argument(
                "E_LINALG_MAX_DIM",
                format!("max_dim={max_dim} must be within 1..={MAX_LINALG_DIM}"),
                MAX_LINALG_DIM,
                max_dim,
            ));
        }
        let scratch_len = required_linalg_scratch(max_dim);
        let byte_cap = scratch_len
            .saturating_add(16)
            .saturating_mul(std::mem::size_of::<f64>());
        let scratch = OwnedByteArena::try_with_capacity(byte_cap)?;
        let mat_cap = max_dim * max_dim;
        Ok(Self {
            max_dim,
            scratch,
            pivots: vec![0; max_dim],
            mat_buf: vec![0.0; mat_cap],
            rhs_buf: vec![0.0; max_dim],
            sol_buf: vec![0.0; max_dim],
            u_buf: vec![0.0; mat_cap],
            vt_buf: vec![0.0; mat_cap],
        })
    }

    /// Maximum matrix row/column dimension supported without reallocating.
    #[wasm_bindgen(js_name = maxDim)]
    #[must_use]
    pub fn max_dim(&self) -> usize {
        self.max_dim
    }

    /// Byte offset of the internal `max_dim * max_dim` matrix buffer in WASM linear memory.
    #[wasm_bindgen(js_name = matrixPtr)]
    #[must_use]
    pub fn matrix_ptr(&self) -> usize {
        self.mat_buf.as_ptr() as usize
    }

    /// Byte offset of the internal `max_dim` right-hand-side buffer in WASM linear memory.
    #[wasm_bindgen(js_name = rhsPtr)]
    #[must_use]
    pub fn rhs_ptr(&self) -> usize {
        self.rhs_buf.as_ptr() as usize
    }

    /// Byte offset of the internal `max_dim` solution / singular-value buffer in WASM linear memory.
    #[wasm_bindgen(js_name = solutionPtr)]
    #[must_use]
    pub fn solution_ptr(&self) -> usize {
        self.sol_buf.as_ptr() as usize
    }

    fn check_dim(&self, m: usize, n: usize) -> Result<(), WasmSdkError> {
        if m == 0 || n == 0 || m > self.max_dim || n > self.max_dim {
            return Err(WasmSdkError::invalid_argument(
                "E_LINALG_DIM_RANGE",
                format!(
                    "Requested ({m}, {n}) exceeds workspace max_dim={}",
                    self.max_dim
                ),
                self.max_dim,
                m.max(n),
            ));
        }
        Ok(())
    }

    /// Solves square `n x n` linear system `A x = b` via partial-pivoting LU with zero allocations.
    #[wasm_bindgen(js_name = solveLu)]
    pub fn solve_lu(
        &mut self,
        a: &[f64],
        b: &[f64],
        x_out: &mut [f64],
        n: usize,
    ) -> Result<(), WasmSdkError> {
        self.check_dim(n, n)?;
        let mut ws = self.scratch.as_arena();
        lu_solve(a, b, x_out, n, &mut self.pivots[..n], &mut ws)?;
        self.sol_buf[..n].copy_from_slice(&x_out[..n]);
        Ok(())
    }

    /// Solves `A x = b` in-place from [`Self::matrix_ptr`] and [`Self::rhs_ptr`] into [`Self::solution_ptr`].
    #[wasm_bindgen(js_name = solveLuInPlace)]
    pub fn solve_lu_in_place(&mut self, n: usize) -> Result<(), WasmSdkError> {
        self.check_dim(n, n)?;
        let mut ws = self.scratch.as_arena();
        lu_solve(
            &self.mat_buf[..n * n],
            &self.rhs_buf[..n],
            &mut self.sol_buf[..n],
            n,
            &mut self.pivots[..n],
            &mut ws,
        )?;
        Ok(())
    }

    /// Computes `det(A)` of a square `n x n` matrix via partial-pivoting LU with zero allocations.
    #[wasm_bindgen(js_name = determinant)]
    pub fn determinant(&mut self, a: &[f64], n: usize) -> Result<f64, WasmSdkError> {
        self.check_dim(n, n)?;
        if a.len() != n * n {
            return Err(LinalgError::DimensionMismatch.into());
        }
        let mut ws = self.scratch.as_arena();
        let lu_buf = ws.alloc_slice::<f64>(n * n).map_err(LinalgError::from)?;
        lu_buf.copy_from_slice(a);
        match lu_factor_in_place(lu_buf, n, &mut self.pivots[..n], 0.0) {
            Ok(parity) => Ok(lu_determinant(lu_buf, n, parity)?),
            Err(LinalgError::SingularMatrix) => Ok(0.0),
            Err(other) => Err(other.into()),
        }
    }

    /// Solves symmetric positive-definite `n x n` system `A x = b` via $L L^T$ Cholesky.
    #[wasm_bindgen(js_name = solveCholesky)]
    pub fn solve_cholesky(
        &mut self,
        a: &[f64],
        b: &[f64],
        x_out: &mut [f64],
        n: usize,
    ) -> Result<(), WasmSdkError> {
        self.check_dim(n, n)?;
        let mut ws = self.scratch.as_arena();
        cholesky_solve(a, b, x_out, n, &mut ws)?;
        self.sol_buf[..n].copy_from_slice(&x_out[..n]);
        Ok(())
    }

    /// Solves over- or under-determined least squares $\min_x \|A x - b\|_2$ via column-pivoted Householder QR.
    #[wasm_bindgen(js_name = solveQrLeastSquares)]
    pub fn solve_qr_least_squares(
        &mut self,
        a: &[f64],
        m: usize,
        n: usize,
        b: &[f64],
        x_out: &mut [f64],
    ) -> Result<usize, WasmSdkError> {
        self.check_dim(m, n)?;
        let mut ws = self.scratch.as_arena();
        let rank = qr_solve_least_squares(a, m, n, b, x_out, &mut self.pivots[..n], &mut ws)?;
        self.sol_buf[..n].copy_from_slice(&x_out[..n]);
        Ok(rank)
    }

    /// Solves Tikhonov ridge regression $\min_x \|A x - b\|_2^2 + \lambda^2 \|x\|_2^2$ via SVD filter factors.
    #[wasm_bindgen(js_name = solveTikhonov)]
    pub fn solve_tikhonov(
        &mut self,
        a: &[f64],
        m: usize,
        n: usize,
        b: &[f64],
        lambda: f64,
        x_out: &mut [f64],
    ) -> Result<(), WasmSdkError> {
        self.check_dim(m, n)?;
        let mut ws = self.scratch.as_arena();
        tikhonov_solve(a, m, n, b, lambda, x_out, &mut ws)?;
        self.sol_buf[..n].copy_from_slice(&x_out[..n]);
        Ok(())
    }

    /// Computes thin Hestenes-Jacobi singular values `s_out` (`min(m, n)`) with zero heap allocations.
    #[wasm_bindgen(js_name = singularValuesJacobi)]
    pub fn singular_values_jacobi(
        &mut self,
        a: &[f64],
        m: usize,
        n: usize,
        s_out: &mut [f64],
    ) -> Result<(), WasmSdkError> {
        self.check_dim(m, n)?;
        let k = m.min(n);
        let mut ws = self.scratch.as_arena();
        svd_jacobi(
            a,
            (m, n),
            &mut self.u_buf[..m * k],
            s_out,
            &mut self.vt_buf[..k * n],
            64,
            &mut ws,
        )?;
        self.sol_buf[..k].copy_from_slice(&s_out[..k]);
        Ok(())
    }

    /// Solves sparse SPD CSR system `A x = b` via Conjugate Gradient, returning the residual norm.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = solveCgCsr)]
    pub fn solve_cg_csr(
        &mut self,
        n: usize,
        row_ptrs: &[usize],
        col_indices: &[usize],
        values: &[f64],
        b: &[f64],
        x_in_out: &mut [f64],
        tol: f64,
        max_iter: usize,
    ) -> Result<f64, WasmSdkError> {
        self.check_dim(n, n)?;
        let csr = CsrMatrixView::new(n, n, row_ptrs, col_indices, values)?;
        let mut ws = self.scratch.as_arena();
        let report = cg_solve(&csr, b, x_in_out, tol, max_iter, &mut ws)?;
        self.sol_buf[..n].copy_from_slice(&x_in_out[..n]);
        Ok(report.residual_norm)
    }
}

impl WasmLinalgWorkspace {
    /// Mutable slice view into the internal `n * n` matrix buffer for zero-copy Rust/WASM setup.
    #[must_use]
    pub fn matrix_slice_mut(&mut self, len: usize) -> &mut [f64] {
        let clamped = len.min(self.mat_buf.len());
        &mut self.mat_buf[..clamped]
    }

    /// Mutable slice view into the internal right-hand-side vector buffer.
    #[must_use]
    pub fn rhs_slice_mut(&mut self, len: usize) -> &mut [f64] {
        let clamped = len.min(self.rhs_buf.len());
        &mut self.rhs_buf[..clamped]
    }

    /// Immutable slice view of the latest solution vector.
    #[must_use]
    pub fn solution_slice(&self, len: usize) -> &[f64] {
        let clamped = len.min(self.sol_buf.len());
        &self.sol_buf[..clamped]
    }
}
