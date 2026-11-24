#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:engine_linalg]
// [cite:advanced_linear_algebra]
//! Zero-copy Compressed Sparse Row ([`CsrMatrixView`]) matrix views and sparse matrix-vector products.

use crate::scalar::Scalar;
use crate::types::{LinalgError, check_all_finite};

/// Immutable `#![no_std]` Compressed Sparse Row (CSR) matrix view over caller slices.
#[derive(Copy, Clone, Debug)]
pub struct CsrMatrixView<'a, T> {
    nrows: usize,
    ncols: usize,
    row_ptr: &'a [usize],
    col_ind: &'a [usize],
    values: &'a [T],
}

impl<'a, T: Scalar> CsrMatrixView<'a, T> {
    /// Constructs and validates a CSR sparse matrix view of shape `nrows x ncols`.
    pub fn new(
        nrows: usize,
        ncols: usize,
        row_ptr: &'a [usize],
        col_ind: &'a [usize],
        values: &'a [T],
    ) -> Result<Self, LinalgError> {
        if row_ptr.len() != nrows + 1 || col_ind.len() != values.len() {
            return Err(LinalgError::DimensionMismatch);
        }
        if row_ptr[0] != 0 || row_ptr[nrows] != values.len() {
            return Err(LinalgError::InvalidInput);
        }
        for r in 0..nrows {
            if row_ptr[r] > row_ptr[r + 1] {
                return Err(LinalgError::InvalidInput);
            }
        }
        for &col in col_ind {
            if col >= ncols {
                return Err(LinalgError::InvalidInput);
            }
        }
        check_all_finite(values)?;
        Ok(Self {
            nrows,
            ncols,
            row_ptr,
            col_ind,
            values,
        })
    }

    /// Number of rows.
    #[inline]
    pub const fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    #[inline]
    pub const fn ncols(&self) -> usize {
        self.ncols
    }

    /// Number of stored structural non-zeros.
    #[inline]
    pub const fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Row pointer array of length `nrows + 1`.
    #[inline]
    pub const fn row_ptr(&self) -> &'a [usize] {
        self.row_ptr
    }

    /// Column indices of length `nnz`.
    #[inline]
    pub const fn col_ind(&self) -> &'a [usize] {
        self.col_ind
    }

    /// Non-zero scalar values of length `nnz`.
    #[inline]
    pub const fn values(&self) -> &'a [T] {
        self.values
    }

    /// Computes the sparse matrix-vector product $y = A x$ with zero allocations.
    pub fn spmv(&self, x: &[T], y: &mut [T]) -> Result<(), LinalgError> {
        if x.len() != self.ncols || y.len() != self.nrows {
            return Err(LinalgError::DimensionMismatch);
        }
        check_all_finite(x)?;
        for (r, slot) in y.iter_mut().enumerate() {
            let start = self.row_ptr[r];
            let end = self.row_ptr[r + 1];
            let mut sum = T::ZERO;
            for idx in start..end {
                sum += self.values[idx] * x[self.col_ind[idx]];
            }
            *slot = sum;
        }
        Ok(())
    }
}
