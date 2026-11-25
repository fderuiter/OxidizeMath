#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:engine_linalg]
// [cite:advanced_linear_algebra]
//! Error types, caller-owned [`SolverWorkspace`] buffers, row-major views, and stack array kernels.

use crate::scalar::Scalar;
use core::fmt;

/// Computes the flat row-major index for `(row, col)` in a matrix with `ncols` columns.
/// `[cite:engine_linalg]`
#[inline]
pub const fn mat_idx(row: usize, col: usize, ncols: usize) -> usize {
    row * ncols + col
}

/// Validates that every element in `slice` is finite (`!is_nan()` and `!is_infinite()`).
/// `[cite:engine_linalg]`
#[inline]
pub fn check_all_finite<T: Scalar>(slice: &[T]) -> Result<(), LinalgError> {
    for &val in slice {
        if !val.is_finite() {
            return Err(LinalgError::InvalidInput);
        }
    }
    Ok(())
}

/// Typed `#![no_std]` error returned by [`crate`] factorizations and iterative solvers.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LinalgError {
    /// Matrix or vector dimensions do not conform to the requested operation.
    DimensionMismatch,
    /// Caller-supplied [`SolverWorkspace`] slice does not have enough capacity.
    WorkspaceTooSmall {
        /// Minimum scalar elements required by the routine.
        required: usize,
        /// Available scalar elements in the workspace.
        available: usize,
    },
    /// Input contains `NaN`, `Infinity`, or an out-of-domain parameter (such as negative ridge $\lambda$).
    InvalidInput,
    /// Matrix is numerically singular or rank-deficient beyond the solver's requirements.
    SingularMatrix,
    /// Matrix is asymmetric or fails positive-definite / positive-semidefinite checks.
    NotPositiveDefinite,
    /// Iterative decomposition or Krylov solver exceeded its iteration budget without converging.
    NonConvergence,
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch => f.write_str("matrix or vector dimensions do not match"),
            Self::WorkspaceTooSmall {
                required,
                available,
            } => write!(
                f,
                "solver workspace too small: required {required} scalars, available {available}"
            ),
            Self::InvalidInput => {
                f.write_str("linear algebra input contains non-finite values or invalid parameters")
            }
            Self::SingularMatrix => f.write_str("matrix is numerically singular or rank-deficient"),
            Self::NotPositiveDefinite => {
                f.write_str("matrix is not symmetric positive-definite or positive-semidefinite")
            }
            Self::NonConvergence => {
                f.write_str("iterative linear algebra algorithm failed to converge within budget")
            }
        }
    }
}

impl core::error::Error for LinalgError {}

impl math_diagnostics::Diagnostic for LinalgError {
    fn severity(&self) -> math_diagnostics::Severity {
        math_diagnostics::Severity::Error
    }

    fn visit_fields(&self, visitor: &mut dyn math_diagnostics::DiagnosticVisitor) {
        visitor.record_str("error_type", "LinalgError");
        visitor.record_display("description", self);
        match self {
            Self::DimensionMismatch => {
                visitor.record_str("code", "LINALG-001");
            }
            Self::WorkspaceTooSmall {
                required,
                available,
            } => {
                visitor.record_str("code", "LINALG-002");
                visitor.record_usize("required", *required);
                visitor.record_usize("available", *available);
            }
            Self::InvalidInput => {
                visitor.record_str("code", "LINALG-003");
            }
            Self::SingularMatrix => {
                visitor.record_str("code", "LINALG-004");
            }
            Self::NotPositiveDefinite => {
                visitor.record_str("code", "LINALG-005");
            }
            Self::NonConvergence => {
                visitor.record_str("code", "LINALG-006");
            }
        }
    }
}

impl From<math_memory::ArenaError> for LinalgError {
    #[inline]
    fn from(err: math_memory::ArenaError) -> Self {
        match err {
            math_memory::ArenaError::InsufficientCapacity {
                requested,
                available,
            } => Self::WorkspaceTooSmall {
                required: requested,
                available,
            },
            math_memory::ArenaError::CapacityOverflow { count, .. } => Self::WorkspaceTooSmall {
                required: count,
                available: 0,
            },
            math_memory::ArenaError::InvalidAlignment { .. }
            | math_memory::ArenaError::InvalidCheckpoint { .. } => Self::WorkspaceTooSmall {
                required: 1,
                available: 0,
            },
        }
    }
}

pub use math_memory::{ArenaScope, ByteArena, SolverWorkspace};

/// Carves `count` zero-initialized scalars of type `T` from `scope`, mapping capacity errors to [`LinalgError::WorkspaceTooSmall`].
#[inline]
pub(crate) fn carve_scalars<'a, T: Scalar>(
    scope: &mut ArenaScope<'a>,
    count: usize,
) -> Result<&'a mut [T], LinalgError> {
    let elem = core::mem::size_of::<T>().max(1);
    let avail_before = scope.remaining() / elem;
    scope.alloc_slice::<T>(count).map_err(|err| match err {
        math_memory::ArenaError::InsufficientCapacity {
            requested,
            available,
        } => LinalgError::WorkspaceTooSmall {
            required: requested.div_ceil(elem),
            available: available / elem,
        },
        math_memory::ArenaError::CapacityOverflow { .. }
        | math_memory::ArenaError::InvalidAlignment { .. }
        | math_memory::ArenaError::InvalidCheckpoint { .. } => LinalgError::WorkspaceTooSmall {
            required: count,
            available: avail_before,
        },
    })
}

/// Immutable row-major dense matrix view over a caller slice.
#[derive(Copy, Clone, Debug)]
pub struct DenseView<'a, T> {
    data: &'a [T],
    nrows: usize,
    ncols: usize,
}

impl<'a, T: Scalar> DenseView<'a, T> {
    /// Constructs a validated `nrows x ncols` row-major view.
    #[inline]
    pub fn new(data: &'a [T], nrows: usize, ncols: usize) -> Result<Self, LinalgError> {
        if nrows.checked_mul(ncols) != Some(data.len()) {
            return Err(LinalgError::DimensionMismatch);
        }
        Ok(Self { data, nrows, ncols })
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

    /// Underlying row-major slice.
    #[inline]
    pub const fn as_slice(&self) -> &'a [T] {
        self.data
    }

    /// Reads entry `(r, c)`.
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> T {
        self.data[mat_idx(r, c, self.ncols)]
    }
}

/// Computes row-major $C = A B$ for $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$.
/// `[cite:engine_linalg]`
pub fn mat_mul<T: Scalar>(
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
    c: &mut [T],
) -> Result<(), LinalgError> {
    if m.checked_mul(k) != Some(a.len())
        || k.checked_mul(n) != Some(b.len())
        || m.checked_mul(n) != Some(c.len())
    {
        return Err(LinalgError::DimensionMismatch);
    }
    c.fill(T::ZERO);
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[mat_idx(i, p, k)];
            for j in 0..n {
                c[mat_idx(i, j, n)] += a_ip * b[mat_idx(p, j, n)];
            }
        }
    }
    Ok(())
}

/// Computes row-major $C = A B^T$ for $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{n \times k}$.
/// `[cite:engine_linalg]`
pub fn mat_mul_transpose_right<T: Scalar>(
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
    c: &mut [T],
) -> Result<(), LinalgError> {
    if m.checked_mul(k) != Some(a.len())
        || n.checked_mul(k) != Some(b.len())
        || m.checked_mul(n) != Some(c.len())
    {
        return Err(LinalgError::DimensionMismatch);
    }
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::ZERO;
            for p in 0..k {
                sum += a[mat_idx(i, p, k)] * b[mat_idx(j, p, k)];
            }
            c[mat_idx(i, j, n)] = sum;
        }
    }
    Ok(())
}

/// Computes $y = A x$ for row-major $A \in \mathbb{R}^{m \times n}$.
/// `[cite:engine_linalg]`
pub fn mat_vec_mul<T: Scalar>(
    a: &[T],
    m: usize,
    n: usize,
    x: &[T],
    y: &mut [T],
) -> Result<(), LinalgError> {
    if m.checked_mul(n) != Some(a.len()) || x.len() != n || y.len() != m {
        return Err(LinalgError::DimensionMismatch);
    }
    for (i, slot) in y.iter_mut().enumerate() {
        let mut sum = T::ZERO;
        for j in 0..n {
            sum += a[mat_idx(i, j, n)] * x[j];
        }
        *slot = sum;
    }
    Ok(())
}

/// Returns the $N \times N$ stack identity matrix $I_N$.
/// `[cite:engine_linalg]`
#[inline]
pub fn identity_array<T: Scalar, const N: usize>() -> [[T; N]; N] {
    let mut out = [[T::ZERO; N]; N];
    let mut i = 0;
    while i < N {
        out[i][i] = T::ONE;
        i += 1;
    }
    out
}

/// Returns `base + scale * dir` on stack vectors of length `N`.
/// `[cite:engine_linalg]`
#[inline]
pub fn add_scaled_vec_array<T: Scalar, const N: usize>(
    base: &[T; N],
    dir: &[T; N],
    scale: T,
) -> [T; N] {
    let mut out = [T::ZERO; N];
    for i in 0..N {
        out[i] = base[i] + scale * dir[i];
    }
    out
}

/// Returns $I_N + \text{scale} \cdot J$ on $N \times N$ stack arrays.
/// `[cite:engine_linalg]`
#[inline]
pub fn identity_plus_scaled_array<T: Scalar, const N: usize>(
    jac: &[[T; N]; N],
    scale: T,
) -> [[T; N]; N] {
    let mut out = identity_array::<T, N>();
    for r in 0..N {
        for c in 0..N {
            out[r][c] += scale * jac[r][c];
        }
    }
    out
}

/// Multiplies $M \times K$ by $K \times N$ stack arrays: $C = A B$.
/// `[cite:engine_linalg]`
#[inline]
pub fn mat_mul_array<T: Scalar, const M: usize, const K: usize, const N: usize>(
    a: &[[T; K]; M],
    b: &[[T; N]; K],
) -> [[T; N]; M] {
    let mut out = [[T::ZERO; N]; M];
    for r in 0..M {
        for p in 0..K {
            let a_rp = a[r][p];
            for c in 0..N {
                out[r][c] += a_rp * b[p][c];
            }
        }
    }
    out
}

/// Multiplies $M \times K$ by the transpose of $N \times K$ stack arrays: $C = A B^T$.
/// `[cite:engine_linalg]`
#[inline]
pub fn mat_mul_transpose_right_array<T: Scalar, const M: usize, const K: usize, const N: usize>(
    a: &[[T; K]; M],
    b: &[[T; K]; N],
) -> [[T; N]; M] {
    let mut out = [[T::ZERO; N]; M];
    for r in 0..M {
        for c in 0..N {
            let mut sum = T::ZERO;
            for p in 0..K {
                sum += a[r][p] * b[c][p];
            }
            out[r][c] = sum;
        }
    }
    out
}
