#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:engine_linalg]
// [cite:advanced_linear_algebra]
//! Error types, caller-owned [`LinalgWorkspace`] and [`SolverWorkspace`] buffers, row-major views, and stack array kernels.

use crate::scalar::Scalar;
use core::marker::PhantomData;
#[cfg(feature = "alloc")]
pub use math_memory::OwnedByteArena;
pub use math_memory::{ArenaCheckpoint, ArenaError, ByteArena, ByteCursor, SolverWorkspace};
pub use math_scalars::{LinalgError, LinearAlgebraError, ScalarError, mat_idx};

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

/// Carves `count` zeroed scalar `T` elements from `cur`, mapping byte exhaustion to scalar counts.
#[inline]
pub(crate) fn carve_scalars<'a, T: Scalar>(
    cur: &mut ByteCursor<'a>,
    count: usize,
) -> Result<&'a mut [T], LinalgError> {
    let elem_size = core::mem::size_of::<T>();
    cur.alloc_slice::<T>(count).map_err(|err| match err {
        ArenaError::InsufficientBytes { available, .. } => LinalgError::WorkspaceTooSmall {
            required: count,
            available: available
                .checked_div(elem_size)
                .unwrap_or(0)
                .min(count.saturating_sub(1)),
        },
        _ => LinalgError::WorkspaceTooSmall {
            required: count,
            available: 0,
        },
    })
}

/// Caller-owned mutable scalar slice wrapper implementing [`SolverWorkspace`] without heap allocation.
#[derive(Debug)]
pub struct LinalgWorkspace<'a, T: Scalar> {
    bytes: &'a mut [u8],
    _marker: PhantomData<&'a mut [T]>,
}

impl<'a, T: Scalar> LinalgWorkspace<'a, T> {
    /// Wraps a caller-owned mutable scalar slice as a reusable linear algebra workspace.
    #[inline]
    #[must_use]
    pub fn new(buf: &'a mut [T]) -> Self {
        Self {
            bytes: bytemuck::cast_slice_mut(buf),
            _marker: PhantomData,
        }
    }

    /// Creates an empty zero-capacity workspace for routines that operate strictly in-place.
    #[inline]
    #[must_use]
    pub fn empty() -> Self {
        Self {
            bytes: &mut [],
            _marker: PhantomData,
        }
    }

    /// Returns the number of aligned scalar `T` elements remaining in this workspace.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        bytemuck::pod_align_to::<u8, T>(self.bytes).1.len()
    }

    /// Returns `true` if the workspace holds zero aligned scalar `T` elements.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Splits off `count` zero-initialized scalar elements and returns the remaining sub-workspace.
    #[inline]
    pub fn try_split(
        &mut self,
        count: usize,
    ) -> Result<(&mut [T], LinalgWorkspace<'_, T>), LinalgError> {
        let available = self.len();
        if available < count {
            return Err(LinalgError::WorkspaceTooSmall {
                required: count,
                available,
            });
        }
        let mut cur = ByteCursor::new(self.bytes);
        let head = carve_scalars::<T>(&mut cur, count)?;
        let rest_bytes = cur.into_bytes();
        Ok((
            head,
            LinalgWorkspace {
                bytes: rest_bytes,
                _marker: PhantomData,
            },
        ))
    }
}

impl<T: Scalar> SolverWorkspace for LinalgWorkspace<'_, T> {
    #[inline]
    fn byte_capacity(&self) -> usize {
        self.bytes.len()
    }

    #[inline]
    fn remaining_bytes(&self) -> usize {
        self.bytes.len()
    }

    #[inline]
    fn reset(&mut self) {}

    #[inline]
    fn reborrow(&mut self) -> ByteArena<'_> {
        ByteArena::new(self.bytes)
    }

    #[inline]
    fn cursor(&mut self) -> ByteCursor<'_> {
        ByteCursor::new(self.bytes)
    }

    fn alloc_slice<U: bytemuck::Pod + bytemuck::Zeroable>(
        &mut self,
        len: usize,
    ) -> Result<&mut [U], ArenaError> {
        let full = core::mem::take(&mut self.bytes);
        let mut cur = ByteCursor::new(full);
        match cur.alloc_slice::<U>(len) {
            Ok(slice) => {
                self.bytes = cur.into_bytes();
                Ok(slice)
            }
            Err(err) => {
                self.bytes = cur.into_bytes();
                Err(err)
            }
        }
    }
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
