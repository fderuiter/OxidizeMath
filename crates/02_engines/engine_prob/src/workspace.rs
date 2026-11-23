//! Borrowed zero-allocation [`SolverWorkspace`] slice container and optional `alloc` backing.

use crate::error::ProbError;

/// Borrowed scratch slice container supplying `&mut [f64]` and `&mut [usize]` buffers to solvers.
///
/// All `engine_prob` cubature, variance-reduction, and bootstrap routines carve temporary arrays
/// out of `SolverWorkspace` rather than allocating on the heap.
#[derive(Debug)]
pub struct SolverWorkspace<'a> {
    floats: &'a mut [f64],
    indices: &'a mut [usize],
}

impl<'a> SolverWorkspace<'a> {
    /// Wraps caller-owned mutable `floats` and `indices` slices.
    #[inline]
    #[must_use]
    pub fn new(floats: &'a mut [f64], indices: &'a mut [usize]) -> Self {
        Self { floats, indices }
    }

    /// Wraps a caller-owned `floats` slice with an empty `indices` slice.
    #[inline]
    #[must_use]
    pub fn from_floats(floats: &'a mut [f64]) -> Self {
        Self {
            floats,
            indices: &mut [],
        }
    }

    /// Number of `f64` slots available in this workspace view.
    #[inline]
    #[must_use]
    pub fn float_len(&self) -> usize {
        self.floats.len()
    }

    /// Number of `usize` slots available in this workspace view.
    #[inline]
    #[must_use]
    pub fn index_len(&self) -> usize {
        self.indices.len()
    }

    /// Immutable view of the underlying `f64` scratch buffer.
    #[inline]
    #[must_use]
    pub fn floats(&self) -> &[f64] {
        self.floats
    }

    /// Mutable view of the underlying `f64` scratch buffer.
    #[inline]
    pub fn floats_mut(&mut self) -> &mut [f64] {
        self.floats
    }

    /// Immutable view of the underlying `usize` scratch buffer.
    #[inline]
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        self.indices
    }

    /// Mutable view of the underlying `usize` scratch buffer.
    #[inline]
    pub fn indices_mut(&mut self) -> &mut [usize] {
        self.indices
    }

    /// Reborrows both scratch slices with a shorter lifetime for sequential solver invocations.
    #[inline]
    pub fn reborrow(&mut self) -> SolverWorkspace<'_> {
        SolverWorkspace {
            floats: self.floats,
            indices: self.indices,
        }
    }

    /// Verifies that at least `req_floats` and `req_indices` slots are available.
    #[inline]
    pub fn ensure_capacity(&self, req_floats: usize, req_indices: usize) -> Result<(), ProbError> {
        if self.floats.len() < req_floats || self.indices.len() < req_indices {
            return Err(ProbError::InsufficientWorkspace {
                required_floats: req_floats,
                available_floats: self.floats.len(),
                required_indices: req_indices,
                available_indices: self.indices.len(),
            });
        }
        Ok(())
    }

    /// Splits off `n_floats` from the front of `floats` and returns the remainder workspace.
    #[inline]
    pub fn split_floats(
        &mut self,
        n_floats: usize,
    ) -> Result<(&mut [f64], SolverWorkspace<'_>), ProbError> {
        self.ensure_capacity(n_floats, 0)?;
        let (head, tail) = self.floats.split_at_mut(n_floats);
        Ok((
            head,
            SolverWorkspace {
                floats: tail,
                indices: self.indices,
            },
        ))
    }

    /// Splits off `n_indices` from the front of `indices` and returns the remainder workspace.
    #[inline]
    pub fn split_indices(
        &mut self,
        n_indices: usize,
    ) -> Result<(&mut [usize], SolverWorkspace<'_>), ProbError> {
        self.ensure_capacity(0, n_indices)?;
        let (head, tail) = self.indices.split_at_mut(n_indices);
        Ok((
            head,
            SolverWorkspace {
                floats: self.floats,
                indices: tail,
            },
        ))
    }

    /// Splits off both `n_floats` and `n_indices` simultaneously and returns the remainder workspace.
    #[inline]
    pub fn split_parts(
        &mut self,
        n_floats: usize,
        n_indices: usize,
    ) -> Result<(&mut [f64], &mut [usize], SolverWorkspace<'_>), ProbError> {
        self.ensure_capacity(n_floats, n_indices)?;
        let (f_head, f_tail) = self.floats.split_at_mut(n_floats);
        let (i_head, i_tail) = self.indices.split_at_mut(n_indices);
        Ok((
            f_head,
            i_head,
            SolverWorkspace {
                floats: f_tail,
                indices: i_tail,
            },
        ))
    }
}

/// Optional `alloc`-backed workspace owner that preallocates reusable buffers outside hot loops.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, Default)]
pub struct OwnedSolverWorkspace {
    floats: alloc::vec::Vec<f64>,
    indices: alloc::vec::Vec<usize>,
}

#[cfg(feature = "alloc")]
impl OwnedSolverWorkspace {
    /// Preallocates zero-initialized `f64` and `usize` buffers of the requested lengths.
    #[must_use]
    pub fn with_capacity(n_floats: usize, n_indices: usize) -> Self {
        Self {
            floats: alloc::vec![0.0; n_floats],
            indices: alloc::vec![0; n_indices],
        }
    }

    /// Grows backing buffers if needed so at least `n_floats` and `n_indices` elements exist.
    pub fn reserve_to(&mut self, n_floats: usize, n_indices: usize) {
        if self.floats.len() < n_floats {
            self.floats.resize(n_floats, 0.0);
        }
        if self.indices.len() < n_indices {
            self.indices.resize(n_indices, 0);
        }
    }

    /// Borrows the owned buffers as a zero-allocation [`SolverWorkspace`].
    #[inline]
    #[must_use]
    pub fn as_workspace(&mut self) -> SolverWorkspace<'_> {
        SolverWorkspace::new(&mut self.floats, &mut self.indices)
    }
}
