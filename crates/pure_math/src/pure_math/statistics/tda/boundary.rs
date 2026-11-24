//! Preallocated bit-packed $\mathbb{Z}_2$ boundary matrix reduction workspace.
//!
//! Performs left-to-right column reduction over $\mathbb{Z}_2$ using packed 64-bit words
//! (`u64` XOR operations) without allocating heap memory inside the reduction loop,
//! satisfying `VerifiedAllocator` post-initialization memory locks.

use crate::error::TdaError;

/// Maximum number of boundary faces stored inline for up to 2-simplices (triangles have 3 edges).
pub const MAX_INLINE_BOUNDARY_FACES: usize = 3;

const NO_INDEX: usize = usize::MAX;
const EMPTY_FACE: u32 = u32::MAX;

/// Preallocated bit-packed workspace for exact $\mathbb{Z}_2$ column reduction of $\partial_1$ and $\partial_2$.
///
/// All backing buffers are allocated once in [`Z2BoundaryWorkspace::new`]. Calling
/// [`Z2BoundaryWorkspace::reduce`] mutates only preallocated slices and performs zero dynamic
/// heap allocations.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::tda::Z2BoundaryWorkspace;
///
/// // Four triangles of a hollow tetrahedron K_4 over 6 edges have Z_2 rank 3 (not 4).
/// let mut ws = Z2BoundaryWorkspace::new(6, 4).unwrap();
/// ws.set_column_faces(0, &[0, 1, 3]).unwrap();
/// ws.set_column_faces(1, &[0, 2, 4]).unwrap();
/// ws.set_column_faces(2, &[1, 2, 5]).unwrap();
/// ws.set_column_faces(3, &[3, 4, 5]).unwrap();
/// assert_eq!(ws.reduce(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct Z2BoundaryWorkspace {
    num_rows: usize,
    num_cols: usize,
    words_per_col: usize,
    max_pivots: usize,
    input_cols: Vec<[u32; MAX_INLINE_BOUNDARY_FACES]>,
    cleared_cols: Vec<bool>,
    pivot_words: Vec<u64>,
    row_to_slot: Vec<usize>,
    row_to_pivot_col: Vec<usize>,
    col_pivots: Vec<usize>,
    work_col: Vec<u64>,
    rank: usize,
}

impl Z2BoundaryWorkspace {
    /// Allocates a bounded $\mathbb{Z}_2$ reduction workspace for a `num_rows × num_cols` boundary matrix.
    ///
    /// # Errors
    ///
    /// Returns [`TdaError::MatrixError`] if dimensions overflow `u32` face indexing.
    #[verified_engine::verified(alloc_ok = "constructor pre-allocates state")]
    pub fn new(num_rows: usize, num_cols: usize) -> Result<Self, TdaError> {
        if num_rows >= EMPTY_FACE as usize {
            return Err(TdaError::MatrixError {
                reason: format!("boundary row count {num_rows} exceeds u32 capacity"),
            });
        }
        let words_per_col = num_rows.div_ceil(64);
        let max_pivots = num_rows.min(num_cols);
        let pivot_word_len =
            max_pivots
                .checked_mul(words_per_col)
                .ok_or_else(|| TdaError::MatrixError {
                    reason: "boundary workspace word capacity overflow".to_string(),
                })?;

        Ok(Self {
            num_rows,
            num_cols,
            words_per_col,
            max_pivots,
            input_cols: vec![[EMPTY_FACE; MAX_INLINE_BOUNDARY_FACES]; num_cols],
            cleared_cols: vec![false; num_cols],
            pivot_words: vec![0_u64; pivot_word_len],
            row_to_slot: vec![NO_INDEX; num_rows],
            row_to_pivot_col: vec![NO_INDEX; num_rows],
            col_pivots: vec![NO_INDEX; num_cols],
            work_col: vec![0_u64; words_per_col],
            rank: 0,
        })
    }

    /// Sets the non-zero boundary row indices for column `col` (at most 3 faces for $\partial_1$/$\partial_2$).
    ///
    /// # Errors
    ///
    /// Returns [`TdaError::MatrixError`] if `col >= num_cols`, `faces.len() > 3`, or any row index is out of bounds.
    #[verified_engine::verified(alloc_ok = "allocates owned output buffer")]
    pub fn set_column_faces(&mut self, col: usize, faces: &[usize]) -> Result<(), TdaError> {
        if col >= self.num_cols {
            return Err(TdaError::MatrixError {
                reason: format!("column index {col} out of bounds for {}", self.num_cols),
            });
        }
        if faces.len() > MAX_INLINE_BOUNDARY_FACES {
            return Err(TdaError::MatrixError {
                reason: format!(
                    "boundary face count {} exceeds inline limit {MAX_INLINE_BOUNDARY_FACES}",
                    faces.len()
                ),
            });
        }
        let mut packed = [EMPTY_FACE; MAX_INLINE_BOUNDARY_FACES];
        for (slot, &row) in packed.iter_mut().zip(faces.iter()) {
            if row >= self.num_rows {
                return Err(TdaError::MatrixError {
                    reason: format!("row index {row} out of bounds for {}", self.num_rows),
                });
            }
            *slot = row as u32;
        }
        self.input_cols[col] = packed;
        Ok(())
    }

    /// Marks `col` as cleared via the persistent homology twist optimization ($\partial_1 \circ \partial_2 = 0$).
    #[verified_engine::verified]
    pub fn clear_column(&mut self, col: usize) {
        if col < self.num_cols {
            self.cleared_cols[col] = true;
        }
    }

    /// Returns `true` if `col` was marked cleared prior to reduction.
    #[verified_engine::verified]
    pub fn is_cleared(&self, col: usize) -> bool {
        self.cleared_cols.get(col).copied().unwrap_or(false)
    }

    /// Returns the $\mathbb{Z}_2$ rank discovered by the most recent [`Self::reduce`] call.
    #[verified_engine::verified]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Returns the reduced pivot row (`low(col)`) of `col`, or `None` if `col` reduced to zero.
    #[verified_engine::verified]
    pub fn col_pivot(&self, col: usize) -> Option<usize> {
        self.col_pivots.get(col).copied().filter(|&r| r != NO_INDEX)
    }

    /// Returns the column index whose reduced pivot equals `row`, or `None` if `row` is unpaired.
    #[verified_engine::verified]
    pub fn pivot_col_for_row(&self, row: usize) -> Option<usize> {
        self.row_to_pivot_col
            .get(row)
            .copied()
            .filter(|&c| c != NO_INDEX)
    }

    /// Executes exact left-to-right $\mathbb{Z}_2$ column reduction with zero heap allocations
    /// and returns $\operatorname{rank}_{\mathbb{Z}_2}(\partial)$.
    #[verified_engine::verified]
    pub fn reduce(&mut self) -> usize {
        self.reduce_with_max_rank(self.max_pivots)
    }

    #[verified_engine::verified]
    pub(crate) fn reduce_with_max_rank(&mut self, max_rank: usize) -> usize {
        self.row_to_slot.fill(NO_INDEX);
        self.row_to_pivot_col.fill(NO_INDEX);
        self.col_pivots.fill(NO_INDEX);
        self.work_col.fill(0);
        self.rank = 0;

        let target_rank = max_rank.min(self.max_pivots);
        if self.num_rows == 0 || self.num_cols == 0 || self.words_per_col == 0 || target_rank == 0 {
            return 0;
        }

        for col in 0..self.num_cols {
            if self.rank >= target_rank {
                break;
            }
            if !self.cleared_cols[col] {
                self.reduce_single_column(col);
            }
        }

        debug_assert!(self.rank <= target_rank);
        self.rank
    }

    #[verified_engine::verified]
    fn reduce_single_column(&mut self, col: usize) {
        let Some(mut top_word) = self.load_input_column(col) else {
            return;
        };

        for _step in 0..=self.num_rows {
            let Some((p_word, p_row)) = highest_set_bit(&self.work_col, top_word) else {
                break;
            };
            top_word = p_word;
            let existing_slot = self.row_to_slot[p_row];
            if existing_slot != NO_INDEX {
                self.xor_pivot_slot(existing_slot, p_word);
            } else {
                self.store_new_pivot(col, p_row, p_word);
                break;
            }
        }

        self.work_col[..=top_word].fill(0);
    }

    #[verified_engine::verified]
    fn load_input_column(&mut self, col: usize) -> Option<usize> {
        let faces = self.input_cols[col];
        let mut max_word = 0_usize;
        let mut any_set = false;
        for raw_row in faces {
            if raw_row != EMPTY_FACE {
                let row = raw_row as usize;
                let word_idx = row >> 6;
                let bit_mask = 1_u64 << (row & 63);
                self.work_col[word_idx] ^= bit_mask;
                if !any_set || word_idx > max_word {
                    max_word = word_idx;
                    any_set = true;
                }
            }
        }
        any_set.then_some(max_word)
    }

    #[verified_engine::verified]
    fn xor_pivot_slot(&mut self, slot: usize, up_to_word: usize) {
        let start = slot * self.words_per_col;
        let end = start + up_to_word + 1;
        let pivot_slice = &self.pivot_words[start..end];
        for (dst, &src) in self.work_col[..=up_to_word].iter_mut().zip(pivot_slice) {
            *dst ^= src;
        }
    }

    #[verified_engine::verified]
    fn store_new_pivot(&mut self, col: usize, pivot_row: usize, top_word: usize) {
        let slot = self.rank;
        assert!(slot < self.max_pivots, "pivot slot within max_pivots bound");
        let start = slot * self.words_per_col;
        let active_end = start + top_word + 1;
        let full_end = start + self.words_per_col;

        self.pivot_words[start..active_end].copy_from_slice(&self.work_col[..=top_word]);
        if active_end < full_end {
            self.pivot_words[active_end..full_end].fill(0);
        }

        self.row_to_slot[pivot_row] = slot;
        self.row_to_pivot_col[pivot_row] = col;
        self.col_pivots[col] = pivot_row;
        self.rank += 1;
    }
}

#[verified_engine::verified]
fn highest_set_bit(words: &[u64], max_word: usize) -> Option<(usize, usize)> {
    let limit = max_word.min(words.len().saturating_sub(1));
    for rev in 0..=limit {
        let w_idx = limit - rev;
        let word = words[w_idx];
        if word != 0 {
            let bit_in_word = 63 - (word.leading_zeros() as usize);
            let row = (w_idx << 6) + bit_in_word;
            return Some((w_idx, row));
        }
    }
    None
}

/// Computes $\operatorname{rank}_{\mathbb{Z}_2}(\partial)$ for a boundary matrix specified by column face lists.
///
/// # Errors
///
/// Propagates [`TdaError::MatrixError`] if any face list exceeds inline bounds or references an invalid row.
#[verified_engine::verified]
pub fn rank_z2_boundary(num_rows: usize, columns: &[Vec<usize>]) -> Result<usize, TdaError> {
    let mut ws = Z2BoundaryWorkspace::new(num_rows, columns.len())?;
    for (col_idx, faces) in columns.iter().enumerate() {
        ws.set_column_faces(col_idx, faces)?;
    }
    Ok(ws.reduce())
}

#[derive(Clone, Copy, Debug)]
pub(super) struct FilteredEdge {
    pub(super) diameter: f64,
    pub(super) u: u16,
    pub(super) v: u16,
}

impl FilteredEdge {
    #[verified_engine::verified]
    pub(super) fn cmp_filtration(&self, other: &Self) -> std::cmp::Ordering {
        self.diameter
            .total_cmp(&other.diameter)
            .then_with(|| (self.u, self.v).cmp(&(other.u, other.v)))
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct FilteredTriangle {
    pub(super) diameter: f64,
    pub(super) vertices: [u16; 3],
}

impl FilteredTriangle {
    #[verified_engine::verified]
    pub(super) fn cmp_filtration(&self, other: &Self) -> std::cmp::Ordering {
        self.diameter
            .total_cmp(&other.diameter)
            .then_with(|| self.vertices.cmp(&other.vertices))
    }
}

#[verified_engine::verified]
pub(super) fn build_partial_2_workspace(
    n: usize,
    edge_count: usize,
    edge_lookup: &[u32],
    triangles: &[FilteredTriangle],
) -> Result<Z2BoundaryWorkspace, TdaError> {
    let mut ws = Z2BoundaryWorkspace::new(edge_count, triangles.len())?;
    for (t_idx, tri) in triangles.iter().enumerate() {
        let u = tri.vertices[0] as usize;
        let v = tri.vertices[1] as usize;
        let w = tri.vertices[2] as usize;
        let row_u = u * n;
        let row_v = v * n;
        let e0 = edge_lookup[row_u + v] as usize;
        let e1 = edge_lookup[row_u + w] as usize;
        let e2 = edge_lookup[row_v + w] as usize;
        ws.set_column_faces(t_idx, &[e0, e1, e2])?;
    }
    Ok(ws)
}

#[verified_engine::verified]
pub(super) fn build_cleared_partial_1_workspace(
    n: usize,
    edges: &[FilteredEdge],
    partial_2: &Z2BoundaryWorkspace,
) -> Result<Z2BoundaryWorkspace, TdaError> {
    let mut ws = Z2BoundaryWorkspace::new(n, edges.len())?;
    for (e_idx, edge) in edges.iter().enumerate() {
        if partial_2.pivot_col_for_row(e_idx).is_some() {
            ws.clear_column(e_idx);
        } else {
            ws.set_column_faces(e_idx, &[edge.u as usize, edge.v as usize])?;
        }
    }
    Ok(ws)
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub(super) fn reduce_incremental_triangle(
    e0: usize,
    e1: usize,
    e_max: usize,
    edge_words: usize,
    col_scratch: &mut [u64],
    pivot_for_row: &mut [usize],
    basis_words: &mut [Vec<u64>],
    basis_rank: &mut usize,
) {
    if pivot_for_row[e_max] == usize::MAX {
        let slot = *basis_rank;
        pivot_for_row[e_max] = slot;
        let row_words = &mut basis_words[slot][..edge_words];
        row_words.fill(0);
        row_words[e0 >> 6] ^= 1_u64 << (e0 & 63);
        row_words[e1 >> 6] ^= 1_u64 << (e1 & 63);
        row_words[e_max >> 6] ^= 1_u64 << (e_max & 63);
        *basis_rank += 1;
        return;
    }

    let cur_words = (e_max >> 6) + 1;
    col_scratch[..cur_words].fill(0);
    col_scratch[e0 >> 6] ^= 1_u64 << (e0 & 63);
    col_scratch[e1 >> 6] ^= 1_u64 << (e1 & 63);
    col_scratch[e_max >> 6] ^= 1_u64 << (e_max & 63);

    if reduce_scratch_words(
        cur_words,
        edge_words,
        *basis_rank,
        col_scratch,
        pivot_for_row,
        basis_words,
    ) {
        *basis_rank += 1;
    }
}

#[verified_engine::verified]
fn reduce_scratch_words(
    cur_words: usize,
    edge_words: usize,
    basis_idx: usize,
    col_scratch: &mut [u64],
    pivot_for_row: &mut [usize],
    basis_words: &mut [Vec<u64>],
) -> bool {
    for wi in (0..cur_words).rev() {
        while col_scratch[wi] != 0 {
            let bit = 63 - col_scratch[wi].leading_zeros() as usize;
            let pivot_row = (wi << 6) | bit;
            let existing = pivot_for_row[pivot_row];
            if existing == usize::MAX {
                pivot_for_row[pivot_row] = basis_idx;
                let dst = &mut basis_words[basis_idx];
                dst[..cur_words].copy_from_slice(&col_scratch[..cur_words]);
                if cur_words < edge_words {
                    dst[cur_words..edge_words].fill(0);
                }
                return true;
            }
            let existing_words = &basis_words[existing];
            for w in 0..=wi {
                col_scratch[w] ^= existing_words[w];
            }
        }
    }
    false
}

// [cite:clinical_trials_statistics]

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    #[verified_engine::verified]
    fn test_k4_boundary_rank_is_three() {
        let cols = vec![vec![0, 1, 3], vec![0, 2, 4], vec![1, 2, 5], vec![3, 4, 5]];
        assert_eq!(rank_z2_boundary(6, &cols).unwrap(), 3);
    }

    #[test]
    #[verified_engine::verified]
    fn test_zero_allocation_inside_reduce_loop() {
        // Preallocate workspace with >64 rows (multi-word bitsets) before locking allocations
        let mut ws = Z2BoundaryWorkspace::new(130, 64).unwrap();
        for col in 0..64 {
            ws.set_column_faces(col, &[col, col + 32, col + 65])
                .unwrap();
        }
        ws.set_column_faces(63, &[0, 32, 65]).unwrap(); // linearly dependent duplicate

        let (rank, _metrics) = VerifiedEngine::run_verified(|| ws.reduce());
        assert_eq!(rank, 63);
        assert!(ws.col_pivot(63).is_none());
    }
}
