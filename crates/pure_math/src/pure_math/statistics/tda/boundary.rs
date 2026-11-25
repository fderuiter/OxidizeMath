//! Preallocated bit-packed $\mathbb{Z}_2$ boundary matrix reduction workspace.
//!
//! Performs left-to-right column reduction over $\mathbb{Z}_2$ using packed 64-bit words
//! (`u64` XOR operations) without allocating heap memory inside the reduction loop,
//! satisfying `VerifiedAllocator` post-initialization memory locks.

use crate::error::TdaError;
use math_memory::{ArenaLayout, ArenaRegion, FlagSlice, OwnedByteArena};

/// Maximum number of boundary faces stored inline for up to 2-simplices (triangles have 3 edges).
pub const MAX_INLINE_BOUNDARY_FACES: usize = 3;

const NO_INDEX: usize = usize::MAX;
const EMPTY_FACE: u32 = u32::MAX;

struct Z2Slices<'a> {
    pivot_words: &'a mut [u64],
    work_col: &'a mut [u64],
    row_to_slot: &'a mut [usize],
    row_to_pivot_col: &'a mut [usize],
    col_pivots: &'a mut [usize],
    input_cols: &'a mut [[u32; MAX_INLINE_BOUNDARY_FACES]],
    cleared_cols: FlagSlice<'a>,
}

impl Z2Slices<'_> {
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

    fn xor_pivot_slot(&mut self, slot: usize, words_per_col: usize, up_to_word: usize) {
        let start = slot * words_per_col;
        let end = start + up_to_word + 1;
        let pivot_slice = &self.pivot_words[start..end];
        for (dst, &src) in self.work_col[..=up_to_word].iter_mut().zip(pivot_slice) {
            *dst ^= src;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn store_new_pivot(
        &mut self,
        col: usize,
        pivot_row: usize,
        top_word: usize,
        words_per_col: usize,
        max_pivots: usize,
        rank: &mut usize,
    ) {
        let slot = *rank;
        assert!(slot < max_pivots, "pivot slot within max_pivots bound");
        let start = slot * words_per_col;
        let active_end = start + top_word + 1;
        let full_end = start + words_per_col;

        self.pivot_words[start..active_end].copy_from_slice(&self.work_col[..=top_word]);
        if active_end < full_end {
            self.pivot_words[active_end..full_end].fill(0);
        }

        self.row_to_slot[pivot_row] = slot;
        self.row_to_pivot_col[pivot_row] = col;
        self.col_pivots[col] = pivot_row;
        *rank += 1;
    }

    fn reduce_single_column(
        &mut self,
        col: usize,
        num_rows: usize,
        words_per_col: usize,
        max_pivots: usize,
        rank: &mut usize,
    ) {
        let Some(mut top_word) = self.load_input_column(col) else {
            return;
        };

        for _step in 0..=num_rows {
            let Some((p_word, p_row)) = highest_set_bit(self.work_col, top_word) else {
                break;
            };
            top_word = p_word;
            let existing_slot = self.row_to_slot[p_row];
            if existing_slot != NO_INDEX {
                self.xor_pivot_slot(existing_slot, words_per_col, p_word);
            } else {
                self.store_new_pivot(col, p_row, p_word, words_per_col, max_pivots, rank);
                break;
            }
        }

        self.work_col[..=top_word].fill(0);
    }
}

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
    arena: OwnedByteArena,
    row_to_pivot_col_reg: ArenaRegion<usize>,
    col_pivots_reg: ArenaRegion<usize>,
    input_cols_reg: ArenaRegion<[u32; MAX_INLINE_BOUNDARY_FACES]>,
    cleared_cols_reg: ArenaRegion<u8>,
    rank: usize,
}

impl Z2BoundaryWorkspace {
    /// Allocates a bounded $\mathbb{Z}_2$ reduction workspace for a `num_rows × num_cols` boundary matrix.
    ///
    /// # Errors
    ///
    /// Returns [`TdaError::MatrixError`] if dimensions overflow `u32` face indexing.
    #[verified_engine::verified]
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
        let map_err = |_| TdaError::MatrixError {
            reason: "boundary workspace arena layout overflow".to_string(),
        };
        let mut layout = ArenaLayout::new();
        let _ = layout.add_slice::<u64>(pivot_word_len).map_err(map_err)?;
        let _ = layout.add_slice::<u64>(words_per_col).map_err(map_err)?;
        let row_to_slot_reg = layout.add_slice::<usize>(num_rows).map_err(map_err)?;
        let row_to_pivot_col_reg = layout.add_slice::<usize>(num_rows).map_err(map_err)?;
        let col_pivots_reg = layout.add_slice::<usize>(num_cols).map_err(map_err)?;
        let input_cols_reg = layout
            .add_slice::<[u32; MAX_INLINE_BOUNDARY_FACES]>(num_cols)
            .map_err(map_err)?;
        let cleared_cols_reg = layout.add_slice::<u8>(num_cols).map_err(map_err)?;

        let mut arena = OwnedByteArena::with_capacity(layout.total_bytes());
        arena.region_mut(row_to_slot_reg).fill(NO_INDEX);
        arena.region_mut(row_to_pivot_col_reg).fill(NO_INDEX);
        arena.region_mut(col_pivots_reg).fill(NO_INDEX);
        arena
            .region_mut(input_cols_reg)
            .fill([EMPTY_FACE; MAX_INLINE_BOUNDARY_FACES]);

        Ok(Self {
            num_rows,
            num_cols,
            words_per_col,
            max_pivots,
            arena,
            row_to_pivot_col_reg,
            col_pivots_reg,
            input_cols_reg,
            cleared_cols_reg,
            rank: 0,
        })
    }

    fn slices_mut(&mut self) -> Z2Slices<'_> {
        let pw_len = self.max_pivots * self.words_per_col;
        let wpc = self.words_per_col;
        let nr = self.num_rows;
        let nc = self.num_cols;
        let mut scope = self.arena.reborrow();
        let pivot_words = scope.retain_slice::<u64>(pw_len).expect("pivot_words");
        let work_col = scope.retain_slice::<u64>(wpc).expect("work_col");
        let row_to_slot = scope.retain_slice::<usize>(nr).expect("row_to_slot");
        let row_to_pivot_col = scope.retain_slice::<usize>(nr).expect("row_to_pivot_col");
        let col_pivots = scope.retain_slice::<usize>(nc).expect("col_pivots");
        let input_cols = scope
            .retain_slice::<[u32; MAX_INLINE_BOUNDARY_FACES]>(nc)
            .expect("input_cols");
        let cleared_bytes = scope.retain_slice::<u8>(nc).expect("cleared_cols");
        Z2Slices {
            pivot_words,
            work_col,
            row_to_slot,
            row_to_pivot_col,
            col_pivots,
            input_cols,
            cleared_cols: FlagSlice::from_bytes(cleared_bytes),
        }
    }

    /// Sets the non-zero boundary row indices for column `col` (at most 3 faces for $\partial_1$/$\partial_2$).
    ///
    /// # Errors
    ///
    /// Returns [`TdaError::MatrixError`] if `col >= num_cols`, `faces.len() > 3`, or any row index is out of bounds.
    #[verified_engine::verified]
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
        self.arena.region_mut(self.input_cols_reg)[col] = packed;
        Ok(())
    }

    /// Marks `col` as cleared via the persistent homology twist optimization ($\partial_1 \circ \partial_2 = 0$).
    #[verified_engine::verified]
    pub fn clear_column(&mut self, col: usize) {
        if col < self.num_cols {
            self.arena.region_mut(self.cleared_cols_reg)[col] = 1;
        }
    }

    /// Returns `true` if `col` was marked cleared prior to reduction.
    #[verified_engine::verified]
    pub fn is_cleared(&self, col: usize) -> bool {
        self.arena
            .region(self.cleared_cols_reg)
            .get(col)
            .copied()
            .unwrap_or(0)
            != 0
    }

    /// Returns the $\mathbb{Z}_2$ rank discovered by the most recent [`Self::reduce`] call.
    #[verified_engine::verified]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Returns the reduced pivot row (`low(col)`) of `col`, or `None` if `col` reduced to zero.
    #[verified_engine::verified]
    pub fn col_pivot(&self, col: usize) -> Option<usize> {
        self.arena
            .region(self.col_pivots_reg)
            .get(col)
            .copied()
            .filter(|&r| r != NO_INDEX)
    }

    /// Returns the column index whose reduced pivot equals `row`, or `None` if `row` is unpaired.
    #[verified_engine::verified]
    pub fn pivot_col_for_row(&self, row: usize) -> Option<usize> {
        self.arena
            .region(self.row_to_pivot_col_reg)
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
        let nr = self.num_rows;
        let nc = self.num_cols;
        let wpc = self.words_per_col;
        let mp = self.max_pivots;
        let target_rank = max_rank.min(mp);
        let mut rank = 0_usize;
        {
            let mut s = self.slices_mut();
            s.row_to_slot.fill(NO_INDEX);
            s.row_to_pivot_col.fill(NO_INDEX);
            s.col_pivots.fill(NO_INDEX);
            s.work_col.fill(0);
            if nr > 0 && nc > 0 && wpc > 0 && target_rank > 0 {
                for col in 0..nc {
                    if rank >= target_rank {
                        break;
                    }
                    if !s.cleared_cols.get(col) {
                        s.reduce_single_column(col, nr, wpc, mp, &mut rank);
                    }
                }
            }
        }
        self.rank = rank;
        debug_assert!(self.rank <= target_rank);
        self.rank
    }
}

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

pub(super) use super::boundary_filtration::{
    FilteredEdge, FilteredTriangle, build_cleared_partial_1_workspace, build_partial_2_workspace,
    reduce_incremental_triangle,
};

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
