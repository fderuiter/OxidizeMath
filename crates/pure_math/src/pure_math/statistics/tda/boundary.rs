//! Preallocated bit-packed $\mathbb{Z}_2$ boundary matrix reduction workspace.
//!
//! Performs left-to-right column reduction over $\mathbb{Z}_2$ using packed 64-bit words
//! (`u64` XOR operations) without allocating heap memory inside the reduction loop,
//! satisfying `VerifiedAllocator` post-initialization memory locks.

use crate::error::TdaError;
use math_commons::math_kernel::types::flatten_2d_index;

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
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::tda::Z2BoundaryWorkspace;
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
    non_pivot_rows: Vec<bool>,
    col_pivots: Vec<usize>,
    work_col: Vec<u64>,
    min_possible_pivot: usize,
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
            non_pivot_rows: vec![false; num_rows],
            col_pivots: vec![NO_INDEX; num_cols],
            work_col: vec![0_u64; words_per_col],
            min_possible_pivot: 0,
            rank: 0,
        })
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
    pub(super) fn reduce_with_max_rank(&mut self, max_rank: usize) -> usize {
        self.row_to_slot.fill(NO_INDEX);
        self.row_to_pivot_col.fill(NO_INDEX);
        self.col_pivots.fill(NO_INDEX);
        self.work_col.fill(0);
        self.min_possible_pivot = 0;
        self.advance_min_possible_pivot();
        self.rank = 0;

        let cap = max_rank.min(self.max_pivots);
        if self.num_rows == 0 || self.num_cols == 0 || self.words_per_col == 0 || cap == 0 {
            return 0;
        }
        self.pivot_words[..cap * self.words_per_col].fill(0);

        for col in 0..self.num_cols {
            if self.rank >= cap {
                break;
            }
            if !self.cleared_cols[col] {
                self.reduce_single_column(col);
            }
        }

        debug_assert!(self.rank <= cap);
        self.rank
    }

    #[verified_engine::verified]
    fn reduce_single_column(&mut self, col: usize) {
        let Some((faces, mut top_word, first_row)) = self.dedup_input_faces(col) else {
            return;
        };
        if self.row_to_slot[first_row] == NO_INDEX {
            self.store_direct_pivot(col, first_row, faces);
            return;
        }
        if self.min_possible_pivot >= first_row {
            return;
        }
        for raw_row in faces {
            if raw_row != EMPTY_FACE {
                let r = raw_row as usize;
                self.work_col[r >> 6] ^= 1_u64 << (r & 63);
            }
        }
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
    fn dedup_input_faces(&self, col: usize) -> Option<([u32; 3], usize, usize)> {
        let [mut a, mut b, mut c] = self.input_cols[col];
        if a != EMPTY_FACE && a == b {
            (a, b) = (EMPTY_FACE, EMPTY_FACE);
        }
        if a != EMPTY_FACE && a == c {
            (a, c) = (EMPTY_FACE, EMPTY_FACE);
        }
        if b != EMPTY_FACE && b == c {
            (b, c) = (EMPTY_FACE, EMPTY_FACE);
        }
        let max_f = [a, b, c].into_iter().filter(|&x| x != EMPTY_FACE).max()? as usize;
        Some(([a, b, c], max_f >> 6, max_f))
    }

    #[verified_engine::verified]
    fn store_direct_pivot(&mut self, col: usize, pivot_row: usize, faces: [u32; 3]) {
        let slot = self.rank;
        let start = slot * self.words_per_col;
        for raw_row in faces {
            if raw_row != EMPTY_FACE {
                let r = raw_row as usize;
                self.pivot_words[start + (r >> 6)] |= 1_u64 << (r & 63);
            }
        }
        self.row_to_slot[pivot_row] = slot;
        self.row_to_pivot_col[pivot_row] = col;
        self.col_pivots[col] = pivot_row;
        self.rank += 1;
        if pivot_row == self.min_possible_pivot {
            self.advance_min_possible_pivot();
        }
    }

    #[verified_engine::verified]
    fn advance_min_possible_pivot(&mut self) {
        while self.min_possible_pivot < self.num_rows
            && (self.non_pivot_rows[self.min_possible_pivot]
                || self.row_to_slot[self.min_possible_pivot] != NO_INDEX)
        {
            self.min_possible_pivot += 1;
        }
    }

    #[verified_engine::verified]
    fn xor_pivot_slot(&mut self, slot: usize, up_to_word: usize) {
        let start = slot * self.words_per_col;
        let pivot_slice = &self.pivot_words[start..=start + up_to_word];
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
        self.pivot_words[start..active_end].copy_from_slice(&self.work_col[..=top_word]);
        self.row_to_slot[pivot_row] = slot;
        self.row_to_pivot_col[pivot_row] = col;
        self.col_pivots[col] = pivot_row;
        self.rank += 1;
        if pivot_row == self.min_possible_pivot {
            self.advance_min_possible_pivot();
        }
    }
}

#[verified_engine::verified]
fn highest_set_bit(words: &[u64], max_word: usize) -> Option<(usize, usize)> {
    let limit = max_word.min(words.len().saturating_sub(1));
    for (w_idx, &word) in words[..=limit].iter().enumerate().rev() {
        if word != 0 {
            let bit_in_word = 63 - (word.leading_zeros() as usize);
            return Some((w_idx, (w_idx << 6) + bit_in_word));
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

#[derive(Clone, Copy, Debug)]
pub(super) struct FilteredTriangle {
    pub(super) diameter: f64,
    pub(super) vertices: [u16; 3],
}

#[verified_engine::verified]
pub(super) fn collect_sorted_edges(dist: &[Vec<f64>]) -> (Vec<FilteredEdge>, Vec<u32>, f64) {
    let n = dist.len();
    let mut edges = Vec::with_capacity(n.saturating_mul(n.saturating_sub(1)) / 2);
    let mut max_radius = 0.0_f64;
    for (i, row) in dist.iter().enumerate() {
        for (j, &d) in (i + 1..n).zip(&row[i + 1..n]) {
            if d > max_radius {
                max_radius = d;
            }
            edges.push(FilteredEdge {
                diameter: d,
                u: i as u16,
                v: j as u16,
            });
        }
    }
    edges.sort_unstable_by(|a, b| {
        a.diameter
            .total_cmp(&b.diameter)
            .then_with(|| (a.u, a.v).cmp(&(b.u, b.v)))
    });
    let mut lookup = vec![u32::MAX; n * n];
    for (idx, edge) in edges.iter().enumerate() {
        let (u, v) = (edge.u as usize, edge.v as usize);
        lookup[flatten_2d_index(u, v, n)] = idx as u32;
        lookup[flatten_2d_index(v, u, n)] = idx as u32;
    }
    (edges, lookup, max_radius)
}

#[verified_engine::verified]
pub(super) fn collect_sorted_triangles(
    dist: &[Vec<f64>],
    max_triangles: usize,
) -> (Vec<FilteredTriangle>, bool) {
    let n = dist.len();
    if n < 3 {
        return (Vec::new(), false);
    }
    if max_triangles == 0 {
        return (Vec::new(), true);
    }
    let cmp_tri = |a: &FilteredTriangle, b: &FilteredTriangle| {
        a.diameter
            .total_cmp(&b.diameter)
            .then_with(|| a.vertices.cmp(&b.vertices))
    };
    let flush_cap = max_triangles.saturating_mul(2).max(1);
    let mut buf = Vec::with_capacity(flush_cap.min(200_000));
    let mut truncated = false;
    for (i, row_i) in dist.iter().enumerate() {
        for (j, &d_ij) in (i + 1..n).zip(&row_i[i + 1..n]) {
            let row_j = &dist[j];
            for (k, (&d_ik, &d_jk)) in (j + 1..n).zip(row_i[j + 1..n].iter().zip(&row_j[j + 1..n]))
            {
                buf.push(FilteredTriangle {
                    diameter: d_ij.max(d_ik).max(d_jk),
                    vertices: [i as u16, j as u16, k as u16],
                });
                if buf.len() >= flush_cap && buf.len() > max_triangles {
                    truncated = true;
                    buf.select_nth_unstable_by(max_triangles, cmp_tri);
                    buf.truncate(max_triangles);
                }
            }
        }
    }
    if buf.len() > max_triangles {
        truncated = true;
        buf.select_nth_unstable_by(max_triangles, cmp_tri);
        buf.truncate(max_triangles);
    }
    buf.sort_unstable_by(cmp_tri);
    (buf, truncated)
}

#[verified_engine::verified]
pub(super) fn build_partial_2_workspace(
    n: usize,
    edges: &[FilteredEdge],
    edge_lookup: &[u32],
    triangles: &[FilteredTriangle],
) -> Result<Z2BoundaryWorkspace, TdaError> {
    let mut ws = Z2BoundaryWorkspace::new(edges.len(), triangles.len())?;
    let mut parent: Vec<usize> = (0..n).collect();
    for (e_idx, edge) in edges.iter().enumerate() {
        let (mut rx, mut ry) = (edge.u as usize, edge.v as usize);
        while parent[rx] != rx {
            parent[rx] = parent[parent[rx]];
            rx = parent[rx];
        }
        while parent[ry] != ry {
            parent[ry] = parent[parent[ry]];
            ry = parent[ry];
        }
        if rx != ry {
            parent[ry] = rx;
            ws.non_pivot_rows[e_idx] = true;
        }
    }
    for (t_idx, tri) in triangles.iter().enumerate() {
        let [u, v, w] = tri.vertices.map(usize::from);
        let faces = [
            edge_lookup[flatten_2d_index(u, v, n)] as usize,
            edge_lookup[flatten_2d_index(u, w, n)] as usize,
            edge_lookup[flatten_2d_index(v, w, n)] as usize,
        ];
        ws.set_column_faces(t_idx, &faces)?;
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
