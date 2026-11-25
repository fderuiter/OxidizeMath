//! Filtration boundary builders and incremental $\mathbb{Z}_2$ triangle reduction helpers.

use super::boundary::Z2BoundaryWorkspace;
use crate::error::TdaError;

#[derive(Clone, Copy, Debug)]
pub(super) struct FilteredEdge {
    pub(super) diameter: f64,
    pub(super) u: u16,
    pub(super) v: u16,
}

impl FilteredEdge {
    #[inline]
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
    #[inline]
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
    basis_words: &mut [u64],
    basis_rank: &mut usize,
) {
    if pivot_for_row[e_max] == usize::MAX {
        let slot = *basis_rank;
        pivot_for_row[e_max] = slot;
        let base = slot * edge_words;
        let row_words = &mut basis_words[base..base + edge_words];
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
    basis_words: &mut [u64],
) -> bool {
    for wi in (0..cur_words).rev() {
        while col_scratch[wi] != 0 {
            let bit = 63 - col_scratch[wi].leading_zeros() as usize;
            let pivot_row = (wi << 6) | bit;
            let existing = pivot_for_row[pivot_row];
            if existing == usize::MAX {
                pivot_for_row[pivot_row] = basis_idx;
                let base = basis_idx * edge_words;
                let dst = &mut basis_words[base..base + edge_words];
                dst[..cur_words].copy_from_slice(&col_scratch[..cur_words]);
                if cur_words < edge_words {
                    dst[cur_words..edge_words].fill(0);
                }
                return true;
            }
            let ex_base = existing * edge_words;
            let existing_words = &basis_words[ex_base..ex_base + edge_words];
            for w in 0..=wi {
                col_scratch[w] ^= existing_words[w];
            }
        }
    }
    false
}

// [cite:clinical_trials_statistics]
