//! Arena region layout and incremental $\mathbb{Z}_2$ triangle reduction helpers for TDA workspaces.

use super::homology::uf_reset_slices;
use math_memory::{ArenaLayout, ArenaRegion, OwnedByteArena};

pub(super) const DEFAULT_RADII_SCRATCH_CAP: usize = 512;

#[derive(Debug, Clone, Copy)]
pub(super) struct TdaRegions {
    pub(super) dist: ArenaRegion<f64>,
    pub(super) sorted_edges: ArenaRegion<[u64; 2]>,
    pub(super) adj_words: ArenaRegion<u64>,
    pub(super) uf_parent: ArenaRegion<usize>,
    pub(super) uf_rank: ArenaRegion<usize>,
    pub(super) edges: ArenaRegion<[usize; 2]>,
    pub(super) edge_lookup: ArenaRegion<usize>,
    pub(super) triangles: ArenaRegion<[usize; 3]>,
    pub(super) pivot_for_row: ArenaRegion<usize>,
    pub(super) basis_words: ArenaRegion<u64>,
    pub(super) col_scratch: ArenaRegion<u64>,
    pub(super) component_births: ArenaRegion<f64>,
    pub(super) hole_births: ArenaRegion<f64>,
    pub(super) radii_scratch: ArenaRegion<f64>,
}

pub(super) struct AllocatedTdaArena {
    pub(super) cap: usize,
    pub(super) max_vertex_words: usize,
    pub(super) max_edge_words: usize,
    pub(super) arena: OwnedByteArena,
    pub(super) regions: TdaRegions,
}

#[verified_engine::verified]
pub(super) fn allocate_tda_arena(max_points: usize) -> AllocatedTdaArena {
    let cap = max_points.max(1);
    let max_edges = cap.saturating_mul(cap.saturating_sub(1)) / 2;
    let edge_slots = max_edges.max(1);
    let max_tris = cap
        .saturating_mul(cap.saturating_sub(1))
        .saturating_mul(cap.saturating_sub(2))
        / 6;
    let tri_slots = max_tris.max(1);
    let max_edge_words = max_edges.div_ceil(64).max(1);
    let max_vertex_words = cap.div_ceil(64).max(1);

    let mut layout = ArenaLayout::new();
    let dist = layout.reserve::<f64>(cap * cap).expect("tda dist layout");
    let sorted_edges = layout
        .reserve::<[u64; 2]>(edge_slots)
        .expect("tda sorted_edges layout");
    let adj_words = layout
        .reserve::<u64>(cap * max_vertex_words)
        .expect("tda adj_words layout");
    let uf_parent = layout.reserve::<usize>(cap).expect("tda uf_parent layout");
    let uf_rank = layout.reserve::<usize>(cap).expect("tda uf_rank layout");
    let edges = layout
        .reserve::<[usize; 2]>(edge_slots)
        .expect("tda edges layout");
    let edge_lookup = layout
        .reserve::<usize>(cap * cap)
        .expect("tda edge_lookup layout");
    let triangles = layout
        .reserve::<[usize; 3]>(tri_slots)
        .expect("tda triangles layout");
    let pivot_for_row = layout
        .reserve::<usize>(edge_slots)
        .expect("tda pivot_for_row layout");
    let basis_words = layout
        .reserve::<u64>(edge_slots * max_edge_words)
        .expect("tda basis_words layout");
    let col_scratch = layout
        .reserve::<u64>(max_edge_words)
        .expect("tda col_scratch layout");
    let component_births = layout
        .reserve::<f64>(cap)
        .expect("tda component_births layout");
    let hole_births = layout
        .reserve::<f64>(edge_slots)
        .expect("tda hole_births layout");
    let radii_scratch = layout
        .reserve::<f64>(DEFAULT_RADII_SCRATCH_CAP)
        .expect("tda radii_scratch layout");

    let mut arena = OwnedByteArena::from_layout(&layout);
    {
        let mut cur = arena.raw_cursor();
        let uf_p = cur.region_mut(uf_parent);
        let uf_r = cur.region_mut(uf_rank);
        let _ = uf_reset_slices(uf_p, uf_r, cap);
        cur.region_mut(edge_lookup).fill(usize::MAX);
        cur.region_mut(pivot_for_row).fill(usize::MAX);
    }

    AllocatedTdaArena {
        cap,
        max_vertex_words,
        max_edge_words,
        arena,
        regions: TdaRegions {
            dist,
            sorted_edges,
            adj_words,
            uf_parent,
            uf_rank,
            edges,
            edge_lookup,
            triangles,
            pivot_for_row,
            basis_words,
            col_scratch,
            component_births,
            hole_births,
            radii_scratch,
        },
    }
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub(super) fn reduce_incremental_triangle(
    e0: usize,
    e1: usize,
    e_max: usize,
    edge_words: usize,
    basis_stride: usize,
    col_scratch: &mut [u64],
    pivot_for_row: &mut [usize],
    basis_words: &mut [u64],
    basis_rank: &mut usize,
) {
    if pivot_for_row[e_max] == usize::MAX {
        let slot = *basis_rank;
        pivot_for_row[e_max] = slot;
        let slot_off = slot * basis_stride;
        let row_words = &mut basis_words[slot_off..slot_off + edge_words];
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
        basis_stride,
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
    basis_stride: usize,
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
                let dst_off = basis_idx * basis_stride;
                let dst = &mut basis_words[dst_off..dst_off + edge_words];
                dst[..cur_words].copy_from_slice(&col_scratch[..cur_words]);
                if cur_words < edge_words {
                    dst[cur_words..edge_words].fill(0);
                }
                return true;
            }
            let ex_off = existing * basis_stride;
            let existing_words = &basis_words[ex_off..ex_off + edge_words];
            for w in 0..=wi {
                col_scratch[w] ^= existing_words[w];
            }
        }
    }
    false
}

// [cite:clinical_trials_statistics]
