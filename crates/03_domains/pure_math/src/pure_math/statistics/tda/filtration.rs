//! Bitset-accelerated simplex filtration and incremental zero-allocation streaming helpers.

use super::boundary::{
    FilteredEdge, FilteredTriangle, Z2BoundaryWorkspace, build_cleared_partial_1_workspace,
    build_partial_2_workspace, reduce_incremental_triangle,
};
use super::complex::validate_distance_matrix;
use super::homology::UnionFind;
use super::persistence::{RipsInterval, RipsPersistenceResult};
use crate::error::TdaError;

pub(super) struct FilteredSimplexSet {
    pub(super) edges: Vec<FilteredEdge>,
    pub(super) edge_lookup: Vec<u32>,
    pub(super) triangles: Vec<FilteredTriangle>,
    pub(super) truncated_triangles: bool,
    pub(super) max_radius: f64,
    pub(super) final_beta0: usize,
}

#[verified_engine::verified]
pub(super) fn collect_filtered_simplices_up_to(
    dist: &[Vec<f64>],
    cutoff: f64,
    max_triangles: usize,
) -> FilteredSimplexSet {
    let n = dist.len();
    let words = n.div_ceil(64).max(1);
    let mut adj_bits = vec![vec![0_u64; words]; n];
    let mut sym_bits = vec![vec![0_u64; words]; n];
    let mut uf = UnionFind::new(n);
    let mut edges = Vec::with_capacity(n.saturating_mul(n.saturating_sub(1)) / 2);
    let mut max_radius = 0.0_f64;

    for (i, row) in dist.iter().enumerate().take(n) {
        for (j, &d) in row.iter().enumerate().take(n).skip(i + 1) {
            if d > max_radius {
                max_radius = d;
            }
            if d <= cutoff {
                edges.push(FilteredEdge {
                    diameter: d,
                    u: i as u16,
                    v: j as u16,
                });
                adj_bits[i][j >> 6] |= 1_u64 << (j & 63);
                sym_bits[i][j >> 6] |= 1_u64 << (j & 63);
                sym_bits[j][i >> 6] |= 1_u64 << (i & 63);
                uf.union(i, j);
            }
        }
    }

    edges.sort_unstable_by(FilteredEdge::cmp_filtration);
    let edge_lookup = build_edge_lookup(n, &edges);
    let (mut triangles, truncated_triangles) = if edges.len() >= 3 {
        collect_triangles_from_bitsets(dist, &adj_bits, max_triangles)
    } else {
        (Vec::new(), false)
    };
    triangles.retain(|tri| !is_tetrahedron_dominated(dist, &sym_bits, words, tri));

    FilteredSimplexSet {
        edges,
        edge_lookup,
        triangles,
        truncated_triangles,
        max_radius,
        final_beta0: uf.count_components(),
    }
}

#[verified_engine::verified]
fn build_edge_lookup(n: usize, edges: &[FilteredEdge]) -> Vec<u32> {
    let mut lookup = vec![u32::MAX; n * n];
    for (idx, edge) in edges.iter().enumerate() {
        let u = edge.u as usize;
        let v = edge.v as usize;
        let row_u = u * n;
        let row_v = v * n;
        lookup[row_u + v] = idx as u32;
        lookup[row_v + u] = idx as u32;
    }
    lookup
}

#[verified_engine::verified]
fn collect_triangles_from_bitsets(
    dist: &[Vec<f64>],
    adj_bits: &[Vec<u64>],
    max_triangles: usize,
) -> (Vec<FilteredTriangle>, bool) {
    let n = dist.len();
    if n < 3 {
        return (Vec::new(), false);
    }
    let words = n.div_ceil(64).max(1);
    let flush_cap = max_triangles.saturating_mul(2).max(1);
    let mut buf = Vec::with_capacity(flush_cap.min(200_000));
    let mut truncated = false;

    for i in 0..n {
        let row_i = &dist[i];
        let bits_i = &adj_bits[i];
        for wi in 0..words {
            let mut word_j = bits_i[wi];
            while word_j != 0 {
                let tz_j = word_j.trailing_zeros() as usize;
                word_j &= word_j - 1;
                let j = (wi << 6) | tz_j;
                if scan_common_forward_neighbors(
                    dist,
                    row_i,
                    bits_i,
                    &adj_bits[j],
                    i,
                    j,
                    words,
                    max_triangles,
                    flush_cap,
                    &mut buf,
                    &mut truncated,
                ) {
                    return (Vec::new(), true);
                }
            }
        }
    }

    if buf.len() > max_triangles {
        truncated = true;
        buf.select_nth_unstable_by(max_triangles, FilteredTriangle::cmp_filtration);
        buf.truncate(max_triangles);
    }
    buf.sort_unstable_by(FilteredTriangle::cmp_filtration);
    (buf, truncated)
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn scan_common_forward_neighbors(
    dist: &[Vec<f64>],
    row_i: &[f64],
    bits_i: &[u64],
    bits_j: &[u64],
    i: usize,
    j: usize,
    words: usize,
    max_triangles: usize,
    flush_cap: usize,
    buf: &mut Vec<FilteredTriangle>,
    truncated: &mut bool,
) -> bool {
    let d_ij = row_i[j];
    let row_j = &dist[j];
    let start_w = (j + 1) >> 6;
    let first_shift = (j + 1) & 63;

    for wk in start_w..words {
        let mut common = bits_i[wk] & bits_j[wk];
        if wk == start_w && first_shift > 0 {
            common &= !0_u64 << first_shift;
        }
        while common != 0 {
            if max_triangles == 0 {
                return true;
            }
            let tz_k = common.trailing_zeros() as usize;
            common &= common - 1;
            let k = (wk << 6) | tz_k;
            buf.push(FilteredTriangle {
                diameter: d_ij.max(row_i[k]).max(row_j[k]),
                vertices: [i as u16, j as u16, k as u16],
            });
            if buf.len() >= flush_cap && buf.len() > max_triangles {
                *truncated = true;
                buf.select_nth_unstable_by(max_triangles, FilteredTriangle::cmp_filtration);
                buf.truncate(max_triangles);
            }
        }
    }
    false
}

#[verified_engine::verified]
pub(super) fn prepare_sorted_workspace_edges(
    dist: &[Vec<f64>],
    n: usize,
    out: &mut Vec<FilteredEdge>,
) {
    out.clear();
    for (i, row) in dist.iter().enumerate().take(n) {
        for (j, &d) in row.iter().enumerate().take(n).skip(i + 1) {
            out.push(FilteredEdge {
                diameter: d,
                u: i as u16,
                v: j as u16,
            });
        }
    }
    out.sort_unstable_by(FilteredEdge::cmp_filtration);
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub(super) fn append_edge_triangles_and_reduce(
    u: usize,
    v: usize,
    e_uv: usize,
    n_words: usize,
    edge_words: usize,
    curr_cycles: usize,
    adj_words: &[Vec<u64>],
    edge_lookup: &[Vec<usize>],
    triangles: &mut Vec<[usize; 3]>,
    col_scratch: &mut [u64],
    pivot_for_row: &mut [usize],
    basis_words: &mut [Vec<u64>],
    basis_rank: &mut usize,
) {
    let bits_u = &adj_words[u];
    let bits_v = &adj_words[v];
    for w_idx in 0..n_words {
        let mut common = bits_u[w_idx] & bits_v[w_idx];
        while common != 0 {
            let tz = common.trailing_zeros() as usize;
            common &= common - 1;
            let w = (w_idx << 6) | tz;
            let tri = canonical_triple(u, v, w);
            triangles.push(tri);
            if *basis_rank < curr_cycles
                && !has_earlier_common_apex(bits_u, bits_v, &adj_words[w], w_idx, tz)
            {
                let e_uw = edge_lookup[u][w];
                let e_vw = edge_lookup[v][w];
                reduce_incremental_triangle(
                    e_uw,
                    e_vw,
                    e_uv,
                    edge_words,
                    col_scratch,
                    pivot_for_row,
                    basis_words,
                    basis_rank,
                );
            }
        }
    }
}

#[verified_engine::verified]
fn has_earlier_common_apex(
    bits_u: &[u64],
    bits_v: &[u64],
    bits_w: &[u64],
    w_idx: usize,
    tz: usize,
) -> bool {
    for k in 0..w_idx {
        if (bits_u[k] & bits_v[k] & bits_w[k]) != 0 {
            return true;
        }
    }
    (bits_u[w_idx] & bits_v[w_idx] & bits_w[w_idx] & ((1_u64 << tz) - 1)) != 0
}

#[verified_engine::verified]
fn is_tetrahedron_dominated(
    dist: &[Vec<f64>],
    sym_bits: &[Vec<u64>],
    words: usize,
    tri: &FilteredTriangle,
) -> bool {
    let (u, v, w) = (
        tri.vertices[0] as usize,
        tri.vertices[1] as usize,
        tri.vertices[2] as usize,
    );
    let (su, sv, sw) = (&sym_bits[u], &sym_bits[v], &sym_bits[w]);
    for (wk, ((&bu, &bv), &bw)) in su.iter().zip(sv).zip(sw).enumerate().take(words) {
        let mut common = bu & bv & bw;
        while common != 0 {
            let tz = common.trailing_zeros() as usize;
            common &= common - 1;
            let a = (wk << 6) | tz;
            if face_precedes(dist, u, v, a, tri)
                && face_precedes(dist, u, w, a, tri)
                && face_precedes(dist, v, w, a, tri)
            {
                return true;
            }
        }
    }
    false
}

#[verified_engine::verified]
fn face_precedes(
    dist: &[Vec<f64>],
    x: usize,
    y: usize,
    a: usize,
    target: &FilteredTriangle,
) -> bool {
    let [p, q, r] = canonical_triple(x, y, a);
    FilteredTriangle {
        diameter: dist[p][q].max(dist[p][r]).max(dist[q][r]),
        vertices: [p as u16, q as u16, r as u16],
    }
    .cmp_filtration(target)
    .is_lt()
}

#[verified_engine::verified]
fn canonical_triple(u: usize, v: usize, w: usize) -> [usize; 3] {
    if w < u {
        [w, u, v]
    } else if w < v {
        [u, w, v]
    } else {
        [u, v, w]
    }
}

#[verified_engine::verified]
pub(super) fn compute_rips_persistence_up_to(
    dist: &[Vec<f64>],
    cutoff: f64,
    max_triangles: usize,
) -> Result<RipsPersistenceResult, TdaError> {
    validate_distance_matrix(dist)?;
    let n = dist.len();
    let filtered = collect_filtered_simplices_up_to(dist, cutoff, max_triangles);
    let max_tree_rank = n.saturating_sub(filtered.final_beta0);
    let max_cycle_rank = filtered.edges.len().saturating_sub(max_tree_rank);

    let mut partial_2 = build_partial_2_workspace(
        n,
        filtered.edges.len(),
        &filtered.edge_lookup,
        &filtered.triangles,
    )?;
    partial_2.reduce_with_max_rank(max_cycle_rank);

    let mut partial_1 = build_cleared_partial_1_workspace(n, &filtered.edges, &partial_2)?;
    partial_1.reduce_with_max_rank(max_tree_rank);

    let intervals = assemble_rips_intervals(
        n,
        &filtered.edges,
        &filtered.triangles,
        &partial_1,
        &partial_2,
        filtered.max_radius,
    );
    Ok(RipsPersistenceResult {
        intervals,
        max_radius: filtered.max_radius,
        truncated_triangles: filtered.truncated_triangles,
    })
}

#[verified_engine::verified]
fn assemble_rips_intervals(
    n: usize,
    edges: &[FilteredEdge],
    triangles: &[FilteredTriangle],
    partial_1: &Z2BoundaryWorkspace,
    partial_2: &Z2BoundaryWorkspace,
    max_radius: f64,
) -> Vec<RipsInterval> {
    let mut intervals = Vec::with_capacity(n + edges.len().min(n * 4));
    push_h0_intervals(n, edges, partial_1, max_radius, &mut intervals);
    push_h1_intervals(
        edges,
        triangles,
        partial_1,
        partial_2,
        max_radius,
        &mut intervals,
    );
    intervals.sort_by(|a, b| {
        a.dimension
            .cmp(&b.dimension)
            .then_with(|| a.birth.total_cmp(&b.birth))
            .then_with(|| a.death.total_cmp(&b.death))
            .then_with(|| a.is_essential.cmp(&b.is_essential))
    });
    intervals
}

#[verified_engine::verified]
fn push_h0_intervals(
    n: usize,
    edges: &[FilteredEdge],
    partial_1: &Z2BoundaryWorkspace,
    max_radius: f64,
    intervals: &mut Vec<RipsInterval>,
) {
    for v in 0..n {
        if let Some(e_idx) = partial_1.pivot_col_for_row(v) {
            let death = edges[e_idx].diameter;
            if death > 0.0 {
                intervals.push(RipsInterval {
                    dimension: 0,
                    birth: 0.0,
                    death,
                    is_essential: false,
                });
            }
        } else {
            intervals.push(RipsInterval {
                dimension: 0,
                birth: 0.0,
                death: max_radius,
                is_essential: true,
            });
        }
    }
}

#[verified_engine::verified]
fn push_h1_intervals(
    edges: &[FilteredEdge],
    triangles: &[FilteredTriangle],
    partial_1: &Z2BoundaryWorkspace,
    partial_2: &Z2BoundaryWorkspace,
    max_radius: f64,
    intervals: &mut Vec<RipsInterval>,
) {
    for (e_idx, edge) in edges.iter().enumerate() {
        let birth = edge.diameter;
        if let Some(t_idx) = partial_2.pivot_col_for_row(e_idx) {
            let death = triangles[t_idx].diameter;
            if death > birth {
                intervals.push(RipsInterval {
                    dimension: 1,
                    birth,
                    death,
                    is_essential: false,
                });
            }
        } else if !partial_1.is_cleared(e_idx) && partial_1.col_pivot(e_idx).is_none() {
            intervals.push(RipsInterval {
                dimension: 1,
                birth,
                death: max_radius.max(birth),
                is_essential: true,
            });
        }
    }
}

// [cite:clinical_trials_statistics]
