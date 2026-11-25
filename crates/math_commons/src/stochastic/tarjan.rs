//! Iterative zero-allocation Tarjan strongly connected component (SCC) decomposition.
// [cite:math_commons]

use math_memory::{ArenaScope, FlagSlice};

use super::SccSummary;
use crate::math_kernel::types::flatten_2d_index;

const UNVISITED: usize = usize::MAX;
const EDGE_REL_TOL: f64 = 1e-14;

struct TarjanBuffers<'a> {
    tarjan_index: &'a mut [usize],
    tarjan_lowlink: &'a mut [usize],
    tarjan_on_stack: FlagSlice<'a>,
    tarjan_stack: &'a mut [usize],
    call_node: &'a mut [usize],
    call_next_edge: &'a mut [usize],
    component_id: &'a mut [usize],
    component_is_sink: FlagSlice<'a>,
    row_scales: &'a mut [f64],
}

/// Computes per-row off-diagonal maximum positive magnitude for scale-invariant edge detection.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub(super) fn compute_off_diagonal_scales(matrix: &[f64], n: usize, row_scales: &mut [f64]) {
    debug_assert!(matrix.len() >= n * n);
    debug_assert!(row_scales.len() >= n);
    for (i, row_scale) in row_scales.iter_mut().enumerate().take(n) {
        let row_offset = i * n;
        let mut max_pos = 0.0_f64;
        for j in 0..n {
            if i != j {
                let v = matrix[row_offset + j];
                if v > max_pos {
                    max_pos = v;
                }
            }
        }
        *row_scale = max_pos;
    }
}

#[inline]
fn has_directed_edge(matrix: &[f64], row_scales: &[f64], n: usize, u: usize, v: usize) -> bool {
    let scale = row_scales[u];
    u != v && scale > 0.0 && matrix[flatten_2d_index(v, u, n)] > scale * EDGE_REL_TOL
}

/// Runs iterative Tarjan SCC traversal on `matrix` ($n \times n$) carving transient buffers from
/// `scratch` and writing unique sink members into `sink_members`.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub(super) fn decompose_tarjan(
    scratch: &mut ArenaScope<'_>,
    sink_members: &mut [usize],
    matrix: &[f64],
    n: usize,
) -> Option<SccSummary> {
    debug_assert!(n > 0 && matrix.len() >= n * n && sink_members.len() >= n);
    let mut sub = scratch.reborrow();
    let mut bufs = TarjanBuffers {
        tarjan_index: sub.alloc_slice::<usize>(n).ok()?,
        tarjan_lowlink: sub.alloc_slice::<usize>(n).ok()?,
        tarjan_on_stack: sub.alloc_flags(n).ok()?,
        tarjan_stack: sub.alloc_slice::<usize>(n).ok()?,
        call_node: sub.alloc_slice::<usize>(n).ok()?,
        call_next_edge: sub.alloc_slice::<usize>(n).ok()?,
        component_id: sub.alloc_slice::<usize>(n).ok()?,
        component_is_sink: sub.alloc_flags(n).ok()?,
        row_scales: sub.alloc_slice::<f64>(n).ok()?,
    };

    compute_off_diagonal_scales(matrix, n, bufs.row_scales);
    bufs.tarjan_index.fill(UNVISITED);
    bufs.component_id.fill(UNVISITED);

    let mut next_index = 0usize;
    let mut num_components = 0usize;
    let mut tarjan_sp = 0usize;

    for start in 0..n {
        if bufs.tarjan_index[start] != UNVISITED {
            continue;
        }
        tarjan_visit_root(
            &mut bufs,
            matrix,
            n,
            start,
            &mut next_index,
            &mut num_components,
            &mut tarjan_sp,
        );
    }

    Some(classify_closed_sinks(
        &mut bufs,
        sink_members,
        matrix,
        n,
        num_components,
    ))
}

#[cfg_attr(feature = "std", verified_engine::verified)]
fn tarjan_visit_root(
    bufs: &mut TarjanBuffers<'_>,
    matrix: &[f64],
    n: usize,
    start: usize,
    next_index: &mut usize,
    num_components: &mut usize,
    tarjan_sp: &mut usize,
) {
    debug_assert!(start < n);
    push_tarjan_node(bufs, start, 0, next_index, tarjan_sp);
    let mut call_sp = 1usize;

    while call_sp > 0 {
        let top = call_sp - 1;
        let u = bufs.call_node[top];
        let v = bufs.call_next_edge[top];

        if v < n {
            bufs.call_next_edge[top] = v + 1;
            if !has_directed_edge(matrix, bufs.row_scales, n, u, v) {
                continue;
            }
            if bufs.tarjan_index[v] == UNVISITED {
                push_tarjan_node(bufs, v, call_sp, next_index, tarjan_sp);
                call_sp += 1;
            } else if bufs.tarjan_on_stack.get(v) && bufs.tarjan_index[v] < bufs.tarjan_lowlink[u] {
                bufs.tarjan_lowlink[u] = bufs.tarjan_index[v];
            }
        } else {
            call_sp -= 1;
            finish_tarjan_node(bufs, u, call_sp, num_components, tarjan_sp);
        }
    }
}

#[inline]
fn push_tarjan_node(
    bufs: &mut TarjanBuffers<'_>,
    node: usize,
    call_sp: usize,
    next_index: &mut usize,
    tarjan_sp: &mut usize,
) {
    bufs.tarjan_index[node] = *next_index;
    bufs.tarjan_lowlink[node] = *next_index;
    *next_index += 1;
    bufs.tarjan_stack[*tarjan_sp] = node;
    *tarjan_sp += 1;
    bufs.tarjan_on_stack.set(node, true);
    bufs.call_node[call_sp] = node;
    bufs.call_next_edge[call_sp] = 0;
}

#[inline]
fn finish_tarjan_node(
    bufs: &mut TarjanBuffers<'_>,
    u: usize,
    call_sp: usize,
    num_components: &mut usize,
    tarjan_sp: &mut usize,
) {
    if call_sp > 0 {
        let parent = bufs.call_node[call_sp - 1];
        if bufs.tarjan_lowlink[u] < bufs.tarjan_lowlink[parent] {
            bufs.tarjan_lowlink[parent] = bufs.tarjan_lowlink[u];
        }
    }
    if bufs.tarjan_lowlink[u] == bufs.tarjan_index[u] {
        let cid = *num_components;
        *num_components += 1;
        loop {
            *tarjan_sp -= 1;
            let w = bufs.tarjan_stack[*tarjan_sp];
            bufs.tarjan_on_stack.set(w, false);
            bufs.component_id[w] = cid;
            if w == u {
                break;
            }
        }
    }
}

#[cfg_attr(feature = "std", verified_engine::verified)]
fn classify_closed_sinks(
    bufs: &mut TarjanBuffers<'_>,
    sink_members: &mut [usize],
    matrix: &[f64],
    n: usize,
    num_components: usize,
) -> SccSummary {
    debug_assert!(num_components >= 1 && num_components <= n);
    for c in 0..num_components {
        bufs.component_is_sink.set(c, true);
    }

    for u in 0..n {
        let cu = bufs.component_id[u];
        if !bufs.component_is_sink.get(cu) {
            continue;
        }
        for v in 0..n {
            if bufs.component_id[v] != cu && has_directed_edge(matrix, bufs.row_scales, n, u, v) {
                bufs.component_is_sink.set(cu, false);
                break;
            }
        }
    }

    let mut num_sink_components = 0usize;
    let mut unique_sink_cid = 0usize;
    for cid in 0..num_components {
        if bufs.component_is_sink.get(cid) {
            num_sink_components += 1;
            unique_sink_cid = cid;
        }
    }

    let mut unique_sink_size = 0usize;
    if num_sink_components == 1 {
        for (state, &cid) in bufs.component_id[..n].iter().enumerate() {
            if cid == unique_sink_cid {
                sink_members[unique_sink_size] = state;
                unique_sink_size += 1;
            }
        }
    }

    SccSummary {
        num_components,
        num_sink_components,
        unique_sink_size,
    }
}
