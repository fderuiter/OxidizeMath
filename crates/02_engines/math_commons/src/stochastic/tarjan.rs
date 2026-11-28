//! Iterative zero-allocation Tarjan strongly connected component (SCC) decomposition.
// [cite:math_commons]

use super::SccSummary;
use super::views::StochasticViews;
use crate::math_kernel::types::flatten_2d_index;

const UNVISITED: usize = usize::MAX;
const EDGE_REL_TOL: f64 = 1e-14;

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

/// Runs iterative Tarjan SCC traversal on `matrix` ($n \times n$) using `ws` views and
/// identifies closed sink components.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub(super) fn decompose_tarjan(
    ws: &mut StochasticViews<'_>,
    matrix: &[f64],
    n: usize,
) -> SccSummary {
    debug_assert!(n > 0 && matrix.len() >= n * n);
    compute_off_diagonal_scales(matrix, n, &mut ws.row_scales[..n]);
    ws.tarjan_index[..n].fill(UNVISITED);
    ws.tarjan_lowlink[..n].fill(0);
    ws.tarjan_on_stack.fill(false);
    ws.component_id[..n].fill(UNVISITED);

    let mut next_index = 0usize;
    let mut num_components = 0usize;
    let mut tarjan_sp = 0usize;

    for start in 0..n {
        if ws.tarjan_index[start] != UNVISITED {
            continue;
        }
        tarjan_visit_root(
            ws,
            matrix,
            n,
            start,
            &mut next_index,
            &mut num_components,
            &mut tarjan_sp,
        );
    }

    classify_closed_sinks(ws, matrix, n, num_components)
}

#[cfg_attr(feature = "std", verified_engine::verified)]
fn tarjan_visit_root(
    ws: &mut StochasticViews<'_>,
    matrix: &[f64],
    n: usize,
    start: usize,
    next_index: &mut usize,
    num_components: &mut usize,
    tarjan_sp: &mut usize,
) {
    debug_assert!(start < n);
    push_tarjan_node(ws, start, 0, next_index, tarjan_sp);
    let mut call_sp = 1usize;

    while call_sp > 0 {
        let top = call_sp - 1;
        let u = ws.call_node[top];
        let v = ws.call_next_edge[top];

        if v < n {
            ws.call_next_edge[top] = v + 1;
            if !has_directed_edge(matrix, &ws.row_scales[..n], n, u, v) {
                continue;
            }
            if ws.tarjan_index[v] == UNVISITED {
                push_tarjan_node(ws, v, call_sp, next_index, tarjan_sp);
                call_sp += 1;
            } else if ws.tarjan_on_stack.get(v) && ws.tarjan_index[v] < ws.tarjan_lowlink[u] {
                ws.tarjan_lowlink[u] = ws.tarjan_index[v];
            }
        } else {
            call_sp -= 1;
            finish_tarjan_node(ws, u, call_sp, num_components, tarjan_sp);
        }
    }
}

#[inline]
fn push_tarjan_node(
    ws: &mut StochasticViews<'_>,
    node: usize,
    call_sp: usize,
    next_index: &mut usize,
    tarjan_sp: &mut usize,
) {
    ws.tarjan_index[node] = *next_index;
    ws.tarjan_lowlink[node] = *next_index;
    *next_index += 1;
    ws.tarjan_stack[*tarjan_sp] = node;
    *tarjan_sp += 1;
    ws.tarjan_on_stack.set(node, true);
    ws.call_node[call_sp] = node;
    ws.call_next_edge[call_sp] = 0;
}

#[inline]
fn finish_tarjan_node(
    ws: &mut StochasticViews<'_>,
    u: usize,
    call_sp: usize,
    num_components: &mut usize,
    tarjan_sp: &mut usize,
) {
    if call_sp > 0 {
        let parent = ws.call_node[call_sp - 1];
        if ws.tarjan_lowlink[u] < ws.tarjan_lowlink[parent] {
            ws.tarjan_lowlink[parent] = ws.tarjan_lowlink[u];
        }
    }
    if ws.tarjan_lowlink[u] == ws.tarjan_index[u] {
        let cid = *num_components;
        *num_components += 1;
        loop {
            *tarjan_sp -= 1;
            let w = ws.tarjan_stack[*tarjan_sp];
            ws.tarjan_on_stack.set(w, false);
            ws.component_id[w] = cid;
            if w == u {
                break;
            }
        }
    }
}

#[cfg_attr(feature = "std", verified_engine::verified)]
fn classify_closed_sinks(
    ws: &mut StochasticViews<'_>,
    matrix: &[f64],
    n: usize,
    num_components: usize,
) -> SccSummary {
    debug_assert!(num_components >= 1 && num_components <= n);
    ws.component_is_sink.fill(true);

    for u in 0..n {
        let cu = ws.component_id[u];
        if !ws.component_is_sink.get(cu) {
            continue;
        }
        for v in 0..n {
            if ws.component_id[v] != cu && has_directed_edge(matrix, &ws.row_scales[..n], n, u, v) {
                ws.component_is_sink.set(cu, false);
                break;
            }
        }
    }

    let mut num_sink_components = 0usize;
    let mut unique_sink_cid = 0usize;
    for cid in 0..num_components {
        if ws.component_is_sink.get(cid) {
            num_sink_components += 1;
            unique_sink_cid = cid;
        }
    }

    let mut unique_sink_size = 0usize;
    if num_sink_components == 1 {
        for (state, &cid) in ws.component_id[..n].iter().enumerate() {
            if cid == unique_sink_cid {
                ws.sink_members[unique_sink_size] = state;
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
