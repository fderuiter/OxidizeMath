//! Preallocated min-fill upper-bound elimination solver for graphs with `|V| > 24`.

use super::TreewidthScratch;
use crate::pure_math::graph_theory::graph::Graph;
use petgraph::visit::EdgeRef;

#[verified_engine::verified]
pub(super) fn min_fill_large_in_place<N, E>(
    g: &Graph<N, E>,
    n: usize,
    scratch: &mut TreewidthScratch,
) -> usize {
    scratch.prepare(n);
    let wpr = scratch.words_per_row;
    let mut has_edge = false;
    for edge in g.graph.edge_references() {
        let u = edge.source().index();
        let v = edge.target().index();
        if u != v && u < n && v < n {
            scratch.adj_words[u * wpr + (v / 64)] |= 1u64 << (v % 64);
            scratch.adj_words[v * wpr + (u / 64)] |= 1u64 << (u % 64);
            has_edge = true;
        }
    }
    if !has_edge {
        return 0;
    }
    let mut width = 0usize;
    let mut step = 0usize;
    while step < n {
        let best_v = select_min_fill_large(scratch, n);
        let deg = eliminate_large_vertex(scratch, n, best_v);
        width = width.max(deg);
        step += 1;
    }
    width
}

#[verified_engine::verified]
fn select_min_fill_large(scratch: &TreewidthScratch, n: usize) -> usize {
    let wpr = scratch.words_per_row;
    let mut best_v = 0usize;
    let mut best_key = (usize::MAX, usize::MAX);
    let mut v = 0usize;
    while v < n {
        if scratch.active[v] {
            let deg = row_degree(&scratch.adj_words, wpr, v);
            let fill = row_fill_edges(&scratch.adj_words, wpr, n, v);
            let key = (fill, deg);
            if key < best_key {
                best_key = key;
                best_v = v;
            }
        }
        v += 1;
    }
    best_v
}

#[verified_engine::verified]
fn row_degree(adj_words: &[u64], wpr: usize, v: usize) -> usize {
    let base = v * wpr;
    let mut deg = 0usize;
    let mut w = 0usize;
    while w < wpr {
        deg += adj_words[base + w].count_ones() as usize;
        w += 1;
    }
    deg
}

#[verified_engine::verified]
fn row_fill_edges(adj_words: &[u64], wpr: usize, n: usize, v: usize) -> usize {
    let v_base = v * wpr;
    let mut fill = 0usize;
    let mut u = 0usize;
    while u < n {
        let uw = u / 64;
        if (adj_words[v_base + uw] & (1u64 << (u % 64))) != 0 {
            let u_base = u * wpr;
            let mut w = uw;
            while w < wpr {
                let mask = if w == uw {
                    let shift = (u % 64) + 1;
                    if shift >= 64 { 0 } else { !0u64 << shift }
                } else {
                    !0u64
                };
                let higher_nbrs = adj_words[v_base + w] & mask;
                fill += (higher_nbrs & !adj_words[u_base + w]).count_ones() as usize;
                w += 1;
            }
        }
        u += 1;
    }
    fill
}

#[verified_engine::verified]
fn eliminate_large_vertex(scratch: &mut TreewidthScratch, n: usize, v: usize) -> usize {
    let wpr = scratch.words_per_row;
    let v_base = v * wpr;
    let deg = row_degree(&scratch.adj_words, wpr, v);
    scratch.active[v] = false;
    let vw = v / 64;
    let v_bit = 1u64 << (v % 64);
    let mut u = 0usize;
    while u < n {
        if (scratch.adj_words[v_base + (u / 64)] & (1u64 << (u % 64))) != 0 {
            let u_base = u * wpr;
            scratch.adj_words[u_base + vw] &= !v_bit;
            let mut w = 0usize;
            while w < wpr {
                let nbr_word = scratch.adj_words[v_base + w];
                scratch.adj_words[u_base + w] |= nbr_word;
                w += 1;
            }
            scratch.adj_words[u_base + (u / 64)] &= !(1u64 << (u % 64));
        }
        u += 1;
    }
    scratch.adj_words[v_base..(v_base + wpr)].fill(0);
    deg
}

// [cite:graph_parameters_rust]
