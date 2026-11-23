//! Exact bitmask branch-and-bound (`|V| <= 24`) and min-fill (`|V| > 24`) treewidth solvers.

use crate::pure_math::graph_theory::graph::Graph;
use petgraph::visit::EdgeRef;

/// Maximum vertex count solved via exact bitmask branch-and-bound elimination ordering.
pub const MAX_EXACT_TREEWIDTH_VERTICES: usize = 24;

const MAX_BNB_STEPS: usize = 1_048_576;
const TRANSPOSITION_SLOTS: usize = 4096;

#[derive(Clone, Copy)]
struct SearchFrame {
    adj: [u32; MAX_EXACT_TREEWIDTH_VERTICES],
    active: u32,
    cur_max: u8,
    candidates: [u8; MAX_EXACT_TREEWIDTH_VERTICES],
    cand_len: u8,
    cand_idx: u8,
}

impl SearchFrame {
    #[verified_engine::verified]
    const fn empty() -> Self {
        Self {
            adj: [0; MAX_EXACT_TREEWIDTH_VERTICES],
            active: 0,
            cur_max: 0,
            candidates: [0; MAX_EXACT_TREEWIDTH_VERTICES],
            cand_len: 0,
            cand_idx: 0,
        }
    }
}

/// Preallocated bitset workspace for zero-allocation treewidth evaluation across arbitrary `|V|`.
#[derive(Clone, Debug, Default)]
pub struct TreewidthScratch {
    pub(self) words_per_row: usize,
    pub(self) adj_words: Vec<u64>,
    pub(self) active: Vec<bool>,
}

impl TreewidthScratch {
    /// Preallocates bitset buffers sufficient to evaluate graphs up to `max_vertices` with zero reallocations.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_capacity(max_vertices: usize) -> Self {
        let words = max_vertices.div_ceil(64);
        Self {
            words_per_row: words,
            adj_words: vec![0; max_vertices.saturating_mul(words)],
            active: vec![false; max_vertices],
        }
    }

    #[verified_engine::verified]
    fn prepare(&mut self, n: usize) {
        let words = n.div_ceil(64);
        self.words_per_row = words;
        let total = n.saturating_mul(words);
        if self.adj_words.len() < total {
            self.adj_words.resize(total, 0);
        }
        self.adj_words[..total].fill(0);
        if self.active.len() < n {
            self.active.resize(n, false);
        }
        self.active[..n].fill(true);
    }
}

/// Computes the treewidth of `g` (exact for `|V| <= 24`, min-fill upper bound for `|V| > 24`).
///
/// # Examples
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::graph_theory::graph::Graph;
/// use math_explorer::pure_math::graph_theory::parameters::treewidth::treewidth;
///
/// let mut c5: Graph<(), ()> = Graph::new();
/// let v: Vec<_> = (0..5).map(|_| c5.add_node(())).collect();
/// for i in 0..5 {
///     c5.add_edge(v[i], v[(i + 1) % 5], ());
/// }
/// assert_eq!(treewidth(&c5), 2);
/// ```
#[must_use]
#[verified_engine::verified]
pub fn treewidth<N, E>(g: &Graph<N, E>) -> usize {
    let n = g.graph.node_count();
    if n <= 1 {
        return 0;
    }
    if n <= MAX_EXACT_TREEWIDTH_VERTICES {
        return exact_treewidth_bitmask(g, n);
    }
    let mut scratch = TreewidthScratch::with_capacity(n);
    min_fill_large_in_place(g, n, &mut scratch)
}

/// Computes treewidth using preallocated `scratch` buffers with zero post-init heap allocations.
///
/// # Examples
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::graph_theory::graph::Graph;
/// use math_explorer::pure_math::graph_theory::parameters::treewidth::{
///     TreewidthScratch, treewidth_in_place,
/// };
///
/// let mut k4: Graph<(), ()> = Graph::new();
/// let v: Vec<_> = (0..4).map(|_| k4.add_node(())).collect();
/// for i in 0..4 {
///     for j in (i + 1)..4 {
///         k4.add_edge(v[i], v[j], ());
///     }
/// }
/// let mut scratch = TreewidthScratch::with_capacity(8);
/// assert_eq!(treewidth_in_place(&k4, &mut scratch), 3);
/// ```
#[must_use]
#[verified_engine::verified]
pub fn treewidth_in_place<N, E>(g: &Graph<N, E>, scratch: &mut TreewidthScratch) -> usize {
    let n = g.graph.node_count();
    if n <= 1 {
        return 0;
    }
    if n <= MAX_EXACT_TREEWIDTH_VERTICES {
        return exact_treewidth_bitmask(g, n);
    }
    min_fill_large_in_place(g, n, scratch)
}

#[verified_engine::verified]
fn build_bitmask_adj<N, E>(
    g: &Graph<N, E>,
    n: usize,
) -> ([u32; MAX_EXACT_TREEWIDTH_VERTICES], bool) {
    let mut adj = [0u32; MAX_EXACT_TREEWIDTH_VERTICES];
    let mut has_edge = false;
    for edge in g.graph.edge_references() {
        let u = edge.source().index();
        let v = edge.target().index();
        if u != v && u < n && v < n {
            adj[u] |= 1u32 << v;
            adj[v] |= 1u32 << u;
            has_edge = true;
        }
    }
    (adj, has_edge)
}

#[verified_engine::verified]
fn eliminate_bitmask_vertex(
    adj: &mut [u32; MAX_EXACT_TREEWIDTH_VERTICES],
    active: &mut u32,
    v: usize,
) -> usize {
    let v_bit = 1u32 << v;
    let nbrs = adj[v] & *active;
    let deg = nbrs.count_ones() as usize;
    *active &= !v_bit;
    adj[v] = 0;
    let mut rem = nbrs;
    while rem != 0 {
        let u = rem.trailing_zeros() as usize;
        let u_bit = 1u32 << u;
        rem &= !u_bit;
        adj[u] = (adj[u] & *active) | (nbrs & !u_bit);
    }
    deg
}

#[verified_engine::verified]
fn missing_clique_edges_capped(
    adj: &[u32; MAX_EXACT_TREEWIDTH_VERTICES],
    nbrs: u32,
    cap: usize,
) -> usize {
    let mut missing = 0usize;
    let mut rem = nbrs;
    while rem != 0 {
        let u = rem.trailing_zeros() as usize;
        rem &= rem - 1;
        missing += (rem & !adj[u]).count_ones() as usize;
        if missing > cap {
            return missing;
        }
    }
    missing
}

#[verified_engine::verified]
fn mmd_lower_bound(adj: &[u32; MAX_EXACT_TREEWIDTH_VERTICES], active: u32) -> usize {
    let mut sub = active;
    let mut best = 0usize;
    while sub != 0 {
        let mut min_v = 0usize;
        let mut min_deg = usize::MAX;
        let mut iter = sub;
        while iter != 0 {
            let v = iter.trailing_zeros() as usize;
            iter &= iter - 1;
            let d = (adj[v] & sub).count_ones() as usize;
            if d < min_deg {
                min_deg = d;
                min_v = v;
            }
        }
        best = best.max(min_deg);
        sub &= !(1u32 << min_v);
    }
    best
}

#[verified_engine::verified]
fn min_fill_upper_bound(mut adj: [u32; MAX_EXACT_TREEWIDTH_VERTICES], mut active: u32) -> usize {
    let mut width = 0usize;
    while active != 0 {
        let mut best_v = active.trailing_zeros() as usize;
        let mut best_key = (usize::MAX, usize::MAX);
        let mut iter = active;
        while iter != 0 {
            let v = iter.trailing_zeros() as usize;
            iter &= iter - 1;
            let nbrs = adj[v] & active;
            let fill = missing_clique_edges_capped(&adj, nbrs, usize::MAX);
            let key = (fill, nbrs.count_ones() as usize);
            if key < best_key {
                best_key = key;
                best_v = v;
            }
        }
        let d = eliminate_bitmask_vertex(&mut adj, &mut active, best_v);
        width = width.max(d);
    }
    width
}

#[verified_engine::verified]
fn apply_safe_reductions(
    adj: &mut [u32; MAX_EXACT_TREEWIDTH_VERTICES],
    active: &mut u32,
    known_lb: usize,
    cur_max: &mut usize,
) {
    let mut progress = true;
    while progress && *active != 0 {
        progress = false;
        let mut iter = *active;
        while iter != 0 {
            let v = iter.trailing_zeros() as usize;
            iter &= iter - 1;
            let nbrs = adj[v] & *active;
            let deg = nbrs.count_ones() as usize;
            let missing = missing_clique_edges_capped(adj, nbrs, 1);
            let threshold = known_lb.max(*cur_max);
            if missing == 0 || (missing == 1 && deg <= threshold) {
                let d = eliminate_bitmask_vertex(adj, active, v);
                *cur_max = (*cur_max).max(d);
                progress = true;
                break;
            }
        }
    }
}

#[verified_engine::verified]
fn populate_ordered_candidates(
    adj: &[u32; MAX_EXACT_TREEWIDTH_VERTICES],
    active: u32,
    best_ub: usize,
    out: &mut [u8; MAX_EXACT_TREEWIDTH_VERTICES],
) -> u8 {
    let mut keys = [(u16::MAX, 0u8); MAX_EXACT_TREEWIDTH_VERTICES];
    let mut len = 0usize;
    let mut iter = active;
    while iter != 0 {
        let v = iter.trailing_zeros() as usize;
        iter &= iter - 1;
        let nbrs = adj[v] & active;
        let deg = nbrs.count_ones() as usize;
        if deg < best_ub {
            let fill = missing_clique_edges_capped(adj, nbrs, usize::MAX);
            let score = ((fill as u16) << 5) | (deg as u16);
            let mut pos = len;
            while pos > 0 && score < keys[pos - 1].0 {
                keys[pos] = keys[pos - 1];
                pos -= 1;
            }
            keys[pos] = (score, v as u8);
            len += 1;
        }
    }
    let mut i = 0usize;
    while i < len {
        out[i] = keys[i].1;
        i += 1;
    }
    len as u8
}

#[verified_engine::verified]
fn check_or_record_transposition(
    seen_masks: &mut [u32; TRANSPOSITION_SLOTS],
    seen_costs: &mut [u8; TRANSPOSITION_SLOTS],
    active: u32,
    cur_max: u8,
) -> bool {
    let slot = ((active.wrapping_mul(0x9E37_79B9)) >> 20) as usize & (TRANSPOSITION_SLOTS - 1);
    if seen_masks[slot] == active && seen_costs[slot] <= cur_max {
        return false;
    }
    seen_masks[slot] = active;
    seen_costs[slot] = cur_max;
    true
}

#[verified_engine::verified]
fn exact_treewidth_bitmask<N, E>(g: &Graph<N, E>, n: usize) -> usize {
    let (mut adj, has_edge) = build_bitmask_adj(g, n);
    if !has_edge {
        return 0;
    }
    let mut active = (1u32 << n) - 1;
    let mut root_max = 0usize;
    let init_lb = mmd_lower_bound(&adj, active);
    apply_safe_reductions(&mut adj, &mut active, init_lb, &mut root_max);
    if active == 0 {
        return root_max;
    }
    let root_lb = root_max.max(mmd_lower_bound(&adj, active));
    let mut best_ub = root_max.max(min_fill_upper_bound(adj, active));
    if root_lb >= best_ub {
        return best_ub;
    }
    run_exact_bnb(adj, active, root_max, root_lb, &mut best_ub);
    best_ub
}

#[verified_engine::verified]
fn run_exact_bnb(
    root_adj: [u32; MAX_EXACT_TREEWIDTH_VERTICES],
    root_active: u32,
    root_max: usize,
    root_lb: usize,
    best_ub: &mut usize,
) {
    let mut stack = [SearchFrame::empty(); MAX_EXACT_TREEWIDTH_VERTICES + 1];
    let mut seen_masks = [0u32; TRANSPOSITION_SLOTS];
    let mut seen_costs = [u8::MAX; TRANSPOSITION_SLOTS];
    stack[0].adj = root_adj;
    stack[0].active = root_active;
    stack[0].cur_max = root_max as u8;
    stack[0].cand_len =
        populate_ordered_candidates(&root_adj, root_active, *best_ub, &mut stack[0].candidates);
    let mut depth = 1usize;
    let mut steps = 0usize;

    while depth > 0 && steps < MAX_BNB_STEPS && *best_ub > root_lb {
        steps += 1;
        let top = depth - 1;
        if stack[top].cand_idx >= stack[top].cand_len {
            depth -= 1;
            continue;
        }
        let v = stack[top].candidates[stack[top].cand_idx as usize] as usize;
        stack[top].cand_idx += 1;
        if let Some(child) = expand_bnb_child(
            &stack[top],
            v,
            root_lb,
            best_ub,
            &mut seen_masks,
            &mut seen_costs,
        ) {
            stack[depth] = child;
            depth += 1;
        }
    }
}

#[verified_engine::verified]
fn expand_bnb_child(
    parent: &SearchFrame,
    v: usize,
    root_lb: usize,
    best_ub: &mut usize,
    seen_masks: &mut [u32; TRANSPOSITION_SLOTS],
    seen_costs: &mut [u8; TRANSPOSITION_SLOTS],
) -> Option<SearchFrame> {
    let mut adj = parent.adj;
    let mut active = parent.active;
    let d = eliminate_bitmask_vertex(&mut adj, &mut active, v);
    let mut cur_max = (parent.cur_max as usize).max(d);
    if cur_max >= *best_ub {
        return None;
    }
    apply_safe_reductions(&mut adj, &mut active, root_lb, &mut cur_max);
    if cur_max >= *best_ub {
        return None;
    }
    if active == 0 {
        *best_ub = cur_max;
        return None;
    }
    let lb = cur_max.max(mmd_lower_bound(&adj, active));
    if lb >= *best_ub
        || !check_or_record_transposition(seen_masks, seen_costs, active, cur_max as u8)
    {
        return None;
    }
    let mut child = SearchFrame::empty();
    child.adj = adj;
    child.active = active;
    child.cur_max = cur_max as u8;
    child.cand_len = populate_ordered_candidates(&adj, active, *best_ub, &mut child.candidates);
    Some(child)
}

#[path = "treewidth_large.rs"]
mod treewidth_large;
use treewidth_large::min_fill_large_in_place;

// [cite:graph_parameters_rust]
