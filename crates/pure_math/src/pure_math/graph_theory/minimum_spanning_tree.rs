//! Preallocated zero-allocation Kruskal minimum spanning tree (and forest) solver.

use super::error::GraphTheoryError;

/// Single edge evaluation step during Kruskal's sorted union-find scan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MstEdgeStep {
    /// Index into the original input edge slice.
    pub edge_index: usize,
    /// First endpoint vertex.
    pub u: usize,
    /// Second endpoint vertex.
    pub v: usize,
    /// Edge weight.
    pub weight: f64,
    /// `true` if union-find merged two distinct components; `false` if rejected as a cycle.
    pub accepted: bool,
}

/// Zero-allocation scalar summary returned by [`solve_minimum_spanning_tree_in_place`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MstSummary {
    /// Total weight of all selected spanning-tree/forest edges.
    pub total_weight: f64,
    /// Number of edges accepted into the spanning tree/forest.
    pub selected_edge_count: usize,
    /// Number of connected components remaining (`1` when the graph is connected or empty).
    pub component_count: usize,
}

/// Complete minimum spanning tree/forest result with stepped Kruskal trace.
#[derive(Clone, Debug, PartialEq)]
pub struct MstResult {
    /// Total weight of the minimum spanning tree/forest.
    pub total_weight: f64,
    /// Number of connected components (`1` if connected).
    pub component_count: usize,
    /// Indices into the input edge slice accepted into the minimum spanning tree/forest.
    pub tree_edge_indices: Vec<usize>,
    /// Ordered sequence of Kruskal edge evaluations (sorted by non-decreasing weight).
    pub steps: Vec<MstEdgeStep>,
}

#[derive(Clone, Copy, Debug)]
struct IndexedEdge {
    u: usize,
    v: usize,
    weight: f64,
    original_index: usize,
}

/// Preallocated heapsort and disjoint-set workspace for zero-allocation Kruskal execution.
#[derive(Clone, Debug, Default)]
pub struct KruskalScratch {
    sorted_edges: Vec<IndexedEdge>,
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl KruskalScratch {
    /// Preallocates disjoint-set and edge-sorting buffers for up to `max_nodes` and `max_edges`.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_capacity(max_nodes: usize, max_edges: usize) -> Self {
        Self {
            sorted_edges: Vec::with_capacity(max_edges),
            parent: Vec::with_capacity(max_nodes),
            rank: Vec::with_capacity(max_nodes),
        }
    }

    /// Resizes internal buffers if needed so subsequent [`solve_minimum_spanning_tree_in_place`]
    /// calls run with zero heap allocations.
    #[verified_engine::verified]
    pub fn reserve_for(&mut self, vertex_count: usize, edge_count: usize) {
        if self.sorted_edges.capacity() < edge_count {
            self.sorted_edges
                .reserve(edge_count - self.sorted_edges.capacity());
        }
        if self.parent.len() < vertex_count {
            self.parent.resize(vertex_count, 0);
        }
        if self.rank.len() < vertex_count {
            self.rank.resize(vertex_count, 0);
        }
    }

    #[verified_engine::verified]
    fn init_union_find(&mut self, vertex_count: usize) {
        let mut i = 0usize;
        while i < vertex_count {
            self.parent[i] = i;
            self.rank[i] = 0;
            i += 1;
        }
    }

    #[verified_engine::verified]
    fn find_root(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            let grand = self.parent[self.parent[x]];
            self.parent[x] = grand;
            x = grand;
        }
        x
    }

    #[verified_engine::verified]
    fn union_roots(&mut self, u: usize, v: usize) -> bool {
        let ru = self.find_root(u);
        let rv = self.find_root(v);
        if ru == rv {
            return false;
        }
        if self.rank[ru] < self.rank[rv] {
            self.parent[ru] = rv;
        } else if self.rank[ru] > self.rank[rv] {
            self.parent[rv] = ru;
        } else {
            self.parent[rv] = ru;
            self.rank[ru] = self.rank[ru].saturating_add(1);
        }
        true
    }
}

#[verified_engine::verified]
fn edge_less(a: &IndexedEdge, b: &IndexedEdge) -> bool {
    match a.weight.partial_cmp(&b.weight) {
        Some(std::cmp::Ordering::Less) => true,
        Some(std::cmp::Ordering::Greater) => false,
        _ => a.original_index < b.original_index,
    }
}

#[verified_engine::verified]
fn sift_down_edges(slice: &mut [IndexedEdge], mut root: usize, end: usize) {
    while 2 * root + 1 < end {
        let left = 2 * root + 1;
        let right = left + 1;
        let mut largest = root;
        if edge_less(&slice[largest], &slice[left]) {
            largest = left;
        }
        if right < end && edge_less(&slice[largest], &slice[right]) {
            largest = right;
        }
        if largest == root {
            break;
        }
        slice.swap(root, largest);
        root = largest;
    }
}

#[verified_engine::verified]
fn heapsort_edges_in_place(slice: &mut [IndexedEdge]) {
    let len = slice.len();
    if len <= 1 {
        return;
    }
    let mut start = len / 2;
    while start > 0 {
        start -= 1;
        sift_down_edges(slice, start, len);
    }
    let mut end = len;
    while end > 1 {
        end -= 1;
        slice.swap(0, end);
        sift_down_edges(slice, 0, end);
    }
}

#[verified_engine::verified]
fn populate_and_validate_edges(
    scratch: &mut KruskalScratch,
    vertex_count: usize,
    edges: &[(usize, usize, f64)],
) -> Result<(), GraphTheoryError> {
    scratch.sorted_edges.clear();
    let mut i = 0usize;
    while i < edges.len() {
        let (u, v, w) = edges[i];
        if u >= vertex_count {
            return Err(GraphTheoryError::InvalidTerminal {
                node: u,
                vertex_count,
            });
        }
        if v >= vertex_count {
            return Err(GraphTheoryError::InvalidTerminal {
                node: v,
                vertex_count,
            });
        }
        if !w.is_finite() {
            return Err(GraphTheoryError::InvalidWeight { value: w });
        }
        scratch.sorted_edges.push(IndexedEdge {
            u,
            v,
            weight: w,
            original_index: i,
        });
        i += 1;
    }
    heapsort_edges_in_place(&mut scratch.sorted_edges);
    Ok(())
}

/// Computes a minimum spanning tree (or forest) in-place with zero heap allocations after [`KruskalScratch::reserve_for`].
///
/// `out_selected_mask` must have length `edges.len()` and receives `true` for every accepted tree edge.
///
/// # Examples
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::graph_theory::minimum_spanning_tree::{
///     KruskalScratch, solve_minimum_spanning_tree_in_place,
/// };
///
/// let edges = [(0, 1, 4.0), (0, 2, 1.0), (2, 1, 2.0), (1, 3, 5.0), (2, 3, 3.0)];
/// let mut scratch = KruskalScratch::with_capacity(4, 5);
/// scratch.reserve_for(4, 5);
/// let mut selected = [false; 5];
/// let summary = solve_minimum_spanning_tree_in_place(&mut scratch, 4, &edges, &mut selected)?;
/// assert!((summary.total_weight - 6.0).abs() < 1e-9);
/// assert_eq!(summary.selected_edge_count, 3);
/// # Ok::<(), math_explorer::error::Error>(())
/// ```
#[verified_engine::verified]
pub fn solve_minimum_spanning_tree_in_place(
    scratch: &mut KruskalScratch,
    vertex_count: usize,
    edges: &[(usize, usize, f64)],
    out_selected_mask: &mut [bool],
) -> Result<MstSummary, GraphTheoryError> {
    if out_selected_mask.len() != edges.len() {
        return Err(GraphTheoryError::ScratchBufferMismatch {
            expected: edges.len(),
            actual: out_selected_mask.len(),
        });
    }
    if scratch.parent.len() < vertex_count || scratch.rank.len() < vertex_count {
        return Err(GraphTheoryError::ScratchBufferMismatch {
            expected: vertex_count,
            actual: scratch.parent.len().min(scratch.rank.len()),
        });
    }
    populate_and_validate_edges(scratch, vertex_count, edges)?;
    scratch.init_union_find(vertex_count);
    out_selected_mask.fill(false);

    let mut total_weight = 0.0;
    let mut selected_count = 0usize;
    let mut i = 0usize;
    while i < scratch.sorted_edges.len() {
        let item = scratch.sorted_edges[i];
        if scratch.union_roots(item.u, item.v) {
            out_selected_mask[item.original_index] = true;
            total_weight += item.weight;
            selected_count += 1;
        }
        i += 1;
    }
    let component_count = if vertex_count == 0 {
        0
    } else {
        vertex_count - selected_count
    };
    Ok(MstSummary {
        total_weight,
        selected_edge_count: selected_count,
        component_count,
    })
}

/// Computes a minimum spanning tree (or forest) and returns accepted edge indices plus stepped Kruskal trace.
///
/// # Examples
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::graph_theory::minimum_spanning_tree::minimum_spanning_tree;
///
/// let edges = [(0, 1, 4.0), (0, 2, 1.0), (2, 1, 2.0), (1, 3, 5.0), (2, 3, 3.0)];
/// let mst = minimum_spanning_tree(4, &edges)?;
/// assert!((mst.total_weight - 6.0).abs() < 1e-9);
/// assert_eq!(mst.tree_edge_indices, vec![1, 2, 4]);
/// # Ok::<(), math_explorer::error::Error>(())
/// ```
#[verified_engine::verified]
pub fn minimum_spanning_tree(
    vertex_count: usize,
    edges: &[(usize, usize, f64)],
) -> Result<MstResult, GraphTheoryError> {
    let mut scratch = KruskalScratch::with_capacity(vertex_count, edges.len());
    scratch.reserve_for(vertex_count, edges.len());
    populate_and_validate_edges(&mut scratch, vertex_count, edges)?;
    scratch.init_union_find(vertex_count);

    let mut total_weight = 0.0;
    let mut tree_edge_indices = Vec::new();
    let mut steps = Vec::with_capacity(scratch.sorted_edges.len());
    let mut i = 0usize;
    while i < scratch.sorted_edges.len() {
        let item = scratch.sorted_edges[i];
        let accepted = scratch.union_roots(item.u, item.v);
        if accepted {
            total_weight += item.weight;
            tree_edge_indices.push(item.original_index);
        }
        steps.push(MstEdgeStep {
            edge_index: item.original_index,
            u: item.u,
            v: item.v,
            weight: item.weight,
            accepted,
        });
        i += 1;
    }
    let component_count = if vertex_count == 0 {
        0
    } else {
        vertex_count - tree_edge_indices.len()
    };
    Ok(MstResult {
        total_weight,
        component_count,
        tree_edge_indices,
        steps,
    })
}

// [cite:graph_parameters_rust]
