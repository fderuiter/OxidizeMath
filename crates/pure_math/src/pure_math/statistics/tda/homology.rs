//! Homology computations and Betti numbers.

use super::complex::SimplicialComplex;
use super::workspace::lowest_set_bit;
use crate::error::TdaError;
use std::collections::{HashMap, HashSet};

/// Union-Find data structure for connected component tracking.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    #[verified_engine::verified]
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    #[verified_engine::verified]
    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut curr = x;
        while curr != root {
            let next = self.parent[curr];
            self.parent[curr] = root;
            curr = next;
        }
        root
    }

    #[verified_engine::verified]
    fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x != root_y {
            if self.rank[root_x] < self.rank[root_y] {
                self.parent[root_x] = root_y;
            } else if self.rank[root_x] > self.rank[root_y] {
                self.parent[root_y] = root_x;
            } else {
                self.parent[root_y] = root_x;
                self.rank[root_x] += 1;
            }
        }
    }

    #[verified_engine::verified]
    fn count_components(&mut self) -> usize {
        let n = self.parent.len();
        let mut roots = HashSet::new();
        for i in 0..n {
            roots.insert(self.find(i));
        }
        roots.len()
    }
}

/// Computes the 0-th Betti number (number of connected components).
///
/// `β₀` counts the number of connected components in the simplicial complex.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::tda::{
///     PointCloud, Point2D, vietoris_rips_complex, betti_number_0
/// };
///
/// let points = vec![
///     Point2D::new(0.0, 0.0),
///     Point2D::new(1.0, 0.0),
///     Point2D::new(10.0, 10.0),
///     Point2D::new(11.0, 10.0),
/// ];
/// let cloud = PointCloud::new(points).unwrap();
/// let complex = vietoris_rips_complex(&cloud, 1.5).unwrap();
///
/// let beta0 = betti_number_0(&complex).unwrap();
/// assert_eq!(beta0, 2);
/// ```
#[verified_engine::verified]
pub fn betti_number_0(complex: &SimplicialComplex) -> Result<usize, TdaError> {
    let vertices = complex.get_simplices(0);
    if vertices.is_empty() {
        return Ok(0);
    }

    let n = vertices.len();
    let mut vertex_to_index = HashMap::with_capacity(n);
    for (i, vertex) in vertices.iter().enumerate() {
        vertex_to_index.insert(vertex.vertices[0], i);
    }

    let mut uf = UnionFind::new(n);
    for edge in complex.get_simplices(1) {
        let i = vertex_to_index[&edge.vertices[0]];
        let j = vertex_to_index[&edge.vertices[1]];
        uf.union(i, j);
    }

    Ok(uf.count_components())
}

/// Computes the exact 1st Betti number `β₁ = dim H₁(K; ℤ₂)` (independent 1-cycles).
///
/// Uses `β₁ = dim(ker ∂₁) - rank_ℤ₂(∂₂)`, where `dim(ker ∂₁) = |E| - |V| + β₀` and
/// `rank_ℤ₂(∂₂)` is computed by exact `ℤ₂` Gaussian column reduction on the 2-boundary
/// matrix `∂₂`. Dependent triangles inside dense cliques (such as the 4th triangle of a
/// `K₄` tetrahedron boundary) reduce to zero mod 2 and never cancel disjoint 1-cycles.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::tda::{
///     PointCloud, Point2D, vietoris_rips_complex, betti_number_1
/// };
///
/// let points = vec![
///     Point2D::new(0.0, 0.0),
///     Point2D::new(1.0, 0.0),
///     Point2D::new(1.0, 1.0),
///     Point2D::new(0.0, 1.0),
/// ];
/// let cloud = PointCloud::new(points).unwrap();
/// let complex = vietoris_rips_complex(&cloud, 1.1).unwrap();
/// assert_eq!(betti_number_1(&complex).unwrap(), 1);
/// ```
#[verified_engine::verified]
pub fn betti_number_1(complex: &SimplicialComplex) -> Result<usize, TdaError> {
    let v = complex.count_simplices(0);
    let e = complex.count_simplices(1);
    let t = complex.count_simplices(2);

    if v == 0 || e == 0 {
        return Ok(0);
    }

    let beta0 = betti_number_0(complex)?;
    let cycles = (e + beta0).saturating_sub(v);
    if cycles == 0 {
        return Ok(0);
    }
    if t == 0 {
        return Ok(cycles);
    }

    let rank_d2 = rank_boundary_2_z2(complex, cycles);
    Ok(cycles.saturating_sub(rank_d2))
}

#[verified_engine::verified]
fn collect_sorted_edges(complex: &SimplicialComplex) -> Vec<[usize; 2]> {
    let mut edges: Vec<[usize; 2]> = complex
        .simplices
        .get(1)
        .map(|set| {
            set.iter()
                .filter(|s| s.vertices.len() == 2)
                .map(|s| [s.vertices[0], s.vertices[1]])
                .collect()
        })
        .unwrap_or_default();
    edges.sort_unstable();
    edges
}

#[verified_engine::verified]
fn collect_sorted_triangles(complex: &SimplicialComplex) -> Vec<[usize; 3]> {
    let mut triangles: Vec<[usize; 3]> = complex
        .simplices
        .get(2)
        .map(|set| {
            set.iter()
                .filter(|s| s.vertices.len() == 3)
                .map(|s| [s.vertices[0], s.vertices[1], s.vertices[2]])
                .collect()
        })
        .unwrap_or_default();
    triangles.sort_unstable();
    triangles
}

#[verified_engine::verified]
fn reduce_column_z2(
    rows: [usize; 3],
    words: usize,
    col: &mut [u64],
    pivot_for_row: &mut [usize],
    basis: &mut Vec<Vec<u64>>,
) {
    col.fill(0);
    for r in rows {
        col[r / 64] ^= 1u64 << (r % 64);
    }
    while let Some(pivot) = lowest_set_bit(col) {
        let existing = pivot_for_row[pivot];
        if existing == usize::MAX {
            pivot_for_row[pivot] = basis.len();
            let _ = words;
            basis.push(col.to_vec());
            return;
        }
        for (slot, &bword) in col.iter_mut().zip(basis[existing].iter()) {
            *slot ^= bword;
        }
    }
}

#[verified_engine::verified]
fn rank_boundary_2_z2(complex: &SimplicialComplex, max_rank: usize) -> usize {
    let edges = collect_sorted_edges(complex);
    if edges.len() < 3 {
        return 0;
    }
    let triangles = collect_sorted_triangles(complex);
    let words = edges.len().div_ceil(64).max(1);
    let mut pivot_for_row = vec![usize::MAX; edges.len()];
    let mut basis: Vec<Vec<u64>> = Vec::with_capacity(triangles.len().min(max_rank));
    let mut col = vec![0u64; words];

    for [a, b, c] in triangles {
        if basis.len() >= max_rank {
            break;
        }
        let (Ok(e0), Ok(e1), Ok(e2)) = (
            edges.binary_search(&[a, b]),
            edges.binary_search(&[a, c]),
            edges.binary_search(&[b, c]),
        ) else {
            continue;
        };
        reduce_column_z2(
            [e0, e1, e2],
            words,
            &mut col,
            &mut pivot_for_row,
            &mut basis,
        );
    }
    basis.len()
}

/// Computes Betti numbers up to dimension 1 `(β₀, β₁)`.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::tda::{
///     PointCloud, Point2D, vietoris_rips_complex, betti_numbers
/// };
///
/// let points = vec![
///     Point2D::new(0.0, 0.0),
///     Point2D::new(1.0, 0.0),
///     Point2D::new(0.5, 0.866),
/// ];
/// let cloud = PointCloud::new(points).unwrap();
/// let complex = vietoris_rips_complex(&cloud, 1.0).unwrap();
///
/// let (beta0, beta1) = betti_numbers(&complex).unwrap();
/// assert_eq!(beta0, 1);
/// assert_eq!(beta1, 0);
/// ```
#[verified_engine::verified]
pub fn betti_numbers(complex: &SimplicialComplex) -> Result<(usize, usize), TdaError> {
    let beta0 = betti_number_0(complex)?;
    let beta1 = betti_number_1(complex)?;
    Ok((beta0, beta1))
}

#[cfg(test)]
#[path = "homology_tests.rs"]
mod tests;
