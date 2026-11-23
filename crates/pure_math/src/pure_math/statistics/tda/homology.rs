//! Homology computations and Betti numbers.

use super::boundary::Z2BoundaryWorkspace;
use super::complex::SimplicialComplex;
use super::core::Simplex;
use crate::error::TdaError;
use std::collections::{HashMap, HashSet};

/// Union-Find data structure for connected component tracking.
#[derive(Debug, Clone)]
pub(crate) struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    components: usize,
}

impl UnionFind {
    #[verified_engine::verified]
    pub(crate) fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            components: n,
        }
    }

    #[verified_engine::verified]
    pub(crate) fn reset(&mut self) {
        let n = self.parent.len();
        self.reset_prefix(n);
    }

    #[verified_engine::verified]
    pub(crate) fn reset_prefix(&mut self, n: usize) {
        let limit = n.min(self.parent.len());
        for (i, slot) in self.parent.iter_mut().enumerate().take(limit) {
            *slot = i;
        }
        self.rank[..limit].fill(0);
        self.components = limit;
    }

    #[verified_engine::verified]
    pub(crate) fn find(&mut self, x: usize) -> usize {
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
    pub(crate) fn union(&mut self, x: usize, y: usize) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);
        if root_x == root_y {
            return false;
        }

        if self.rank[root_x] < self.rank[root_y] {
            self.parent[root_x] = root_y;
        } else if self.rank[root_x] > self.rank[root_y] {
            self.parent[root_y] = root_x;
        } else {
            self.parent[root_y] = root_x;
            self.rank[root_x] += 1;
        }
        self.components -= 1;
        true
    }

    #[verified_engine::verified]
    pub(crate) fn count_components(&self) -> usize {
        self.components
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
    let Some(vertices) = complex.simplices.first() else {
        return Ok(0);
    };
    let n = vertices.len();
    if n == 0 {
        return Ok(0);
    }

    let edges = complex.simplices.get(1);
    let dense_ids = vertices
        .iter()
        .all(|v| v.vertices.len() == 1 && v.vertices[0] < n);
    if dense_ids {
        return betti_0_dense(n, edges);
    }
    betti_0_sparse(vertices, edges)
}

#[verified_engine::verified]
fn betti_0_dense(n: usize, edges: Option<&HashSet<Simplex>>) -> Result<usize, TdaError> {
    let mut uf = UnionFind::new(n);
    if let Some(edge_set) = edges {
        for edge in edge_set {
            if edge.vertices.len() != 2 || edge.vertices[0] >= n || edge.vertices[1] >= n {
                return Err(TdaError::InvalidSimplex {
                    reason: "invalid 1-simplex vertex index in complex".to_string(),
                });
            }
            uf.union(edge.vertices[0], edge.vertices[1]);
        }
    }
    Ok(uf.count_components())
}

#[verified_engine::verified]
fn betti_0_sparse(
    vertices: &HashSet<Simplex>,
    edges: Option<&HashSet<Simplex>>,
) -> Result<usize, TdaError> {
    let n = vertices.len();
    let mut vertex_to_index = HashMap::with_capacity(n);
    for (idx, vertex) in vertices.iter().enumerate() {
        if vertex.vertices.len() != 1 {
            return Err(TdaError::InvalidSimplex {
                reason: "expected 0-simplex with 1 vertex".to_string(),
            });
        }
        vertex_to_index.insert(vertex.vertices[0], idx);
    }

    let mut uf = UnionFind::new(n);
    if let Some(edge_set) = edges {
        for edge in edge_set {
            if edge.vertices.len() != 2 {
                return Err(TdaError::InvalidSimplex {
                    reason: "expected 1-simplex with 2 vertices".to_string(),
                });
            }
            let (&i, &j) = vertex_to_index
                .get(&edge.vertices[0])
                .zip(vertex_to_index.get(&edge.vertices[1]))
                .ok_or_else(|| TdaError::InvalidSimplex {
                    reason: "1-simplex references missing 0-simplex vertex".to_string(),
                })?;
            uf.union(i, j);
        }
    }
    Ok(uf.count_components())
}

/// Evaluates $\beta_1 = (|E| - |V| + \beta_0) - \operatorname{rank}_{\mathbb{Z}_2}(\partial_2)$
/// from precomputed chain counts.
#[verified_engine::verified]
pub(crate) fn betti_1_from_counts(
    v: usize,
    e: usize,
    beta0: usize,
    rank_partial_2: usize,
) -> usize {
    if v == 0 || e == 0 {
        return 0;
    }
    (e + beta0).saturating_sub(v).saturating_sub(rank_partial_2)
}

/// Computes the 1st Betti number (number of 1-dimensional holes/cycles) over $\mathbb{Z}_2$.
///
/// Uses the exact rank-nullity identity on the simplicial chain complex:
///
/// $$\beta_1 = \dim(\ker \partial_1) - \dim(\operatorname{im} \partial_2) = (|E| - |V| + \beta_0) - \operatorname{rank}_{\mathbb{Z}_2}(\partial_2)$$
///
/// Unlike raw triangle subtraction, $\mathbb{Z}_2$ column reduction of $\partial_2$ accounts for
/// linearly dependent 2-cycles (such as the fourth face of a $K_4$ clique) without erasing
/// independent 1-cycles elsewhere in the complex.
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
/// let beta1 = betti_number_1(&complex).unwrap();
/// assert_eq!(beta1, 1);
/// ```
#[verified_engine::verified]
pub fn betti_number_1(complex: &SimplicialComplex) -> Result<usize, TdaError> {
    if complex.count_simplices(0) == 0 || complex.count_simplices(1) == 0 {
        return Ok(0);
    }
    let beta0 = betti_number_0(complex)?;
    betti_number_1_with_beta0(complex, beta0)
}

#[verified_engine::verified]
fn betti_number_1_with_beta0(complex: &SimplicialComplex, beta0: usize) -> Result<usize, TdaError> {
    let v = complex.count_simplices(0);
    let e = complex.count_simplices(1);
    let cycles = betti_1_from_counts(v, e, beta0, 0);
    if cycles == 0 {
        return Ok(0);
    }
    let Some(tri_set) = complex.simplices.get(2).filter(|s| !s.is_empty()) else {
        return Ok(cycles);
    };
    let Some(edge_set) = complex.simplices.get(1) else {
        return Ok(0);
    };

    let edges = collect_sorted_edge_pairs(edge_set)?;
    let triangles = collect_sorted_tri_triples(tri_set)?;
    let rank_partial_2 = compute_partial_2_rank(&edges, &triangles, cycles)?;
    Ok(betti_1_from_counts(v, e, beta0, rank_partial_2))
}

#[verified_engine::verified]
fn collect_sorted_edge_pairs(edge_set: &HashSet<Simplex>) -> Result<Vec<[usize; 2]>, TdaError> {
    let mut edges = Vec::with_capacity(edge_set.len());
    for edge in edge_set {
        if edge.vertices.len() != 2 {
            return Err(TdaError::InvalidSimplex {
                reason: format!(
                    "expected 1-simplex with 2 vertices, got {}",
                    edge.vertices.len()
                ),
            });
        }
        edges.push([edge.vertices[0], edge.vertices[1]]);
    }
    edges.sort_unstable();
    Ok(edges)
}

#[verified_engine::verified]
fn collect_sorted_tri_triples(tri_set: &HashSet<Simplex>) -> Result<Vec<[usize; 3]>, TdaError> {
    let mut triangles = Vec::with_capacity(tri_set.len());
    for tri in tri_set {
        if tri.vertices.len() != 3 {
            return Err(TdaError::InvalidSimplex {
                reason: format!(
                    "expected 2-simplex with 3 vertices, got {}",
                    tri.vertices.len()
                ),
            });
        }
        triangles.push([tri.vertices[0], tri.vertices[1], tri.vertices[2]]);
    }
    triangles.sort_unstable();
    Ok(triangles)
}

#[verified_engine::verified]
fn compute_partial_2_rank(
    edges: &[[usize; 2]],
    triangles: &[[usize; 3]],
    max_rank: usize,
) -> Result<usize, TdaError> {
    let mut workspace = Z2BoundaryWorkspace::new(edges.len(), triangles.len())?;
    for (col_idx, &[v0, v1, v2]) in triangles.iter().enumerate() {
        let f0 = find_edge_row(edges, v0, v1)?;
        let f1 = find_edge_row(edges, v0, v2)?;
        let f2 = find_edge_row(edges, v1, v2)?;
        workspace.set_column_faces(col_idx, &[f0, f1, f2])?;
    }
    Ok(workspace.reduce_with_max_rank(max_rank))
}

#[verified_engine::verified]
fn find_edge_row(edges: &[[usize; 2]], u: usize, v: usize) -> Result<usize, TdaError> {
    let pair = if u <= v { [u, v] } else { [v, u] };
    edges
        .binary_search(&pair)
        .map_err(|_| TdaError::InvalidSimplex {
            reason: format!(
                "missing boundary edge [{}, {}] in simplicial complex",
                pair[0], pair[1]
            ),
        })
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
    let beta1 = betti_number_1_with_beta0(complex, beta0)?;
    Ok((beta0, beta1))
}

#[cfg(test)]
#[path = "homology_tests.rs"]
mod tests;

// [cite:clinical_trials_statistics]
