//! Homology computations and Betti numbers.

use super::boundary::Z2BoundaryWorkspace;
use super::complex::SimplicialComplex;
use super::core::Simplex;
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
            // Union by rank
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
/// β₀ counts the number of connected components in the simplicial complex.
///
/// # Algorithm
///
/// Uses Union-Find to efficiently track connected components as we process edges.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::tda::{
///     PointCloud, Point2D, vietoris_rips_complex, betti_number_0
/// };
///
/// // Two separate clusters
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
/// assert_eq!(beta0, 2); // Two connected components
/// ```
#[verified_engine::verified]
pub fn betti_number_0(complex: &SimplicialComplex) -> Result<usize, TdaError> {
    let vertices = complex.get_simplices(0);
    if vertices.is_empty() {
        return Ok(0);
    }

    // Map vertex simplices to indices
    let n = vertices.len();
    let mut vertex_to_index = HashMap::new();
    for (i, vertex) in vertices.iter().enumerate() {
        vertex_to_index.insert(vertex.vertices[0], i);
    }

    let mut uf = UnionFind::new(n);

    // Process all edges to connect components
    let edges = complex.get_simplices(1);
    for edge in edges {
        let i = vertex_to_index[&edge.vertices[0]];
        let j = vertex_to_index[&edge.vertices[1]];
        uf.union(i, j);
    }

    Ok(uf.count_components())
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
/// // Points forming a square (one hole in the middle)
/// let points = vec![
///     Point2D::new(0.0, 0.0),
///     Point2D::new(1.0, 0.0),
///     Point2D::new(1.0, 1.0),
///     Point2D::new(0.0, 1.0),
/// ];
/// let cloud = PointCloud::new(points).unwrap();
///
/// // Radius 1.1 connects adjacent points (dist=1.0) but not diagonals (dist=1.414)
/// let complex = vietoris_rips_complex(&cloud, 1.1).unwrap();
/// let beta1 = betti_number_1(&complex).unwrap();
/// assert_eq!(beta1, 1);
/// ```
#[verified_engine::verified]
pub fn betti_number_1(complex: &SimplicialComplex) -> Result<usize, TdaError> {
    let v = complex.count_simplices(0);
    if v == 0 {
        return Ok(0);
    }

    let mut edges = complex.get_simplices(1);
    let e = edges.len();
    if e == 0 {
        return Ok(0);
    }

    let beta0 = betti_number_0(complex)?;
    let cycles = (e + beta0).saturating_sub(v);
    if cycles == 0 {
        return Ok(0);
    }

    let mut triangles = complex.get_simplices(2);
    if triangles.is_empty() {
        return Ok(cycles);
    }

    edges.sort_unstable();
    triangles.sort_unstable();
    let rank_partial_2 = compute_partial_2_rank(&edges, &triangles)?;
    Ok(cycles.saturating_sub(rank_partial_2))
}

#[verified_engine::verified]
fn compute_partial_2_rank(edges: &[Simplex], triangles: &[Simplex]) -> Result<usize, TdaError> {
    let mut workspace = Z2BoundaryWorkspace::new(edges.len(), triangles.len())?;
    for (col_idx, tri) in triangles.iter().enumerate() {
        if tri.vertices.len() != 3 {
            return Err(TdaError::InvalidSimplex {
                reason: format!(
                    "expected 2-simplex with 3 vertices, got {}",
                    tri.vertices.len()
                ),
            });
        }
        let v0 = tri.vertices[0];
        let v1 = tri.vertices[1];
        let v2 = tri.vertices[2];
        let f0 = find_edge_row(edges, v0, v1)?;
        let f1 = find_edge_row(edges, v0, v2)?;
        let f2 = find_edge_row(edges, v1, v2)?;
        workspace.set_column_faces(col_idx, &[f0, f1, f2])?;
    }
    Ok(workspace.reduce())
}

#[verified_engine::verified]
fn find_edge_row(edges: &[Simplex], u: usize, v: usize) -> Result<usize, TdaError> {
    let (a, b) = if u <= v { (u, v) } else { (v, u) };
    edges
        .binary_search_by(|edge| edge.vertices.as_slice().cmp(&[a, b]))
        .map_err(|_| TdaError::InvalidSimplex {
            reason: format!("missing boundary edge [{a}, {b}] in simplicial complex"),
        })
}

/// Computes Betti numbers up to dimension 1.
///
/// Returns (β₀, β₁) where:
/// - β₀ = number of connected components
/// - β₁ = number of 1-dimensional holes (cycles)
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
/// println!("Connected components: {}", beta0);
/// println!("Holes: {}", beta1);
/// ```
#[verified_engine::verified]
pub fn betti_numbers(complex: &SimplicialComplex) -> Result<(usize, usize), TdaError> {
    let beta0 = betti_number_0(complex)?;
    let beta1 = betti_number_1(complex)?;
    Ok((beta0, beta1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure_math::statistics::tda::complex::vietoris_rips_complex;
    use crate::pure_math::statistics::tda::core::{Point2D, PointCloud, Simplex};

    #[test]
    #[verified_engine::verified]
    fn test_betti_number_0_single_vertex() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0]).unwrap()).unwrap();

        let beta0 = betti_number_0(&complex).unwrap();
        assert_eq!(beta0, 1);
    }

    #[test]
    #[verified_engine::verified]
    fn test_betti_number_0_two_components() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0]).unwrap()).unwrap();
        complex.add_simplex(Simplex::new(vec![1]).unwrap()).unwrap();
        // No edge connecting them

        let beta0 = betti_number_0(&complex).unwrap();
        assert_eq!(beta0, 2);
    }

    #[test]
    #[verified_engine::verified]
    fn test_betti_number_0_connected() {
        let mut complex = SimplicialComplex::new();
        complex
            .add_simplex(Simplex::new(vec![0, 1]).unwrap())
            .unwrap();

        let beta0 = betti_number_0(&complex).unwrap();
        assert_eq!(beta0, 1);
    }

    #[test]
    #[verified_engine::verified]
    fn test_betti_number_1_no_cycle() {
        // Simple edge: no cycle
        let mut complex = SimplicialComplex::new();
        complex
            .add_simplex(Simplex::new(vec![0, 1]).unwrap())
            .unwrap();

        let beta1 = betti_number_1(&complex).unwrap();
        assert_eq!(beta1, 0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_betti_number_1_triangle_hollow() {
        // Three edges forming a triangle (hollow - one cycle)
        let mut complex = SimplicialComplex::new();
        complex
            .add_simplex(Simplex::new(vec![0, 1]).unwrap())
            .unwrap();
        complex
            .add_simplex(Simplex::new(vec![1, 2]).unwrap())
            .unwrap();
        complex
            .add_simplex(Simplex::new(vec![2, 0]).unwrap())
            .unwrap();

        let beta1 = betti_number_1(&complex).unwrap();
        assert_eq!(beta1, 1); // One hole
    }

    #[test]
    #[verified_engine::verified]
    fn test_betti_number_1_triangle_filled() {
        // Filled triangle (no hole)
        let mut complex = SimplicialComplex::new();
        complex
            .add_simplex(Simplex::new(vec![0, 1, 2]).unwrap())
            .unwrap();

        let beta1 = betti_number_1(&complex).unwrap();
        assert_eq!(beta1, 0); // No holes
    }

    #[test]
    #[verified_engine::verified]
    fn test_betti_numbers_two_clusters() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(10.0, 0.0),
            Point2D::new(11.0, 0.0),
        ];
        let cloud = PointCloud::new(points).unwrap();
        let complex = vietoris_rips_complex(&cloud, 1.5).unwrap();

        let (beta0, beta1) = betti_numbers(&complex).unwrap();
        assert_eq!(beta0, 2); // Two components
        assert_eq!(beta1, 0); // No holes
    }

    #[test]
    #[verified_engine::verified]
    fn test_betti_numbers_circle() {
        // Points arranged in a circle
        let n = 8;
        let points: Vec<Point2D> = (0..n)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
                Point2D::new(angle.cos(), angle.sin())
            })
            .collect();

        let cloud = PointCloud::new(points).unwrap();

        // Small radius: forms a cycle around the circle
        let complex = vietoris_rips_complex(&cloud, 0.8).unwrap();
        let (beta0, beta1) = betti_numbers(&complex).unwrap();

        assert_eq!(beta0, 1); // Connected
        assert_eq!(beta1, 1); // One hole in the middle
    }

    #[test]
    #[verified_engine::verified]
    fn test_betti_numbers_line() {
        // Points on a line
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(3.0, 0.0),
        ];
        let cloud = PointCloud::new(points).unwrap();
        let complex = vietoris_rips_complex(&cloud, 1.5).unwrap();

        let (beta0, beta1) = betti_numbers(&complex).unwrap();
        assert_eq!(beta0, 1); // Connected
        assert_eq!(beta1, 0); // No holes
    }

    #[test]
    #[verified_engine::verified]
    fn test_betti_number_1_retains_cycle_near_k4_clique() {
        // Vertices 0, 1, 2, 3 form a complete K4 clique (6 edges, 4 triangles, rank(partial_2) = 3).
        // Vertices 3, 4, 5, 6 form an attached hollow 4-cycle (edges [3,4], [4,5], [5,6], [6,3], 0 triangles).
        // Total: V = 7, E = 10, T = 4, beta_0 = 1.
        // Heuristic (E - V + beta_0).saturating_sub(T) = (10 - 7 + 1) - 4 = 0 (WRONG: erases the real hole!).
        // Exact Z_2 reduction: (10 - 7 + 1) - rank_z2(partial_2) = 4 - 3 = 1.
        let mut complex = SimplicialComplex::new();
        for &(u, v, w) in &[(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)] {
            complex
                .add_simplex(Simplex::new(vec![u, v, w]).unwrap())
                .unwrap();
        }
        for &(u, v) in &[(3, 4), (4, 5), (5, 6), (6, 3)] {
            complex
                .add_simplex(Simplex::new(vec![u, v]).unwrap())
                .unwrap();
        }

        assert_eq!(complex.count_simplices(0), 7);
        assert_eq!(complex.count_simplices(1), 10);
        assert_eq!(complex.count_simplices(2), 4);
        let (beta0, beta1) = betti_numbers(&complex).unwrap();
        assert_eq!(beta0, 1);
        assert_eq!(beta1, 1);
    }

    #[test]
    #[verified_engine::verified]
    fn test_betti_number_1_with_isolated_vertices_and_k4() {
        // Hollow triangle [0,1],[1,2],[2,0] + disjoint K4 [3,4,5,6] + isolated vertex [7].
        let mut complex = SimplicialComplex::new();
        for &(u, v) in &[(0, 1), (1, 2), (2, 0)] {
            complex
                .add_simplex(Simplex::new(vec![u, v]).unwrap())
                .unwrap();
        }
        for &(u, v, w) in &[(3, 4, 5), (3, 4, 6), (3, 5, 6), (4, 5, 6)] {
            complex
                .add_simplex(Simplex::new(vec![u, v, w]).unwrap())
                .unwrap();
        }
        complex.add_simplex(Simplex::new(vec![7]).unwrap()).unwrap();

        let (beta0, beta1) = betti_numbers(&complex).unwrap();
        assert_eq!(beta0, 3);
        assert_eq!(beta1, 1);
    }
}

// [cite:clinical_trials_statistics]
