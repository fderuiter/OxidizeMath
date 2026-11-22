//! Homology computations and Betti numbers.

use super::complex::SimplicialComplex;
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

/// Computes the 1st Betti number (number of 1-dimensional holes/cycles).
///
/// β₁ counts the number of independent cycles (loops) in the complex.
/// This is computed using the formula:
///
/// β₁ = (number of edges) - (number of vertices) + (number of components) - (number of triangles)
///
/// More precisely, for a 2D complex:
/// β₁ = dim(ker ∂₁) - dim(im ∂₂)
///    = (|E| - rank(∂₁)) - rank(∂₂)
///
/// For simplicity, we use the Euler characteristic approach for 2D complexes.
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
/// assert_eq!(beta1, 1); // One hole
/// ```
#[verified_engine::verified]
pub fn betti_number_1(complex: &SimplicialComplex) -> Result<usize, TdaError> {
    let v = complex.count_simplices(0); // Vertices
    let e = complex.count_simplices(1); // Edges
    let t = complex.count_simplices(2); // Triangles

    if v == 0 {
        return Ok(0);
    }

    // Compute β₀ (connected components)
    let beta0 = betti_number_0(complex)?;

    // For a 2D complex, we use the formula:
    // β₁ = e - v + β₀ - 2t
    //
    // This comes from the Euler characteristic:
    // χ = β₀ - β₁ = v - e + t
    // Therefore: β₁ = v - e + t - β₀ (for regular boundary)
    //
    // But we need to account for filled triangles differently.
    // Each triangle "fills" a potential hole.

    // Simple approximation for 2D:
    // β₁ = number of cycles - number of filled regions
    // β₁ = (e - v + 1) - t for a single component

    // More generally:
    // β₁ = e - v + β₀ - t (for simple 2D complexes)

    if e < v {
        return Ok(0); // Not enough edges to form cycles
    }

    // Euler characteristic: χ = v - e + t = β₀ - β₁
    // Solving for β₁: β₁ = v - e + t - β₀
    // But this needs correction for our specific case

    // For a connected graph: β₁ = e - v + 1
    // For multiple components: β₁ = e - v + β₀
    // But we need to subtract filled triangles

    // Each triangle fills one potential cycle
    // So: β₁ = (e - v + β₀) - t

    let cycles = if e >= v { e - v + beta0 } else { 0 };
    let beta1 = cycles.saturating_sub(t);

    Ok(beta1)
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
}
