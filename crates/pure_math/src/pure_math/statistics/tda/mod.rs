//! # Topological Data Analysis (TDA)
//!
//! This module implements fundamental concepts from Topological Data Analysis,
//! focusing on persistent homology and the analysis of point cloud data.
//!
//! ## Overview
//!
//! Topological Data Analysis provides tools for understanding the "shape" of data
//! by studying topological features that persist across multiple scales. Unlike
//! traditional statistical methods that focus on means and variances, TDA captures
//! qualitative features like:
//!
//! - **Connected components** (β₀): Separate clusters
//! - **Loops/holes** (β₁): Circular patterns
//! - **Voids** (β₂): Three-dimensional cavities (not implemented)
//!
//! ## Key Concepts
//!
//! ### Point Clouds
//!
//! A point cloud is a finite set of points in metric space. TDA analyzes the
//! topological structure that emerges from these points at different scales.
//!
//! ### Simplicial Complexes
//!
//! A **simplicial complex** is a collection of simplices (vertices, edges, triangles, etc.)
//! with the property that:
//! - Every face of a simplex is also in the complex
//! - The intersection of two simplices is a face of both
//!
//! Types of simplices:
//! - **0-simplex**: Vertex (single point)
//! - **1-simplex**: Edge (two points)
//! - **2-simplex**: Triangle (three points)
//! - **3-simplex**: Tetrahedron (four points)
//!
//! ### Vietoris-Rips Complex
//!
//! The Vietoris-Rips complex VR(X, ε) at radius ε includes:
//!
//! ```text
//! [v₀, v₁, ..., vₖ] ∈ VR(X, ε) ⟺ d(vᵢ, vⱼ) ≤ ε for all i, j
//! ```
//!
//! In words: include a simplex if all pairwise distances are at most ε.
//!
//! ### Betti Numbers
//!
//! Betti numbers count topological features of different dimensions:
//!
//! - **β₀**: Number of connected components
//! - **β₁**: Number of 1-dimensional holes (loops)
//! - **β₂**: Number of 2-dimensional voids (cavities)
//!
//! This implementation computes β₀ and β₁.
//!
//! #### Computing β₀ (Connected Components)
//!
//! We use a **Union-Find** data structure to efficiently track which vertices
//! are connected through edges:
//!
//! ```text
//! 1. Initialize: Each vertex is its own component
//! 2. For each edge [i, j]: Union(i, j)
//! 3. Count distinct root components
//! ```
//!
//! #### Computing β₁ (Cycles/Holes)
//!
//! For a 2D simplicial complex over $\mathbb{Z}_2$, we reduce the boundary matrix $\partial_2$:
//!
//! ```text
//! β₁ = (# edges) - (# vertices) + β₀ - rank_Z2(∂₂)
//! ```
//!
//! This retains genuine 1-cycles even near higher-order cliques such as $K_4$ where raw
//! triangle counts exceed boundary rank.
//!
//! ### Persistence
//!
//! A **filtration** is a nested sequence of complexes:
//!
//! ```text
//! ∅ = K₀ ⊆ K₁ ⊆ K₂ ⊆ ... ⊆ Kₙ
//! ```
//!
//! As we increase the radius ε, features:
//! - **Birth**: Appear at some radius
//! - **Death**: Disappear at a later radius
//!
//! The **persistence** of a feature is its lifetime: `death - birth`.
//!
//! Features with large persistence are "real" patterns in the data,
//! while short-lived features are likely noise.
//!
//! ### Persistence Barcode
//!
//! A **barcode** visualizes feature lifetimes as horizontal bars:
//!
//! ```text
//! β₀:  |-----------|          (long-lived component)
//! β₀:      |--|                (short-lived component)
//! β₁:         |---------|      (persistent hole)
//! β₁:             |-|          (noise)
//!      0   1   2   3   4  ε
//! ```
//!
//! ## Example: Basic TDA Workflow
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::tda::{
//!     PointCloud, Point2D, vietoris_rips_complex, betti_numbers
//! };
//!
//! // Create a point cloud
//! let points = vec![
//!     Point2D::new(0.0, 0.0),
//!     Point2D::new(1.0, 0.0),
//!     Point2D::new(0.5, 0.866),
//! ];
//! let cloud = PointCloud::new(points).unwrap();
//!
//! // Build Vietoris-Rips complex at radius 1.0
//! let complex = vietoris_rips_complex(&cloud, 1.0).unwrap();
//!
//! // Compute Betti numbers
//! let (beta0, beta1) = betti_numbers(&complex).unwrap();
//! println!("Connected components: {}", beta0);
//! println!("Holes: {}", beta1);
//! ```
//!
//! ## Example: Detecting Clusters
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::tda::{
//!     PointCloud, Point2D, vietoris_rips_complex, betti_number_0
//! };
//!
//! // Two well-separated clusters
//! let points = vec![
//!     // Cluster 1
//!     Point2D::new(0.0, 0.0),
//!     Point2D::new(1.0, 0.0),
//!     Point2D::new(0.5, 0.5),
//!     // Cluster 2
//!     Point2D::new(10.0, 10.0),
//!     Point2D::new(11.0, 10.0),
//!     Point2D::new(10.5, 10.5),
//! ];
//! let cloud = PointCloud::new(points).unwrap();
//!
//! // Small radius: two separate components
//! let complex = vietoris_rips_complex(&cloud, 2.0).unwrap();
//! let components = betti_number_0(&complex).unwrap();
//! assert_eq!(components, 2);
//! ```
//!
//! ## Example: Detecting Circular Patterns
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::tda::{
//!     PointCloud, Point2D, vietoris_rips_complex, betti_numbers
//! };
//!
//! // Points arranged in a circle
//! let n = 12;
//! let points: Vec<Point2D> = (0..n)
//!     .map(|i| {
//!         let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
//!         Point2D::new(angle.cos(), angle.sin())
//!     })
//!     .collect();
//! let cloud = PointCloud::new(points).unwrap();
//!
//! // At the right radius, we see one hole in the middle
//! let complex = vietoris_rips_complex(&cloud, 0.6).unwrap();
//! let (beta0, beta1) = betti_numbers(&complex).unwrap();
//!
//! println!("Connected: {} component(s)", beta0);
//! println!("Holes: {}", beta1);
//! ```
//!
//! ## Example: Persistence Analysis
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::tda::{
//!     PointCloud, Point2D, compute_persistence
//! };
//!
//! // Create a noisy circle
//! let mut points = vec![];
//! for i in 0..20 {
//!     let angle = 2.0 * std::f64::consts::PI * (i as f64) / 20.0;
//!     points.push(Point2D::new(angle.cos(), angle.sin()));
//! }
//! let cloud = PointCloud::new(points).unwrap();
//!
//! // Compute persistence across multiple scales
//! let radii: Vec<f64> = (0..30).map(|i| i as f64 * 0.05).collect();
//! let barcode = compute_persistence(&cloud, &radii).unwrap();
//!
//! // Filter for significant features
//! let significant = barcode.filter_by_persistence(0.2);
//! println!("Found {} significant features", significant.len());
//!
//! // Find the most persistent hole
//! if let Some(hole) = barcode.most_persistent(1) {
//!     println!("Most persistent hole:");
//!     println!("  Birth: {:.3}", hole.birth);
//!     println!("  Death: {:.3}", hole.death);
//!     println!("  Persistence: {:.3}", hole.persistence());
//! }
//! ```
//!
//! ## Applications
//!
//! ### Data Analysis
//! - **Clustering**: Identify natural groupings (β₀)
//! - **Shape detection**: Find circular or spherical patterns (β₁)
//! - **Anomaly detection**: Unusual topological features
//!
//! ### Science & Engineering
//! - **Neuroscience**: Analyze neural connectivity patterns
//! - **Materials science**: Study porous materials and molecular structures
//! - **Sensor networks**: Detect coverage holes
//!
//! ### Machine Learning
//! - **Feature extraction**: Topological features for classification
//! - **Dimensionality reduction**: Preserve topological structure
//! - **Model interpretation**: Understand decision boundaries
//!
//! ## Computational Complexity
//!
//! - **Vietoris-Rips construction**: O(n² d) for d-dimensional simplices
//! - **β₀ computation**: O(n α(n)) with Union-Find (nearly linear)
//! - **β₁ computation**: O(n² + e) where e is the number of edges
//! - **Persistence**: O(k × complexity of Betti computation) for k scales
//!
//! ## Limitations
//!
//! 1. **Computational cost**: Full complexes can be very large
//! 2. **Choice of metric**: Results depend on distance function
//! 3. **Parameter selection**: Filtration scales must be chosen carefully
//! 4. **Dimension limit**: This implementation only computes β₀ and β₁
//!
//! ## References
//!
//! - Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An Introduction*.
//!   American Mathematical Society.
//! - Ghrist, R. (2008). *Barcodes: The persistent topology of data*.
//!   Bulletin of the American Mathematical Society, 45(1), 61-75.
//! - Carlsson, G. (2009). *Topology and data*. Bulletin of the American
//!   Mathematical Society, 46(2), 255-308.

pub mod boundary;
pub mod complex;
pub mod core;
pub mod homology;
pub mod persistence;
#[cfg(feature = "serde")]
mod serde_support;

// Re-export main types and functions
pub use boundary::{Z2BoundaryWorkspace, rank_z2_boundary};
pub use complex::{
    MAX_FILTRATION_TRIANGLES, SimplicialComplex, vietoris_rips_complex, vietoris_rips_complex_nd,
    vietoris_rips_from_distance_matrix,
};
pub use core::{MAX_TDA_DIMENSION, MAX_TDA_POINTS, NdPointCloud, Point2D, PointCloud, Simplex};
pub use homology::{betti_number_0, betti_number_1, betti_numbers};
pub use persistence::{
    PersistenceBarcode, PersistenceInterval, RipsInterval, RipsPersistenceResult,
    compute_persistence, compute_rips_persistence, compute_rips_persistence_from_distances,
    compute_rips_persistence_nd,
};

// [cite:clinical_trials_statistics]
