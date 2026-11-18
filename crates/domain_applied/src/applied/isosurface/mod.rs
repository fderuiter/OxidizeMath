//! # Isosurface Extraction
//!
//! This module provides algorithms to extract 3D surfaces from volumetric data (scalar fields).
//! The primary algorithm implemented is **Marching Cubes**, which converts a 3D grid of values
//! (voxels) into a polygonal mesh of triangles.
//!
//! This is commonly used in:
//! - Medical Imaging (CT/MRI scans) to visualize bones or organs.
//! - Physics simulations to visualize fluid boundaries.
//! - Procedural generation for terrain or organic shapes.
//!
//! ## Quick Usage
//!
//! ```rust
//! use domain_applied::applied::isosurface::{extract_isosurface, VoxelGrid, Point3D};
//!
//! // 1. Create a tiny 2x2x2 grid
//! let grid = VoxelGrid::builder()
//! .dimensions(2, 2, 2)
//! .data(vec![0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0])
//! .voxel_size(Point3D::new(1.0, 1.0, 1.0))
//! .origin(Point3D::new(0.0, 0.0, 0.0))
//! .build()
//! .unwrap();
//!
//! // 2. Extract the surface where value == 0.0
//! let mesh = extract_isosurface(&grid, 0.0).unwrap();
//!
//! println!("Generated {} triangles", mesh.indices.len() / 3);
//! ```
//!
//! ## Examples
//!
//! See `math_explorer/examples/isosurface_torus.rs` for a complete example that:
//! 1. Generates a Signed Distance Field (SDF) for a torus.
//! 2. Extracts the mesh.
//! 3. Exports it to an `.obj` file for visualization in Blender/MeshLab.
//!
//! Run it with:
//! ```bash
//! cargo run --example isosurface_torus
//! ```

#[allow(missing_docs)]
pub mod gradients;
#[allow(missing_docs)]
pub mod marching_cubes;
#[allow(missing_docs)]
pub mod tables;
#[allow(missing_docs)]
pub mod types;

pub use gradients::{CentralDifferenceEstimator, GradientEstimator};
pub use marching_cubes::extract_isosurface;
pub use types::{Mesh, Point3D, Triangle, VoxelGrid};

// [cite:isosurface_extraction]

use pure_math::theory_verification;

theory_verification!(
    module = "isosurface",
    constants = {
        LEVEL = 0.0;
    },
    test = {
        assert_relative_eq!(LEVEL, 0.0, epsilon = math_commons::registry::TOLERANCE_FAST);
    }
);
