//! # Neuroimaging: FreeSurfer Reconstruction
//!
//! This module provides tools for **cortical reconstruction** and **volumetric segmentation**,
//! inspired by the standard FreeSurfer pipeline (`freesurfer.tex`). It transforms structural MRI
//! volumes into triangulated white-matter and pial surfaces, Potts Markov Random Field (MRF)
//! tissue segmentations, symmetric cortical thickness maps, and vertex-wise GLM statistical maps.
//!
//! ## The Pipeline
//!
//! Processing MRI data involves transforming raw voxel intensities into geometric surface models.
//!
//! ```mermaid
//! graph TD
//!     MRI[MRI Volume] -->|Potts MRF ICM| Seg[Segmentation]
//!     Seg -->|White Matter| WhiteSurf[White Surface]
//!     Seg -->|Pial Boundary| PialSurf[Pial Surface]
//!
//!     subgraph Surface_Evolution
//!     WhiteSurf -->|Tension & Bending| EvolveW[Deformable Model]
//!     PialSurf -->|Smoothed MRI Gradient| EvolveP[Deformable Model]
//!     end
//!
//!     EvolveW --> FinalWhite[Final White Surface]
//!     EvolveP --> FinalPial[Final Pial Surface]
//!
//!     FinalWhite & FinalPial --> Thickness[Cortical Thickness Map]
//!     FinalWhite & FinalPial --> GLM[GLM Contrast t-Map]
//!
//!     style MRI fill:#f9f,stroke:#333
//!     style Thickness fill:#dfd,stroke:#333
//! ```
//!
//! ## Quick Start: Triangulated Cortical Thickness & Energy
//!
//! Measure batch cortical thickness and thin-plate internal smoothness energy across triangulated
//! white-matter (inner) and pial (outer) cortical surface patches.
//!
//! ```rust
//! # use domain_applied::__doctest_facade as math_explorer;
//! use math_explorer::applied::freesurfer::{
//!     cortical_thickness, cortical_thickness_map, internal_energy, Surface,
//! };
//!
//! let faces = vec![[0, 1, 2], [1, 3, 2]];
//! let white_surface = Surface::new(
//!     vec![
//!         [0.0, 0.0, 0.0],
//!         [1.0, 0.0, 0.0],
//!         [0.0, 1.0, 0.0],
//!         [1.0, 1.0, 0.0],
//!     ],
//!     faces.clone(),
//! );
//! let pial_surface = Surface::new(
//!     vec![
//!         [0.0, 0.0, 2.5],
//!         [1.0, 0.0, 2.5],
//!         [0.0, 1.0, 2.5],
//!         [1.0, 1.0, 2.5],
//!     ],
//!     faces,
//! );
//!
//! let map = cortical_thickness_map(&white_surface, &pial_surface);
//! assert_eq!(map, vec![2.5, 2.5, 2.5, 2.5]);
//! assert_eq!(
//!     cortical_thickness(
//!         &white_surface.vertices[0],
//!         &pial_surface.vertices[0],
//!         &white_surface,
//!         &pial_surface,
//!     ),
//!     2.5
//! );
//! assert!(internal_energy(&white_surface, 0.5, 0.2) > 0.0);
//! ```
//!
//! ## Submodules
//!
//! - `segmentation`: Multi-label Bayesian classification and 6-connected Potts MRF ICM denoising.
//! - `surface`: Triangulated deformable surface models driven by thin-plate and smoothed MRI energies.
//! - `thickness`: Zero-allocation CSR `FlatSpatialGrid`, `ThicknessWorkspace`, and cortical thickness estimation.
//! - `glm`: Ordinary least squares parameter estimation and contrast `t_statistic` evaluation.

#[allow(missing_docs)]
pub mod glm;
#[allow(missing_docs)]
pub mod segmentation;
#[allow(missing_docs)]
pub mod surface;
#[allow(missing_docs)]
pub mod thickness;

// Re-export public items preserving baseline compatibility and exposing upgraded kernels
pub use glm::{GlmDesignQr, estimate_beta, t_statistic};
pub use segmentation::{
    MAX_TISSUE_CLASSES, TissueClass, bayesian_classification, mrf_segment_volume,
    mrf_voxel_posteriors, normalized_posteriors, normalized_posteriors_into,
};
pub use surface::{
    MriVolume, Surface, evolve_surface, evolve_surface_deformable, external_energy,
    external_energy_volume, internal_energy,
};
pub use thickness::{
    FlatSpatialGrid, ThicknessWorkspace, cortical_thickness, cortical_thickness_indexed,
    cortical_thickness_map, cortical_thickness_map_into,
};

// [cite:freesurfer]

use pure_math::theory_verification;

theory_verification!(
    module = "freesurfer",
    constants = {
        THICKNESS = 2.5;
    },
    test = {
        use approx::assert_relative_eq;
        use nalgebra::{DMatrix, DVector};

        // 1. Deformable surface energy descent on triangulated mesh + MRI boundary volume
        let mut surf = Surface::new(
            vec![
                [0.0, 0.0, 1.6],
                [1.0, 0.0, 0.9],
                [0.0, 1.0, 0.9],
                [-1.0, 0.0, 0.9],
                [0.0, -1.0, 0.9],
            ],
            vec![[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]],
        );
        let dims = [9, 9, 9];
        let mut vox = Vec::with_capacity(9 * 9 * 9);
        for iz in 0..9 {
            for iy in 0..9 {
                for ix in 0..9 {
                    let z = -2.0 + 0.5 * (iz as f64);
                    let _ = (ix, iy);
                    vox.push(if z < 1.0 { 110.0 } else { 40.0 });
                }
            }
        }
        let vol = MriVolume::new(dims, [-2.0, -2.0, -2.0], 0.5, vox).expect("valid MRI volume");
        let e0 = internal_energy(&surf, 0.4, 0.2) + external_energy_volume(&surf, &vol, 0.6);
        evolve_surface_deformable(&mut surf, &vol, 0.4, 0.2, 0.6, 0.015);
        let e1 = internal_energy(&surf, 0.4, 0.2) + external_energy_volume(&surf, &vol, 0.6);
        assert!(
            e1 < e0,
            "Deformable evolution must decrease total energy: {e1} >= {e0}"
        );

        // 2. Potts MRF smoothing removes isolated noisy voxel & normalizes posteriors
        let classes = [
            TissueClass::new(40.0, 10.0, 0.5),
            TissueClass::new(110.0, 10.0, 0.5),
        ];
        let post = normalized_posteriors(&[0.2, 0.6], &[0.5, 0.5]);
        assert_relative_eq!(
            post[0] + post[1],
            1.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        let mut noisy_vox = vec![110.0; 27];
        noisy_vox[13] = 68.0; // isolated center impulse surrounded by 6 white-matter neighbors
        let seg_vol = MriVolume::new([3, 3, 3], [0.0, 0.0, 0.0], 1.0, noisy_vox).unwrap();
        let mut ml_labels = [0_u8; 27];
        let mut mrf_labels = [0_u8; 27];
        let _ = mrf_segment_volume(&seg_vol, &classes, 0.0, 4, &mut ml_labels);
        let _ = mrf_segment_volume(&seg_vol, &classes, 1.5, 6, &mut mrf_labels);
        assert_eq!(ml_labels[13], 0);
        assert_eq!(mrf_labels[13], 1);

        // 3. Batch cortical thickness symmetry & THICKNESS constant verification
        let white = Surface::new(
            vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            vec![[0, 1, 2]],
        );
        let pial = Surface::new(
            vec![
                [0.0, 0.0, THICKNESS],
                [2.0, 0.0, THICKNESS],
                [0.0, 2.0, THICKNESS],
            ],
            vec![[0, 1, 2]],
        );
        let map_wp = cortical_thickness_map(&white, &pial);
        let map_pw = cortical_thickness_map(&pial, &white);
        assert_eq!(map_wp, map_pw);
        assert_relative_eq!(
            map_wp[0],
            THICKNESS,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );

        // 4. Vertex-wise GLM OLS and contrast t-statistic
        let x = DMatrix::from_row_slice(
            6,
            2,
            &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        );
        let y = DVector::from_vec(vec![2.9, 3.0, 3.1, 2.2, 2.3, 2.4]);
        let beta = estimate_beta(&x, &y).expect("invertible two-group design");
        let c = DVector::from_vec(vec![0.0, -1.0]);
        let res = &y - &x * &beta;
        let sigma2 = res.norm_squared() / 4.0;
        let t_val = t_statistic(&c, &beta, &x, sigma2).expect("defined t-statistic");
        assert!(t_val > 8.0);
    }
);
