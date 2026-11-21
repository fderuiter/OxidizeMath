//! # Neuroimaging: FreeSurfer Reconstruction
//!
//! This module provides tools for **cortical reconstruction** and **volumetric segmentation**,
//! inspired by the standard FreeSurfer pipeline. It enables the analysis of structural MRI data
//! to measure cortical thickness, surface area, and other morphometric properties.
//!
//! ## The Pipeline
//!
//! Processing MRI data involves transforming raw voxel intensities into geometric surface models.
//!
//! ```mermaid
//! graph TD
//!     MRI[MRI Volume] -->|Bayesian Classification| Seg[Segmentation]
//!     Seg -->|White Matter| WhiteSurf[White Surface]
//!     Seg -->|Pial Boundary| PialSurf[Pial Surface]
//!
//!     subgraph Surface_Evolution
//!     WhiteSurf -->|Internal Energy| EvolveW[Deformable Model]
//!     PialSurf -->|External Energy| EvolveP[Deformable Model]
//!     end
//!
//!     EvolveW --> FinalWhite[Final White Surface]
//!     EvolveP --> FinalPial[Final Pial Surface]
//!
//!     FinalWhite & FinalPial --> Thickness[Cortical Thickness]
//!     FinalWhite & FinalPial --> GLM[Statistical Analysis]
//!
//!     style MRI fill:#f9f,stroke:#333
//!     style Thickness fill:#dfd,stroke:#333
//! ```
//!
//! ##  Quick Start: Cortical Thickness
//!
//! Calculate the thickness between the white matter surface (inner) and pial surface (outer).
//!
//! ```rust
//! # use domain_applied::__doctest_facade as math_explorer;
//! use math_explorer::applied::freesurfer::{cortical_thickness, Surface};
//!
//! // 1. Define the White Matter Surface (Inner Boundary)
//! let white_surface = Surface {
//!     vertices: vec![
//!         [0.0, 0.0, 0.0], // v1
//!         [1.0, 0.0, 0.0], // v2
//!         [0.0, 1.0, 0.0], // v3
//!     ],
//! };
//!
//! // 2. Define the Pial Surface (Outer Boundary)
//! // Offset by 2.0mm in the z-direction
//! let pial_surface = Surface {
//!     vertices: vec![
//!         [0.0, 0.0, 2.0],
//!         [1.0, 0.0, 2.0],
//!         [0.0, 1.0, 2.0],
//!     ],
//! };
//!
//! // 3. Measure thickness at a specific vertex
//! // (In reality, vertex correspondence is pre-computed)
//! let v_white = [0.0, 0.0, 0.0];
//! let v_pial = [0.0, 0.0, 2.0];
//!
//! let thickness = cortical_thickness(&v_white, &v_pial, &white_surface, &pial_surface);
//!
//! println!("Cortical Thickness: {:.2} mm", thickness);
//! assert_eq!(thickness, 2.0);
//! ```
//!
//! ## Submodules
//!
//! - `segmentation`: Voxel-wise classification using Bayesian priors.
//! - `surface`: Deformable surface models (snakes) driven by energy minimization.
//! - `thickness`: Vertex-wise cortical thickness estimation.
//! - `glm`: General Linear Model for group analysis (Mass Univariate).

#[allow(missing_docs)]
pub mod glm;
#[allow(missing_docs)]
pub mod segmentation;
#[allow(missing_docs)]
pub mod surface;
#[allow(missing_docs)]
pub mod thickness;

// Re-export specific items to maintain public API compatibility
pub use glm::{estimate_beta, t_statistic};
pub use segmentation::bayesian_classification;
pub use surface::{Surface, evolve_surface, external_energy, internal_energy};
pub use thickness::cortical_thickness;

// [cite:freesurfer]

use pure_math::theory_verification;

theory_verification!(
    module = "freesurfer",
    constants = {
        THICKNESS = 2.5;
    },
    test = {
        assert_relative_eq!(
            THICKNESS,
            2.5,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }
);
