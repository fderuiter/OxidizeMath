//! # Knot Theory, Artin Braids, and Laurent Skein Invariants
//!
//! Verified 3D spatial knots and links, rotation-minimizing Bishop tubular meshes,
//! zero-allocation Gauss writhe and linking integrals, oriented planar link diagrams with
//! Reidemeister I/II/III ambient isotopy moves, Fox and reduced Burau Alexander polynomials,
//! and state-sum Kauffman/Jones skein invariants.
//!
//! ## Quick Start
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::algebra::BraidWord;
//! use math_explorer::pure_math::knot_theory::{
//!     PlanarDiagram, alexander_polynomial_burau, alexander_polynomial_fox, jones_polynomial,
//! };
//!
//! let right_trefoil = BraidWord::new(2, vec![1, 1, 1]).unwrap();
//! let left_trefoil = right_trefoil.mirror();
//! let diag_r = PlanarDiagram::from_braid(&right_trefoil);
//! let diag_l = PlanarDiagram::from_braid(&left_trefoil);
//!
//! // Fox and reduced Burau Alexander polynomials agree identically: t^-1 - 1 + t
//! assert_eq!(
//!     alexander_polynomial_fox(&diag_r).unwrap(),
//!     alexander_polynomial_burau(&right_trefoil).unwrap()
//! );
//! // Jones polynomial distinguishes chiral mirror trefoils
//! assert_ne!(jones_polynomial(&diag_r).unwrap(), jones_polynomial(&diag_l).unwrap());
//! ```

// [cite:knot_braid_topology]

use crate::pure_math::algebra::{BraidError, LaurentError};
use thiserror::Error;

/// Exact $O(2^c)$ state-sum expansion ceiling for Kauffman bracket and Jones polynomials
/// so interactive `wasm32` frames never stall.
pub const MAX_KAUFFMAN_CROSSINGS: usize = 16;

/// Domain error type for 3D curves, planar knot diagrams, Reidemeister moves, and skein invariants.
#[derive(Error, Clone, Debug, PartialEq)]
pub enum KnotError {
    /// Invalid geometric or algebraic parameter.
    #[error("invalid knot parameter: {reason}")]
    InvalidParameters {
        /// Human-readable validation explanation.
        reason: &'static str,
    },
    /// Underlying Artin braid group error.
    #[error("{0}")]
    Braid(#[from] BraidError),
    /// Underlying Laurent polynomial exponent, span, or integer coefficient error.
    #[error("{0}")]
    Laurent(#[from] LaurentError),
    /// Exact invariant coefficient or loop-power exceeded signed 64-bit integer bounds.
    #[error("knot polynomial coefficient or loop power overflowed signed 64-bit integer bounds")]
    CoefficientOverflow,
    /// Link contained zero curve components.
    #[error("spatial link must contain at least one closed curve component")]
    EmptyCurve,
    /// Curve or tubular ring had too few sample points.
    #[error("insufficient curve samples (required at least {required}, got {actual})")]
    InsufficientSamples {
        /// Minimum required sample count.
        required: usize,
        /// Provided sample count.
        actual: usize,
    },
    /// Pre-allocated destination slice did not match expected dimension.
    #[error("pre-allocated buffer length mismatch (expected {expected}, got {actual})")]
    BufferLengthMismatch {
        /// Required slice length.
        expected: usize,
        /// Actual slice length.
        actual: usize,
    },
    /// Coincident adjacent curve samples yielded a zero-length tangent vector.
    #[error("degenerate zero-length curve tangent at sample index {index}")]
    DegenerateTangent {
        /// Sample index with zero chord length.
        index: usize,
    },
    /// Planar diagram failed 4-valent slot reciprocity or size validation.
    #[error("invalid planar link diagram: {reason}")]
    InvalidDiagram {
        /// Human-readable validation reason.
        reason: &'static str,
    },
    /// Requested crossing index exceeded `PlanarDiagram::crossings.len()`.
    #[error("crossing index {index} is out of bounds for diagram with {count} crossings")]
    CrossingIndexOutOfBounds {
        /// Requested crossing index.
        index: usize,
        /// Total crossings in diagram.
        count: usize,
    },
    /// Requested segment index exceeded `PlanarDiagram::segments.len()`.
    #[error("segment index {index} is out of bounds for diagram with {count} segments")]
    SegmentIndexOutOfBounds {
        /// Requested segment index.
        index: usize,
        /// Total segments in diagram.
        count: usize,
    },
    /// Local Reidemeister move precondition was not satisfied on the selected crossing(s)/arc(s).
    #[error("Reidemeister move {move_kind} is not applicable: {reason}")]
    ReidemeisterNotApplicable {
        /// Move identifier (`"R-I remove"`, `"R-II add"`, `"R-II remove"`, `"R-III slide"`).
        move_kind: &'static str,
        /// Human-readable geometric precondition failure.
        reason: &'static str,
    },
    /// Diagram exceeded [`MAX_KAUFFMAN_CROSSINGS`] for exact $O(2^c)$ state-sum expansion.
    #[error(
        "diagram has {crossings} crossings, exceeding exact Kauffman/Jones state-sum cap of {max_crossings}; polynomial-time writhe, linking, and Alexander invariants remain active"
    )]
    CrossingCapExceeded {
        /// Actual crossing count.
        crossings: usize,
        /// Configured state-sum crossing cap.
        max_crossings: usize,
    },
}

pub mod burau_workspace;
pub mod curves;
pub mod diagram;
mod diagram_lift;
pub mod frames;
pub mod gauss;
pub mod invariants;
pub mod kauffman_jones;
pub mod mesh_tube;
pub mod reidemeister;

pub use burau_workspace::BurauWorkspace;
pub use curves::{
    ClosedLink3D, LissajousKnot, TorusKnot, borromean_rings, figure_eight_knot, hopf_link,
};
pub use diagram::{
    Crossing, CrossingSign, DiagramSegment, Endpoint, MAX_DIAGRAM_CROSSINGS, PlanarDiagram,
};
pub use frames::{BishopFrame, compute_bishop_frames, compute_bishop_frames_in_place};
pub use gauss::{
    gauss_linking_number, gauss_linking_quadrature, gauss_writhe_quadrature,
    pairwise_linking_matrix, pairwise_linking_numbers_in_place,
};
pub use invariants::{
    alexander_polynomial_burau, alexander_polynomial_fox, reduced_burau_matrix,
    try_reduced_burau_matrix,
};
pub use kauffman_jones::{
    format_jones_polynomial, is_chiral_by_jones, jones_polynomial, jones_polynomial_half_power,
    kauffman_bracket, normalized_kauffman_polynomial,
};
pub use mesh_tube::build_link_tubular_mesh;
pub use reidemeister::ReidemeisterTriangle;

crate::theory_verification!(
    module = "knot_braid_topology",
    constants = {},
    test = {
        let right_braid = crate::pure_math::algebra::BraidWord::new(2, vec![1, 1, 1])
            .expect("valid 2-strand trefoil braid");
        let left_braid = right_braid.mirror();
        let trefoil_r = PlanarDiagram::from_braid(&right_braid);
        let trefoil_l = PlanarDiagram::from_braid(&left_braid);
        assert_eq!(trefoil_r.writhe(), 3);
        assert_eq!(trefoil_l.writhe(), -3);
        let alex_fox = alexander_polynomial_fox(&trefoil_r).expect("valid Fox Alexander");
        let alex_burau = alexander_polynomial_burau(&right_braid).expect("valid Burau Alexander");
        assert_eq!(alex_fox, alex_burau);
        assert_eq!(alex_fox.eval_at_one(), 1);
        assert!(alex_fox.is_reciprocal());
        let jones_r = jones_polynomial(&trefoil_r).expect("valid right trefoil Jones");
        let jones_l = jones_polynomial(&trefoil_l).expect("valid left trefoil Jones");
        assert_ne!(jones_r, jones_l);
        assert_eq!(jones_r.invert_variable(), jones_l);
        assert!(is_chiral_by_jones(&trefoil_r).expect("valid chirality check"));
        let hopf = hopf_link(1.5, true, 32).expect("valid Hopf link");
        let lk = pairwise_linking_matrix(&hopf).expect("valid linking matrix");
        assert_eq!(lk[0][1], 1);
        assert_eq!(lk[1][0], 1);
        let borr = borromean_rings(1.8, 0.9, 36).expect("valid Borromean rings");
        let borr_lk = pairwise_linking_matrix(&borr).expect("valid Borromean linking");
        assert_eq!(borr_lk[0][1], 0);
        assert_eq!(borr_lk[0][2], 0);
        assert_eq!(borr_lk[1][2], 0);
    }
);
