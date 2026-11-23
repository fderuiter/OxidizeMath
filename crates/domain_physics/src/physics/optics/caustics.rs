//! Hamiltonian ray-bundle caustic tracing and optical-bench caustic envelope extraction.
//!
//! Combines pre-allocated RK4 Hamiltonian ray tracing with ray-tube Jacobian sign-flip detection
//! ([`hamiltonian`]) and optical-bench nephroid/cusp envelope extraction with transverse detector
//! intensity histograms ([`envelope`]).

// [cite:optics]

pub mod envelope;
pub mod hamiltonian;

pub use envelope::{
    CausticEnvelopeBuffer, CausticPoint, coffee_cup_cusp_x, extract_caustic_envelope,
    nephroid_caustic_point, thin_biconvex_paraxial_focal_length,
};
pub use hamiltonian::HamiltonianRayTracer;
