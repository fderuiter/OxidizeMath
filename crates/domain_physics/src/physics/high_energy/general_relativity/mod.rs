//! Curved spacetime metrics, analytical Christoffel connections, embeddings, and clock dilation.
//!
//! Bridges `pure_math::tensor::StaticMetric` and `GeodesicSystem` to exact General Relativity
//! solutions:
//! - [`SchwarzschildBlackHole`] (`StaticMetric<4>`): static spherically symmetric black hole in
//!   $(ct, r, \theta, \varphi)$ coordinates, Flamm's paraboloid, effective potentials, photon
//!   sphere, ISCO, and combined gravitational + kinematic time dilation.
//! - [`KerrEquatorialMetric`] (`StaticMetric<4>`): rotating black hole equatorial spacetime with
//!   Lense-Thirring frame dragging and spin-dependent event horizon $r_+$.
//! - [`MorrisThorneWormhole`] (`StaticMetric<3>`): horizon-free traversable wormhole in proper
//!   radial coordinates $(ct, l, \varphi)$ with isometric 3D embedding diagrams.

pub mod geodesic_bundle;
pub mod kerr;
pub mod schwarzschild;
pub mod wormhole;

pub use geodesic_bundle::{
    GeodesicTrajectoryState, MAX_GEODESIC_TRAJECTORIES, NewtonianOrbitState,
    SchwarzschildGeodesicBundle, newtonian_periapsis_radius,
};
pub use kerr::KerrEquatorialMetric;
pub use schwarzschild::{
    GeodesicKind, SchwarzschildBlackHole, combined_clock_rate, critical_impact_parameter,
    effective_potential, flamm_paraboloid_z, photon_sphere_radius,
};
pub use wormhole::MorrisThorneWormhole;

// [cite:general_relativity_geodesics]

pure_math::theory_verification!(
    module = "general_relativity_geodesics",
    constants = {
        PHOTON_SPHERE_FACTOR = 1.5;
        ISCO_FACTOR = 3.0;
    },
    test = {
        let bh = SchwarzschildBlackHole::from_schwarzschild_radius(2.0).unwrap();
        assert!(
            (bh.photon_sphere_radius().unwrap() - PHOTON_SPHERE_FACTOR * 2.0).abs()
                < math_commons::registry::TOLERANCE_FAST
        );
        assert!(
            (bh.isco().unwrap() - ISCO_FACTOR * 2.0).abs() < math_commons::registry::TOLERANCE_FAST
        );
    }
);

#[cfg(test)]
mod tests;
