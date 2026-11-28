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

pub mod custom_metric;
pub mod geodesic_bundle;
pub mod kerr;
pub mod kerr_radial;
pub mod lensing;
pub mod schwarzschild;
pub mod worldline;
pub mod worldline_integrators;
pub mod wormhole;

pub use custom_metric::{
    MIN_REGULAR_LAPSE, MIN_REGULAR_THROAT, build_lapse_static_metric, build_wormhole_static_metric,
    estimate_compiled_lapse_horizon, eval_custom_lapse_lensing_cell, eval_wormhole_lensing_cell,
};
pub use geodesic_bundle::{
    GeodesicTrajectoryState, MAX_GEODESIC_TRAJECTORIES, NewtonianOrbitState,
    SchwarzschildGeodesicBundle, newtonian_periapsis_radius,
};
pub use kerr::KerrEquatorialMetric;
pub use kerr_radial::{
    KerrRadialOrbitParams, LensingRegime, RadialAtlasSummary, eval_kerr_lensing_cell,
    kerr_bpt_isco, kerr_prograde_critical_b, kerr_radial_acc_and_dphi, kerr_radial_rk4_step,
    kerr_veff,
};
pub use lensing::{
    DEFAULT_DEFLECTION_BINS, DeflectionSample, DeflectionTable, EINSTEIN_RING_IMPACT_FACTOR,
    LensedRayHit, LensingMetricSpec,
};
pub use schwarzschild::{
    GeodesicKind, SchwarzschildBlackHole, combined_clock_rate, critical_impact_parameter,
    effective_potential, flamm_paraboloid_z, photon_sphere_radius,
};
pub use worldline::{
    BundleRay, MAX_BUNDLE_RAYS, MAX_TRAJECTORY_SAMPLES, TrajectorySample, compute_4d_clock_factors,
    initial_diagonal_4d_state, initial_kerr_4d_state, planar_velocity_beta, wormhole_initial_vel,
};
pub use worldline_integrators::{
    WorldlineLaunchSpec, integrate_custom_lapse_worldline_into,
    integrate_custom_wormhole_worldline_into, integrate_kerr_worldline_into,
    integrate_photon_fan_bundle_into, integrate_schwarzschild_worldline_into,
    integrate_wormhole_worldline_into, refine_halt_4d,
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
