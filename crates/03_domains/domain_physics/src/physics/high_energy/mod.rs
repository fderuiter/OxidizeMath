//! High Energy Physics module.
//!
//! #  High Energy Astrophysics
//!
//! This module provides the mathematical tools to simulate and analyze the most extreme
//! environments in the universe: Black Holes, Relativistic Jets, and High-Energy Radiation.
//!
//! ## Why this matters
//! High energy physics isn't just about big numbers; it's about the intersection of
//! **Gravity** (General Relativity), **Motion** (Special Relativity), and **Quantum Physics**
//! (Radiation). This module allows you to model how these forces interact to produce
//! observable phenomena.
//!
//! <div class="warning">
//!
//! **Mermaid Diagram**
//!
//! ```mermaid
//! graph TD
//!     Source[Astrophysical Source] -->|Gravity| GR[General Relativity]
//!     Source -->|Emission| Rad[Radiation Processes]
//!     Source -->|Motion| SR[Special Relativity]
//!
//!     GR -->|Metric| BH[Black Hole]
//!     Rad -->|Spectrum| Obs[Observer]
//!     SR -->|Frame Boosting| Obs
//!
//!     BH -->|Time Dilation| Obs
//! ```
//! </div>
//!
//! ##  Quick Start: The Relativistic Observer
//!
//! Calculate the total time dilation experienced by an observer orbiting a supermassive
//! black hole. This combines **Gravitational Time Dilation** (General Relativity) with
//! **Kinematic Time Dilation** (Special Relativity).
//!
//! ```rust
//! # use domain_physics::__doctest_facade as math_explorer;
//! use math_explorer::physics::high_energy::{SchwarzschildBlackHole, C, SOLAR_MASS};
//! use math_explorer::physics::high_energy::observer::calculate_lorentz_factor;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // 1. Define a Supermassive Black Hole (1 million solar masses)
//!     let mass = 1_000_000.0 * SOLAR_MASS;
//!     let bh = SchwarzschildBlackHole::new(mass)?;
//!
//!     // 2. Place an observer at 3x the Schwarzschild radius (ISCO - Innermost Stable Circular Orbit)
//!     let rs = bh.schwarzschild_radius()?;
//!     let r_obs = 3.0 * rs;
//!
//!     // 3. Calculate Gravitational Time Dilation
//!     // How much slower does time pass here compared to infinity?
//!     let g_dilation = bh.gravitational_time_dilation(r_obs)?;
//!
//!     // 4. Incorporate Special Relativity (Orbiting at 0.5c)
//!     // The observer is also moving, further slowing time from the perspective of a distant observer.
//!     let velocity = 0.5 * C;
//!     let gamma = calculate_lorentz_factor(velocity)?;
//!     // Kinematic dilation factor is 1/gamma
//!     let sr_dilation = 1.0 / gamma;
//!
//!     // Total dilation is the product of both effects
//!     let total_dilation = g_dilation * sr_dilation;
//!
//!     println!("At 3 Schwarzschild radii:");
//!     println!("- Gravitational Time Dilation: {:.4}", g_dilation);
//!     println!("- Special Relativistic Dilation: {:.4}", sr_dilation);
//!     println!("- Total Time Dilation: {:.4}", total_dilation);
//!     println!("(1 second for observer = {:.4} seconds at infinity)", 1.0 / total_dilation);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Modules
//!
//! - **`general_relativity`**: Black hole metrics and spacetime geometry.
//! - **`observer`**: Frames of reference, Four-Vectors, and Doppler boosting.
//! - **`radiation`**: Synchrotron, Inverse Compton, and Bremsstrahlung processes.
//! - **`fluid_dynamics`**: Relativistic Euler equations and shock waves.
//! - **`statistics`**: Significance calculations (Li & Ma) for signal detection.

/// Relativistic hydrodynamics, sound speeds, and Rankine-Hugoniot shock jump conditions.
pub mod fluid_dynamics;
/// Schwarzschild, Kerr equatorial, and Morris-Thorne spacetime metrics and geodesics.
pub mod general_relativity;
/// Special-relativistic Lorentz boosts, four-vectors, and Doppler beaming factors.
pub mod observer;
/// Synchrotron, inverse-Compton, and thermal bremsstrahlung radiative transfer models.
pub mod radiation;
/// Li & Ma Poisson likelihood-ratio significance for gamma-ray source detection.
pub mod statistics;

// Re-export constants to match original API
pub use math_commons::constants::{C, G, SIGMA_T, SOLAR_MASS};

// Re-export SchwarzschildBlackHole, lensing deflection tables, geodesic bundle, and jet beaming helpers
pub use general_relativity::{
    BundleRay, DEFAULT_DEFLECTION_BINS, DeflectionSample, DeflectionTable,
    EINSTEIN_RING_IMPACT_FACTOR, GeodesicKind, GeodesicTrajectoryState, KerrEquatorialMetric,
    KerrRadialOrbitParams, LensedRayHit, LensingMetricSpec, LensingRegime, MAX_BUNDLE_RAYS,
    MAX_GEODESIC_TRAJECTORIES, MAX_TRAJECTORY_SAMPLES, MIN_REGULAR_LAPSE, MIN_REGULAR_THROAT,
    MorrisThorneWormhole, NewtonianOrbitState, RadialAtlasSummary, SchwarzschildBlackHole,
    SchwarzschildGeodesicBundle, TrajectorySample, WorldlineLaunchSpec, build_lapse_static_metric,
    build_wormhole_static_metric, combined_clock_rate, compute_4d_clock_factors,
    critical_impact_parameter, effective_potential, estimate_compiled_lapse_horizon,
    eval_custom_lapse_lensing_cell, eval_kerr_lensing_cell, eval_wormhole_lensing_cell,
    flamm_paraboloid_z, initial_diagonal_4d_state, initial_kerr_4d_state,
    integrate_custom_lapse_worldline_into, integrate_custom_wormhole_worldline_into,
    integrate_kerr_worldline_into, integrate_photon_fan_bundle_into,
    integrate_schwarzschild_worldline_into, integrate_wormhole_worldline_into, kerr_bpt_isco,
    kerr_prograde_critical_b, kerr_radial_acc_and_dphi, kerr_radial_rk4_step, kerr_veff,
    newtonian_periapsis_radius, photon_sphere_radius, planar_velocity_beta, refine_halt_4d,
    wormhole_initial_vel,
};
pub use observer::{
    FourVector, apparent_superluminal_beta, calculate_lorentz_factor_from_beta,
    relativistic_doppler_factor,
};
pub use radiation::{
    JetKnotConfig, JetKnotObservables, accretion_ring_beaming_factor,
    accretion_ring_flux_asymmetry, beamed_synchrotron_flux_factor, evaluate_jet_knot_observables,
    inverse_compton_spectral_index, synchrotron_power, twin_jet_flux_asymmetry,
};

// [cite:dwarf_galaxy_empirical_dependencies]
// [cite:general_relativity_geodesics]

use pure_math::theory_verification;

theory_verification!(
    module = "high_energy",
    constants = {
        CRITICAL_IMPACT_RATIO = 2.598_076_211_353_316;
        EINSTEIN_IMPACT_RATIO_SQ = 6.75;
        ISCO_SHELL_BETA = 0.5;
        RELATIVISTIC_SHOCK_COMPRESSION = 7.0;
    },
    test = {
        use approx::assert_relative_eq;

        let bh = SchwarzschildBlackHole::from_schwarzschild_radius(1.0).unwrap();
        assert_relative_eq!(
            bh.critical_impact_parameter().unwrap(),
            CRITICAL_IMPACT_RATIO,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            EINSTEIN_RING_IMPACT_FACTOR * EINSTEIN_RING_IMPACT_FACTOR,
            EINSTEIN_IMPACT_RATIO_SQ,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            bh.circular_shell_beta(3.0).unwrap(),
            ISCO_SHELL_BETA,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            fluid_dynamics::shock_compression_ratio(4.0 / 3.0).unwrap(),
            RELATIVISTIC_SHOCK_COMPRESSION,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }
);
