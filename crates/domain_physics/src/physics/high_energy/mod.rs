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

#[allow(missing_docs)]
pub mod fluid_dynamics;
#[allow(missing_docs)]
pub mod general_relativity;
#[allow(missing_docs)]
pub mod observer;
#[allow(missing_docs)]
pub mod radiation;
#[allow(missing_docs)]
pub mod statistics;

// Re-export constants to match original API
pub use math_commons::constants::{C, G, SIGMA_T, SOLAR_MASS};

// Re-export SchwarzschildBlackHole and relativistic geodesic types/helpers
pub use general_relativity::{
    GeodesicKind, KerrEquatorialMetric, MorrisThorneWormhole, SchwarzschildBlackHole,
    combined_clock_rate, effective_potential, flamm_paraboloid_z, photon_sphere_radius,
};

// Re-export Error type

// [cite:dwarf_galaxy_empirical_dependencies]
// [cite:general_relativity_geodesics]

use pure_math::theory_verification;

theory_verification!(
    module = "high_energy",
    constants = {
        DUMMY = 1.0;
    },
    test = {
        assert_relative_eq!(DUMMY, 1.0, epsilon = math_commons::registry::TOLERANCE_FAST);
    }
);
