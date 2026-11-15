//! # Radar-based Respiratory Gating
//!
//! A high-precision pipeline for non-contact patient monitoring using mmWave radar (e.g., TI IWR6843).
//! This system bridges the gap between raw RF signals and clinical beam control.
//!
//! ## The Pipeline
//!
//! The system processes data in stages, transforming raw ADC samples into a "Gate/No-Gate" decision.
//!
//! ```mermaid
//! graph TD
//!     subgraph "Signal Processing"
//!     Raw[Raw ADC Data] --> Physics[Physics (FMCW)]
//!     Physics --> CZT[CZT (Chirp Z-Transform)]
//!     end
//!
//!     subgraph "Spatial Analysis"
//!     CZT --> Geometry[Geometry (Sensor -> Patient)]
//!     Geometry --> Surface[Surface Fitting (Bi-Quadratic)]
//!     end
//!
//!     subgraph "Decision Engine"
//!     Surface --> Tracking[Tracking (Kalman Filter)]
//!     Tracking --> Gating[Gating Logic (Schmidt Trigger)]
//!     Gating --> LINAC((LINAC Beam Control))
//!     end
//!
//!     style LINAC fill:#f96,stroke:#333,stroke-width:4px
//! ```
//!
//! ## Advanced Processing ("Fonzi Stack")
//!
//! For challenging clinical scenarios (e.g., shallow breathing, heavy clutter), the **Fonzi Stack**
//! activates additional algorithms to enhance signal fidelity.
//!
//! *   **[MIMO Beamforming](mimo)**: "The Digital Lens" - Spatially filters signals to focus on specific organs.
//! *   **[MUSIC Estimator](super_resolution)**: Sub-resolution target identification using eigen-decomposition.
//! *   **[Elliptical Filter](clutter)**: Removes static clutter (treatment couch, immobilization devices).
//! *   **[Phase Unwrapping](phase)**: Detects sub-millimeter chest wall displacements.
//!
//! # Core Modules

pub mod clutter;
pub mod czt;
pub mod gating;
pub mod geometry;
pub mod mimo;
pub mod phase;
pub mod physics;
pub mod super_resolution;
pub mod surface;
pub mod tracking;

pub use clutter::EllipticalFilter;
pub use czt::SpatialCztConfig;
pub use czt::chirp_z_transform;
pub use gating::{BeamState, GatingError, GatingLogic};
pub use geometry::{AngleFftConfig, SensorToPatientTransform, SphericalPoint};
pub use math_commons::constants::C;
pub use mimo::Beamformer;
pub use phase::PhaseUnwrapper;
pub use physics::FmcwConfig;
pub use super_resolution::MusicEstimator;
pub use surface::BiQuadraticSurface;
pub use tracking::{ConstantVelocityModel, TrackingFilter};

// Re-export the core KalmanModel so users can implement their own models if needed
pub use domain_applied::applied::algorithms::kalman::KalmanModel;

// [cite:clinical_trials_statistics]
