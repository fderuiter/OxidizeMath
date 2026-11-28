//! # Millimeter-Wave FMCW Radar Signal Processing
//!
//! High-precision RF signal processing, spatial beamforming, super-resolution spectral estimation,
//! surface reconstruction, and Kalman state tracking for mmWave FMCW radar sensors (e.g., TI IWR6843).
//!
//! ## Core Pipeline
//!
//! - **[`physics`]**: FMCW range-beat reciprocity, range resolution $\Delta R = c / (2B)$, Doppler velocity, and sub-millimeter phase displacement.
//! - **[`czt`]**: Bluestein FFT Chirp Z-Transform (`chirp_z_transform`) for zoom spectral analysis.
//! - **[`geometry`]**: Sensor-frame Angle-FFT spherical-to-Cartesian projection, AoA, and rigid-body transforms.
//! - **[`mimo`]**: Uniform linear array steering and sidelobe tapering (`Beamformer`).
//! - **[`super_resolution`]**: Eigenspace `MusicEstimator` for sub-Rayleigh range resolution.
//! - **[`clutter`]**: CA-CFAR noise floor estimation and sliding-window `EllipticalFilter` static clutter rejection.
//! - **[`phase`]**: Continuous $2\pi$ phase unwrapping (`PhaseUnwrapper`) for sub-millimeter motion tracking.
//! - **[`surface`]**: SVD least-squares `BiQuadraticSurface` fitting and weighted average height (`weighted_average_height`).
//! - **[`tracking`]**: 1D `ConstantVelocityModel` discrete Kalman `TrackingFilter` backed by `pure_math`.

/// CA-CFAR background noise estimation and sliding-window static clutter rejection.
pub mod clutter;
/// Bluestein fast-convolution Chirp Z-Transform (CZT) zoom spectral estimator.
pub mod czt;
/// Sensor-frame Angle-FFT projection, angle-of-arrival, and rigid-body coordinate transforms.
pub mod geometry;
/// Uniform linear array MIMO digital beamforming and tapering weights.
pub mod mimo;
/// Continuous phase extraction and $2\pi$ unwrapping for displacement tracking.
pub mod phase;
/// Fundamental FMCW radar range, beat-frequency, wavelength, and Doppler equations.
pub mod physics;
/// Sub-resolution eigenspace MUSIC range estimator.
pub mod super_resolution;
/// Bi-quadratic polynomial surface SVD fitting and ROI mean height extraction.
pub mod surface;
/// 1D constant-velocity Kalman tracking filter.
pub mod tracking;

pub use clutter::EllipticalFilter;
pub use czt::SpatialCztConfig;
pub use czt::chirp_z_transform;
pub use geometry::{AngleFftConfig, SensorToPatientTransform, SphericalPoint};
pub use math_commons::constants::C;
pub use mimo::Beamformer;
pub use phase::PhaseUnwrapper;
pub use physics::FmcwConfig;
pub use super_resolution::MusicEstimator;
pub use surface::BiQuadraticSurface;
pub use tracking::{ConstantVelocityModel, TrackingFilter};

// Re-export the core KalmanModel trait from pure_math so callers can implement custom motion models.
pub use pure_math::pure_math::analysis::kalman::KalmanModel;

// [cite:mmwave_radiotherapy_setup]

use pure_math::theory_verification;

theory_verification!(
    module = "radar",
    constants = {},
    test = {
        use approx::assert_relative_eq;
        use std::f64::consts::PI;

        let cfg = FmcwConfig::iwr6843_default();
        let slope = cfg.bandwidth / cfg.chirp_time;
        let target_range = 1.25;

        // Physical Invariant 1: Exact reciprocity between forward beat frequency and inverse range mapping.
        let fb = FmcwConfig::beat_frequency(slope, target_range);
        assert_relative_eq!(
            cfg.range_from_beat_frequency(fb),
            target_range,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );

        // Physical Invariant 2: Two-way round-trip phase shift of pi corresponds to lambda / 4 displacement.
        assert_relative_eq!(
            cfg.displacement_from_phase(PI),
            0.25 * cfg.wavelength(),
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }
);
