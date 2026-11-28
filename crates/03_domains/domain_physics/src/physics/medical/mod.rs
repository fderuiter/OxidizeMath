//! # Medical Physics: Treatment Planning
//!
//! This module implements core algorithms for Radiation Therapy Treatment Planning (RTTP).
//! It covers four main domains:
//!
//! 1.  **Calibration**: Converting CT imaging data (Hounsfield Units) into physical density for dose calculation.
//! 2.  **Dose Calculation**: Modeling how radiation interacts with matter using convolution/superposition principles.
//! 3.  **Inverse Planning**: Optimizing beam intensities (IMRT/VMAT) to maximize tumor control while sparing healthy tissue.
//! 4.  **Evaluation**: Quantifying plan quality using Dose-Volume Histograms (DVH) and biological modeling (TCP).
//! 5.  **Motion Management**:
//!     - Radar-based respiratory gating (`radar_gating`).
//!     - Optical surface monitoring (`optical_motion`).
//!
//! ## Context
//!
//! In radiation therapy, the goal is to deliver a lethal dose of radiation to a tumor (PTV - Planning Target Volume)
//! while minimizing the dose to surrounding Organs At Risk (OARs). This is an inverse problem where we solve for
//! the optimal beam fluence map that results in the desired 3D dose distribution.
//!
//! ## Units
//!
//! - **Dose**: Gray (Gy), where 1 Gy = 1 J/kg.
//! - **Distance**: Centimeters (cm).
//! - **Density**: g/cm³.
//! - **Hounsfield Units (HU)**: Dimensionless scale where Water = 0, Air = -1000.

/// Linear accelerator waveguide, bremsstrahlung target, and beam-transport physics.
pub mod accelerator;
pub mod calibration;
/// TERMA calculation and point-spread convolution dose kernels.
pub mod dose;
pub mod evaluation;
/// CT Hounsfield-unit conversion and medical image reconstruction helpers.
pub mod imaging;
/// Stereoscopic optical surface-guided radiotherapy (SGRT) respiratory motion tracking.
pub mod optical_motion;
pub mod optimization;
pub mod radar_gating;
/// Respiratory surrogate filtering, phase extraction, and spectral estimation.
pub mod signal;
pub mod thermodynamics;

pub use radar_gating::GatingError;

// [cite:medical]

use pure_math::theory_verification;

theory_verification!(
    module = "medical",
    constants = {},
    test = {
        use radar_gating::{BeamState, GatingLogic};

        // Clinical Safety Invariant: Latency-compensated predicted amplitude A + v * tau_lat
        // triggers BeamHold above threshold + hysteresis and latches SafetyInterlock on NaN.
        let mut gate = GatingLogic::new(5.0, 1.0, 0.060);
        assert_eq!(gate.arm().unwrap(), BeamState::Armed);
        assert_eq!(gate.evaluate(3.5, 0.0), BeamState::BeamActive);
        // Predicted = 5.5 + 10.0 * 0.060 = 6.1 > 5.0 + 1.0 -> BeamHold
        assert_eq!(gate.evaluate(5.5, 10.0), BeamState::BeamHold);
        assert_eq!(gate.evaluate(f64::NAN, 0.0), BeamState::SafetyInterlock);
    }
);
