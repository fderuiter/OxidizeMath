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
//! 6.  **Bioelectric Lead Integration**:
//!     - Movable 3-lead extracellular dipole projection (`bioelectric`).
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

#[allow(missing_docs)]
pub mod accelerator;
pub mod bioelectric;
mod bioelectric_presets;
pub mod calibration;
#[allow(missing_docs)]
pub mod dose;
pub mod evaluation;
#[allow(missing_docs)]
pub mod imaging;
#[allow(missing_docs)]
pub mod optical_motion;
pub mod optimization;
pub mod radar_gating;
#[allow(missing_docs)]
pub mod signal;
pub mod thermodynamics;

// [cite:favorite_child]

use pure_math::theory_verification;

theory_verification!(
    module = "medical",
    constants = {
        DUMMY = 1.0;
    },
    test = {
        use approx::assert_relative_eq;
        assert_relative_eq!(DUMMY, 1.0, epsilon = math_commons::registry::TOLERANCE_FAST);
    }
);
