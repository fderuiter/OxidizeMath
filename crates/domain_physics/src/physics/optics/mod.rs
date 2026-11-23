//! # Wave Optics, Angular-Spectrum Diffraction & Phase Holography
//!
//! Verified scalar wave-optics models spanning radiometric/LiDAR sensing (`sensing`),
//! zero-allocation 2D Helmholtz angular-spectrum diffraction and Zernike/aperture pupil
//! synthesis (`diffraction`), and iterative Gerchberg-Saxton SLM phase holography (`holography`).

// [cite:graph_parameters_rust]

pub mod diffraction;
pub mod holography;
pub mod sensing;

use pure_math::theory_verification;

theory_verification!(
    module = "optics",
    constants = {
        APERTURE_RADIUS_M = 1.0e-3;
        WAVELENGTH_M = 5.0e-7;
        PROPAGATION_Z_M = 1.0;
        EXPECTED_FRESNEL_NUMBER = 2.0;
        GRATING_PERIOD_M = 1.0e-4;
        EXPECTED_TALBOT_LENGTH_M = 0.04;
    },
    test = {
        use approx::assert_relative_eq;
        let nf = crate::physics::optics::diffraction::fresnel_number(
            APERTURE_RADIUS_M,
            WAVELENGTH_M,
            PROPAGATION_Z_M,
        );
        let zt = crate::physics::optics::diffraction::talbot_length(GRATING_PERIOD_M, WAVELENGTH_M);
        assert_relative_eq!(
            nf,
            EXPECTED_FRESNEL_NUMBER,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            zt,
            EXPECTED_TALBOT_LENGTH_M,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }
);

#[cfg(test)]
mod tests;
