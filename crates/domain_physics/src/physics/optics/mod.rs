//! # Physical, Geometric & Wave Optics
//!
//! Verified physical and geometric optics kernels spanning graded/negative-index refractive media
//! (`refraction`), Hamiltonian ray caustics (`caustics`), zero-allocation 2D Helmholtz
//! angular-spectrum wave diffraction and Zernike/aperture pupil synthesis (`diffraction`),
//! iterative Gerchberg-Saxton SLM phase holography (`holography`), and calibrated radiometric/LiDAR
//! detector models (`sensing`).

// [cite:optics]

pub mod caustics;
pub mod diffraction;
pub mod holography;
pub mod refraction;
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
        N_AIR = 1.0;
        N_GLASS = 1.5;
        N_VESELAGO = -1.0;
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

        let theta_inc = std::f64::consts::FRAC_PI_6;
        let theta_glass = refraction::snell_refracted_angle(N_AIR, N_GLASS, theta_inc)
            .expect("dielectric transmission");
        let expected_glass = (theta_inc.sin() / N_GLASS).asin();
        assert_relative_eq!(
            theta_glass,
            expected_glass,
            epsilon = math_commons::registry::TOLERANCE_STANDARD
        );

        let theta_veselago = refraction::snell_refracted_angle(N_AIR, N_VESELAGO, theta_inc)
            .expect("Veselago negative refraction");
        assert_relative_eq!(
            theta_veselago,
            -theta_inc,
            epsilon = math_commons::registry::TOLERANCE_STANDARD
        );

        let luneburg = refraction::OpticalMediumConfig::default();
        let [cx, cy] = luneburg.lens_center();
        assert_relative_eq!(
            luneburg.refractive_index_at(cx, cy),
            std::f64::consts::SQRT_2,
            epsilon = math_commons::registry::TOLERANCE_STANDARD
        );

        let lamb = sensing::lambertian_intensity(
            2.0 * std::f64::consts::PI,
            1.0,
            std::f64::consts::FRAC_PI_3,
            1.0,
        );
        assert_relative_eq!(lamb, 1.0, epsilon = math_commons::registry::TOLERANCE_FAST);
    }
);

#[cfg(test)]
mod tests;
