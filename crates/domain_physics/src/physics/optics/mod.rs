//! Physical and geometric optics: refractive media, Hamiltonian ray caustics, and sensor models.

pub mod caustics;
pub mod refraction;
pub mod sensing;

// [cite:optics]

use pure_math::theory_verification;

theory_verification!(
    module = "optics",
    constants = {
        N_AIR = 1.0;
        N_GLASS = 1.5;
        N_VESELAGO = -1.0;
    },
    test = {
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
