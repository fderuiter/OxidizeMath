//! Geometric ray tracing, polarized Fresnel refraction, Hamiltonian & nephroid caustics, Fraunhofer diffraction, and optical sensing.
//!
//! All hot-path ray-fan (`trace_ray_fan`), Hamiltonian ray-bundle (`HamiltonianRayTracer::trace_in_place`),
//! caustic envelope (`extract_caustic_envelope`), and far-field diffraction (`sample_fraunhofer_profile`)
//! kernels operate on pre-allocated workspace buffers with zero heap allocations.

// [cite:optics]

pub mod caustics;
pub mod diffraction;
pub mod ray_tracing;
pub mod refraction;
pub mod sensing;

pub use caustics::{
    CausticEnvelopeBuffer, CausticPoint, HamiltonianRayTracer, coffee_cup_cusp_x,
    extract_caustic_envelope, nephroid_caustic_point, thin_biconvex_paraxial_focal_length,
};
pub use diffraction::{
    AIRY_FIRST_ZERO_J1, AIRY_SECOND_ZERO_J1, AIRY_THIRD_ZERO_J1, CriticalPointKind,
    CriticalPointMarker, DiffractionAperture, FraunhoferConfig, FraunhoferProfileBuffer,
    airy_normalized_intensity, fraunhofer_intensity_at_angle, sample_fraunhofer_profile,
};
pub use ray_tracing::{
    EikonalRaySystem, GrinMedium, OpticalBenchScene, PolarizationMode, RayFanConfig,
    RayFanWorkspace, TracedRayMeta, trace_ray_fan,
};
pub use refraction::{
    CauchyDispersion, FresnelCoefficients, GlassMaterial, OpticalMediumConfig, OpticsPreset,
    SnellOutcome, brewster_angle, cauchy_refractive_index, critical_angle, fresnel_polarized,
    intersect_ray_circle_arc, intersect_ray_segment, normalize_2d, reflect_vector_2d,
    snell_refract_angle, snell_refract_vector, snell_refracted_angle, snell_vector_2d,
};
pub use sensing::{
    DetectorConfig, DetectorReadout, lambertian_intensity, lws_range_resolution,
    sample_detector_line_in_place,
};

use pure_math::theory_verification;

theory_verification!(
    module = "optics",
    constants = {
        N_AIR = 1.0;
        N_GLASS = 1.5;
        N_VESELAGO = -1.0;
        BREWSTER_AIR_BK7_RAD = 0.987_923_136_003_447_8;
    },
    test = {
        use approx::assert_relative_eq;

        let computed_brewster = brewster_angle(1.0, 1.5168);
        assert_relative_eq!(
            computed_brewster,
            BREWSTER_AIR_BK7_RAD,
            epsilon = math_commons::registry::TOLERANCE_HIGH
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod verified_alloc_tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn optics_hot_loops_allocate_zero_bytes_over_100_steps() {
        let mut ray_ws = RayFanWorkspace::new(96, 160);
        let mut caustic_buf = CausticEnvelopeBuffer::new(256, 64, -2.0, 2.0);
        let mut diff_buf = FraunhoferProfileBuffer::new(256, 32);

        let ray_cfg = RayFanConfig {
            scene: OpticalBenchScene::ContinuousGrin {
                medium: GrinMedium::Luneburg { radius: 1.0 },
                step_tau: 0.025,
                steps: 120,
            },
            ray_count: 31,
            beam_half_width: 0.85,
            source_x: -1.4,
            source_y: 0.0,
            launch_angle_rad: 0.0,
            wavelength_nm: 550.0,
            polychromatic: false,
            polarization: PolarizationMode::Unpolarized,
            bench_half_extent: 2.0,
        };
        let cup_cfg = RayFanConfig {
            scene: OpticalBenchScene::CoffeeCupMirror { radius: 1.5 },
            ..ray_cfg
        };
        let diff_cfg = FraunhoferConfig {
            aperture: DiffractionAperture::CircularAiry { diameter_um: 12.0 },
            wavelength_nm: 532.0,
            screen_distance_m: 1.0,
            half_angle_rad: 0.12,
        };

        let (_, metrics) = VerifiedEngine::run_verified(|| {
            for step in 0..100usize {
                let active_cfg = if step % 2 == 0 { &ray_cfg } else { &cup_cfg };
                trace_ray_fan(active_cfg, &mut ray_ws);
                extract_caustic_envelope(&ray_ws, 1.0, 2.5, &mut caustic_buf);
                sample_fraunhofer_profile(&diff_cfg, 201, &mut diff_buf);
            }
        });
        assert!(metrics.function_calls >= 300);
        assert!(metrics.arithmetic_ops > 1_000);
        assert!(!diff_buf.markers().is_empty());
    }
}
