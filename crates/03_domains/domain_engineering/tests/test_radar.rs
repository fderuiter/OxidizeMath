//! Integration tests for radar physics, geometry, surface fitting, Kalman tracking, and checked hardware APIs.
use approx::assert_relative_eq;
use domain_engineering::engineering::{
    CuttingSpeed, MaterialConstant, ToolExponent, TotalBits,
    radar::{
        C,
        geometry::{AngleFftConfig, SphericalPoint},
        mimo::Beamformer,
        physics::FmcwConfig,
        surface::{BiQuadraticSurface, weighted_average_height},
        tracking::{ConstantVelocityModel, TrackingFilter},
    },
};
use domain_engineering::error::{HardwareError, RadarError};
use nalgebra::Point3;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_radar_physics() {
    let config = FmcwConfig::iwr6843_default();

    let resolution = config.range_resolution();
    assert_relative_eq!(resolution, 0.037474, epsilon = 1e-4);

    let lambda = C / 60.0e9;
    let tc = 50.0e-6;
    let expected_dphi = (4.0 * std::f64::consts::PI * tc) / lambda;

    let v = config.velocity_from_phase(expected_dphi);
    assert_relative_eq!(v, 1.0, epsilon = math_commons::registry::TOLERANCE_FAST);
}

#[test]
#[verified_engine::verified]
fn test_coordinate_transformation() {
    let fft_config = AngleFftConfig {
        n_fft_azimuth: 64,
        n_fft_elevation: 32,
    };

    let p0 = SphericalPoint {
        range: 10.0,
        azimuth_index: 0,
        elevation_index: 0,
    };
    let c0 = fft_config.spherical_to_cartesian(&p0);
    assert_relative_eq!(c0.x, 0.0);
    assert_relative_eq!(c0.y, 10.0);
    assert_relative_eq!(c0.z, 0.0);

    let p1 = SphericalPoint {
        range: 10.0,
        azimuth_index: 16,
        elevation_index: 0,
    };
    let c1 = fft_config.spherical_to_cartesian(&p1);
    assert_relative_eq!(c1.x, 5.0);
    assert_relative_eq!(c1.y, (75.0_f64).sqrt());
    assert_relative_eq!(c1.z, 0.0);
}

#[test]
#[verified_engine::verified]
fn test_surface_fitting_and_weighted_average_height_zero_alloc() {
    let mut points = Vec::new();
    for x in -2..=2 {
        for y in -2..=2 {
            let x = x as f64;
            let y = y as f64;
            let z = 1.0 + 0.1 * x.powi(2) + 0.1 * y.powi(2);
            points.push(Point3::new(x, y, z));
        }
    }

    let surface = BiQuadraticSurface::fit(&points).expect("Fitting failed");
    let coeffs = &surface.coefficients;
    assert_relative_eq!(
        coeffs[0],
        1.0,
        epsilon = math_commons::registry::TOLERANCE_FAST
    );
    assert_relative_eq!(
        coeffs[1],
        0.0,
        epsilon = math_commons::registry::TOLERANCE_FAST
    );
    assert_relative_eq!(
        coeffs[2],
        0.0,
        epsilon = math_commons::registry::TOLERANCE_FAST
    );
    assert_relative_eq!(
        coeffs[3],
        0.0,
        epsilon = math_commons::registry::TOLERANCE_FAST
    );
    assert_relative_eq!(
        coeffs[4],
        0.1,
        epsilon = math_commons::registry::TOLERANCE_FAST
    );
    assert_relative_eq!(
        coeffs[5],
        0.1,
        epsilon = math_commons::registry::TOLERANCE_FAST
    );

    let vertices = [
        Point3::new(-1.0, -1.0, 1.2),
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(1.0, 1.0, 1.4),
        Point3::new(2.0, -1.0, 1.6),
    ];
    let ((wah_radar, wah_empty), metrics) = VerifiedEngine::run_verified(|| {
        (
            weighted_average_height(&vertices),
            weighted_average_height(&[]),
        )
    });
    assert!(metrics.function_calls >= 2);
    assert_eq!(wah_empty, 0.0);
    assert_relative_eq!(
        wah_radar,
        1.3,
        epsilon = math_commons::registry::TOLERANCE_FAST
    );
}

#[test]
#[verified_engine::verified]
fn test_kalman_tracking_and_checked_radar_hardware_apis() {
    let model = ConstantVelocityModel::new(1e-4, 1e-2);
    let mut filter = TrackingFilter::new(0.0, 0.1, model.clone());

    for i in 0..100 {
        filter.predict();
        let true_pos = i as f64 * 0.1;
        filter.update(true_pos);
    }
    assert_relative_eq!(filter.velocity(), 1.0, epsilon = 0.1);

    assert!(matches!(
        TrackingFilter::try_new(1.0, 0.0, model.clone()),
        Err(RadarError::InvalidTrackingParameter {
            parameter: "dt",
            ..
        })
    ));
    assert!(TrackingFilter::try_new(1.0, 0.05, model).is_ok());

    let mut beamformer = Beamformer::new_ula(4, 0.01, 0.02);
    assert_eq!(
        beamformer.set_weights(&[1.0, 1.0]),
        Err(RadarError::AntennaWeightCountMismatch {
            expected: 4,
            actual: 2,
        })
    );

    assert!(matches!(
        CuttingSpeed::new(0.0),
        Err(HardwareError::InvalidParameter { .. })
    ));
    assert!(matches!(
        MaterialConstant::new(-1.0),
        Err(HardwareError::InvalidParameter { .. })
    ));
    assert!(matches!(
        ToolExponent::new(0.0),
        Err(HardwareError::InvalidParameter { .. })
    ));
    assert!(matches!(
        TotalBits::new(0),
        Err(HardwareError::InvalidParameter { .. })
    ));
}
