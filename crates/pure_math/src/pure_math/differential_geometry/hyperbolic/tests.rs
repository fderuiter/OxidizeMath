#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use approx::assert_relative_eq;
use diagnostics::{Diagnostic, Severity};
use num_complex::Complex64;
use std::f64::consts::PI;

#[test]
fn test_su11_closure_inverse_and_renormalization() {
    let m1 = Su11Matrix::hyperbolic_pan(Complex64::new(0.42, -0.31));
    let m2 = Su11Matrix::rotation(1.17);
    let m3 = Su11Matrix::translation_to_origin(Complex64::new(-0.25, 0.52));

    let composed = m1.compose(&m2).compose(&m3);
    assert_relative_eq!(composed.determinant(), 1.0, epsilon = 1e-12);

    let inv = composed.inverse();
    let id = composed.compose(&inv);
    assert_relative_eq!(id.a.re, 1.0, epsilon = 1e-12);
    assert_relative_eq!(id.a.im, 0.0, epsilon = 1e-12);
    assert_relative_eq!(id.b.norm(), 0.0, epsilon = 1e-12);

    let p = Complex64::new(0.33, -0.44);
    let q = Complex64::new(-0.28, 0.19);
    assert_relative_eq!(
        poincare_distance(p, q),
        poincare_distance(composed.apply(p), composed.apply(q)),
        epsilon = 1e-12
    );

    let mut drifted = Su11Matrix {
        a: composed.a * 1.05,
        b: composed.b * 1.05,
    };
    drifted.renormalize();
    assert_relative_eq!(drifted.determinant(), 1.0, epsilon = 1e-14);
}

#[test]
fn test_poincare_klein_and_cayley_round_trips() {
    let samples = [
        Complex64::new(0.0, 0.0),
        Complex64::new(0.45, -0.35),
        Complex64::new(-0.72, 0.51),
        Complex64::new(0.0, 0.91),
    ];
    for z in samples {
        let w = poincare_to_klein(z);
        assert!(w.norm() <= POINCARE_DISK_RADIUS_MAX);
        let z_back = klein_to_poincare(w);
        assert_relative_eq!(z_back.re, z.re, epsilon = 1e-12);
        assert_relative_eq!(z_back.im, z.im, epsilon = 1e-12);
        assert_relative_eq!(
            poincare_distance(samples[1], z),
            klein_distance(poincare_to_klein(samples[1]), w),
            epsilon = 1e-12
        );

        let tau = poincare_to_upper_half_plane(z);
        assert!(tau.im > 0.0);
        let z_cayley = upper_half_plane_to_poincare(tau).expect("Im(tau) > 0");
        assert_relative_eq!(z_cayley.re, z.re, epsilon = 1e-12);
        assert_relative_eq!(z_cayley.im, z.im, epsilon = 1e-12);
    }

    let clamped = clamp_to_disk(Complex64::new(10.0, -5.0));
    assert_relative_eq!(clamped.norm(), POINCARE_DISK_RADIUS_MAX, epsilon = 1e-15);
    assert!(upper_half_plane_to_poincare(Complex64::new(1.0, -0.2)).is_err());
}

#[test]
fn test_gaussian_curvature_and_gauss_bonnet_7_3_and_5_4() {
    assert_relative_eq!(gaussian_curvature(), -1.0, epsilon = 1e-15);
    for z in [
        Complex64::new(0.0, 0.0),
        Complex64::new(0.3, -0.25),
        Complex64::new(-0.55, 0.42),
    ] {
        assert_relative_eq!(intrinsic_gaussian_curvature(z), -1.0, epsilon = 1e-6);
    }

    // {7, 3} central sector triangle: angles pi/7 (at two vertices) and 2*pi/7 (at center)?
    // Wait: vertex interior angle of {7, 3} is 2*pi/3, split by central rays into pi/3 + pi/3!
    // Central angle is 2*pi/7, base angles are pi/3 and pi/3 -> sum = 2*pi/7 + 2*pi/3 = 20*pi/21,
    // defect = pi/21, and 7 sectors sum to heptagon area = 7 * (pi/21) = pi/3!
    let mut buf_7_3 = HyperbolicTilingBuffer::with_capacity(128);
    let count_7_3 = buf_7_3.configure(7, 3, 2).expect("{7, 3} valid");
    assert!(count_7_3 > 7);
    let central_7_3 = &buf_7_3.polygons()[0];
    let gb_sector_7_3 = gauss_bonnet_triangle(
        central_7_3.center,
        central_7_3.vertices[0],
        central_7_3.vertices[1],
    );
    assert_relative_eq!(gb_sector_7_3.angle_sum, 20.0 * PI / 21.0, epsilon = 1e-11);
    assert_relative_eq!(gb_sector_7_3.angular_defect, PI / 21.0, epsilon = 1e-11);
    assert_relative_eq!(gb_sector_7_3.area, PI / 21.0, epsilon = 1e-11);
    assert_relative_eq!(
        7.0 * gb_sector_7_3.area,
        regular_polygon_gauss_bonnet_area(7, 3).expect("area {7,3}"),
        epsilon = 1e-11
    );

    // {5, 4} right-angled pentagon: central angle 2*pi/5, base angles pi/4 + pi/4 -> sum = 9*pi/10,
    // defect = pi/10, and 5 sectors sum to pentagon area = pi/2!
    let mut buf_5_4 = HyperbolicTilingBuffer::with_capacity(128);
    let count_5_4 = buf_5_4.configure(5, 4, 2).expect("{5, 4} valid");
    assert!(count_5_4 > 5);
    let central_5_4 = &buf_5_4.polygons()[0];
    let gb_sector_5_4 = gauss_bonnet_triangle(
        central_5_4.center,
        central_5_4.vertices[0],
        central_5_4.vertices[1],
    );
    assert_relative_eq!(gb_sector_5_4.angle_sum, 9.0 * PI / 10.0, epsilon = 1e-11);
    assert_relative_eq!(gb_sector_5_4.angular_defect, PI / 10.0, epsilon = 1e-11);
    assert_relative_eq!(
        5.0 * gb_sector_5_4.area,
        regular_polygon_gauss_bonnet_area(5, 4).expect("area {5,4}"),
        epsilon = 1e-11
    );

    // Non-hyperbolic Euclidean {4, 4} and spherical {3, 5} must be rejected cleanly.
    let err = buf_5_4.configure(4, 4, 2).expect_err("{4,4} is Euclidean");
    assert_eq!(err.severity(), Severity::Error);
    assert!(err.metadata().contains_key("error_type"));
}

#[test]
fn test_geodesic_arc_sampling_poincare_vs_klein_chords() {
    let a = Complex64::new(0.45, 0.15);
    let b = Complex64::new(-0.1, 0.58);
    let arc = GeodesicArc::through_points(a, b);
    assert!(!arc.is_diameter);
    assert_relative_eq!(
        arc.circle_center.norm_sqr() - arc.circle_radius * arc.circle_radius,
        1.0,
        epsilon = 1e-11
    );

    let mut poincare_pts = [Complex64::new(0.0, 0.0); 9];
    let mut klein_pts = [Complex64::new(0.0, 0.0); 9];
    arc.sample_into(HyperbolicModel::Poincare, &mut poincare_pts);
    arc.sample_into(HyperbolicModel::Klein, &mut klein_pts);

    // Every Poincare sample lies on the orthogonal circle; every Klein sample is collinear.
    for pt in poincare_pts {
        assert_relative_eq!(
            (pt - arc.circle_center).norm(),
            arc.circle_radius,
            epsilon = 1e-11
        );
    }
    let k0 = klein_pts[0];
    let dir = klein_pts[8] - k0;
    for pt in klein_pts {
        let rel = pt - k0;
        let cross = rel.re * dir.im - rel.im * dir.re;
        assert!(cross.abs() < 1e-12);
    }
}

#[test]
fn test_geodesic_billiards_reflection_invariants_and_lyapunov_divergence() {
    let mut sys = HyperbolicBilliardSystem::new(HyperbolicBilliardConfig {
        p: 7,
        q: 3,
        initial_pos: Complex64::new(0.07, -0.04),
        initial_angle: 0.51,
        epsilon: 1e-6,
    })
    .expect("valid {7, 3} billiard configuration");

    for _ in 0..240 {
        sys.step(0.04);
        let p1 = sys.primary();
        let p2 = sys.twin();
        assert!(p1.pos.norm() <= POINCARE_DISK_RADIUS_MAX);
        assert!(p2.pos.norm() <= POINCARE_DISK_RADIUS_MAX);
        assert_relative_eq!(p1.dir.norm(), 1.0, epsilon = 1e-11);
        assert_relative_eq!(p2.dir.norm(), 1.0, epsilon = 1e-11);

        // Both particles remain inside or on the boundary of every geodesic wall.
        for wall in sys.walls() {
            let d1 = (p1.pos - wall.arc.circle_center).norm();
            let d2 = (p2.pos - wall.arc.circle_center).norm();
            assert!(d1 + 1e-9 >= wall.arc.circle_radius);
            assert!(d2 + 1e-9 >= wall.arc.circle_radius);
        }
    }

    assert!(sys.primary_bounces() >= 4);
    let latest = sys.latest_sample();
    assert!(latest.log_ratio > 2.0);
    assert!(latest.ftle > 0.2);
}

// [cite:modular_polynomials_review]
