//! Tier 5 security tests for `domain_physics`: `BiQuadraticSurface` `checked_mul`
//! overflow, `resolution == 0` / `resolution == 1` division-by-zero guards, `NaN`/infinity
//! rejection, subnormal stability, and zero-allocation surface evaluation.

use domain_physics::physics::medical::radar_gating::surface::BiQuadraticSurface;
use math_commons::error::MathError;
use nalgebra::Point3;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static TEST_ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

fn sample_points(z_scale: f64) -> [Point3<f64>; 9] {
    let coords = [-1.0, 0.0, 1.0];
    let mut pts = [Point3::origin(); 9];
    let mut idx = 0;
    for &x in &coords {
        for &y in &coords {
            pts[idx] = Point3::new(x, y, z_scale * (1.0 + 0.2 * x - 0.1 * y + 0.05 * x * x));
            idx += 1;
        }
    }
    pts
}

#[test]
fn biquadratic_surface_rejects_underdetermined_non_finite_zero_and_overflow_inputs() {
    let pts = sample_points(1.0);
    assert!(BiQuadraticSurface::fit(&pts[..5]).is_none());

    let mut nan_pts = pts;
    nan_pts[3].z = f64::NAN;
    assert!(BiQuadraticSurface::fit(&nan_pts).is_none());

    let mut inf_pts = pts;
    inf_pts[4].x = f64::INFINITY;
    assert!(BiQuadraticSurface::fit(&inf_pts).is_none());

    // Overflowing x^2 / x*y product must return None rather than poisoning SVD.
    let mut huge_pts = pts;
    huge_pts[0].x = 1.0e200;
    assert!(BiQuadraticSurface::fit(&huge_pts).is_none());

    let surface = BiQuadraticSurface::fit(&pts).expect("well-conditioned 9-point surface");
    assert!(matches!(
        surface.try_generate_mesh((-1.0, 1.0), (-1.0, 1.0), 0),
        Err(MathError::StrictlyOutOfBounds { .. })
    ));
    assert!(matches!(
        surface.try_generate_mesh((-1.0, 1.0), (-1.0, 1.0), usize::MAX),
        Err(MathError::CapacityOverflow { .. })
    ));
    assert!(matches!(
        surface.try_generate_mesh((f64::NAN, 1.0), (-1.0, 1.0), 4),
        Err(MathError::NumericalError { .. })
    ));
    assert!(matches!(
        surface.try_generate_mesh((2.0, -1.0), (-1.0, 1.0), 4),
        Err(MathError::NumericalError { .. })
    ));
    assert!(
        surface
            .generate_mesh((-1.0, 1.0), (-1.0, 1.0), 0)
            .is_empty()
    );
}

#[test]
fn biquadratic_surface_resolution_one_subnormals_and_zero_alloc_evaluate() {
    let subnormal = f64::MIN_POSITIVE * 0.5;
    let pts = sample_points(subnormal);
    let surface = BiQuadraticSurface::fit(&pts).expect("subnormal amplitude surface fit");

    // resolution == 1 previously divided by (resolution - 1) == 0 producing NaN coordinates.
    let single = surface
        .try_generate_mesh((-2.0, 4.0), (-1.0, 3.0), 1)
        .expect("resolution == 1 midpoint sample");
    assert_eq!(single.len(), 1);
    assert_eq!(single[0].x, 1.0);
    assert_eq!(single[0].y, 1.0);
    assert!(single[0].z.is_finite());

    let ((z_mid, z_nan_guard), metrics) = VerifiedEngine::run_verified(|| {
        let z_mid = surface.evaluate(subnormal, -subnormal);
        let z_nan_guard = surface.evaluate(f64::INFINITY, 0.0);
        (z_mid, z_nan_guard)
    });

    assert_eq!(metrics.allocations, 0);
    assert_eq!(metrics.allocated_bytes, 0);
    assert!(z_mid.is_finite());
    assert!(z_nan_guard.is_nan());
}
