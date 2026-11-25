//! Verifies zero heap allocations for `BiQuadraticSurface::fit_with_workspace`.

use domain_physics::physics::medical::radar_gating::{
    BiQuadraticSurface, biquadratic_fit_workspace_bytes,
};
use math_memory::OwnedByteArena;
use nalgebra::Point3;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn biquadratic_surface_fit_with_workspace_zero_heap_allocs() {
    let expected = [0.4, -0.2, 0.35, 0.15, -0.08, 0.12];
    let mut points = Vec::new();
    for ix in -2..=2 {
        for iy in -2..=2 {
            let x = ix as f64 * 0.5;
            let y = iy as f64 * 0.4;
            let z = expected[0]
                + expected[1] * x
                + expected[2] * y
                + expected[3] * x * y
                + expected[4] * x * x
                + expected[5] * y * y;
            points.push(Point3::new(x, y, z));
        }
    }
    let mut surface = BiQuadraticSurface::fit(&points).expect("initial surface");
    let mut arena = OwnedByteArena::with_capacity(biquadratic_fit_workspace_bytes(points.len()));
    let mut coeffs = [0.0_f64; 6];
    let (res, _metrics) = VerifiedEngine::run_verified(|| {
        BiQuadraticSurface::fit_with_workspace(&points, &mut coeffs, &mut arena)
            .expect("biquadratic fit succeeds");
        surface
            .fit_into_with_workspace(&points, &mut arena)
            .expect("in-place biquadratic fit succeeds");
        coeffs
    });
    for (got, exp) in res.iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 1e-10, "got {got}, expected {exp}");
    }
    for (got, exp) in surface.coefficients.iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 1e-10, "got {got}, expected {exp}");
    }
}
