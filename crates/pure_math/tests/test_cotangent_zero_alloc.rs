//! Integration tests verifying zero-allocation hot-path stepping under `VerifiedAllocator`
//! for `CotangentLaplacian` and `MeanCurvatureFlow` across 100 steps.

use nalgebra::Point3;
use pure_math::pure_math::differential_geometry::CotangentLaplacian;
use pure_math::pure_math::differential_geometry::mean_curvature_flow::{
    DiscreteSurface, MeanCurvatureFlow,
};
use std::f64::consts::PI;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_cotangent_laplacian_and_mean_curvature_flow_100_steps_verified_allocator() {
    let nu = 16;
    let nv = 12;
    let mut positions = Vec::with_capacity(2 + (nv - 2) * nu);
    positions.push([0.0, 0.0, 1.0]);
    positions.push([0.0, 0.0, -1.0]);
    for j in 1..(nv - 1) {
        let v = PI * (j as f64) / ((nv - 1) as f64);
        let (sv, cv) = v.sin_cos();
        for i in 0..nu {
            let u = 2.0 * PI * (i as f64) / (nu as f64);
            let (su, cu) = u.sin_cos();
            positions.push([sv * cu, sv * su, cv]);
        }
    }
    let ring = |i: usize, j: usize| 2 + (j - 1) * nu + (i % nu);
    let mut tris = Vec::new();
    for i in 0..nu {
        tris.push([0, ring(i, 1), ring(i + 1, 1)]);
    }
    for j in 1..(nv - 2) {
        for i in 0..nu {
            tris.push([ring(i, j), ring(i, j + 1), ring(i + 1, j + 1)]);
            tris.push([ring(i, j), ring(i + 1, j + 1), ring(i + 1, j)]);
        }
    }
    for i in 0..nu {
        tris.push([ring(i, nv - 2), 1, ring(i + 1, nv - 2)]);
    }

    let lap = CotangentLaplacian::from_indexed(&positions, &tris);
    let mut heat_u: Vec<f64> = positions.iter().map(|p| p[2] + 0.3 * p[0]).collect();
    let mut heat_next = vec![0.0; positions.len()];
    let mut lap_out = vec![0.0; positions.len()];
    let mut mean_curv = vec![0.0; positions.len()];

    let mut grid_pts = vec![vec![Point3::origin(); nv]; nu];
    for (i, row) in grid_pts.iter_mut().enumerate() {
        let u = 2.0 * PI * (i as f64) / (nu as f64);
        for (j, slot) in row.iter_mut().enumerate() {
            let v = PI * (j as f64) / ((nv - 1) as f64);
            *slot = Point3::new(v.sin() * u.cos(), v.sin() * u.sin(), v.cos());
        }
    }
    let mut flow = MeanCurvatureFlow::new(DiscreteSurface::new(grid_pts, true, false));

    let ((), metrics) = VerifiedEngine::run_verified(|| {
        lap.apply(&heat_u, &mut lap_out);
        lap.compute_mean_curvature(&positions, &mut mean_curv);
        for _ in 0..100 {
            lap.step_heat_euler(&heat_u, &mut heat_next, 0.04, 1e-3);
            std::mem::swap(&mut heat_u, &mut heat_next);
            flow.step(1e-4);
        }
    });

    assert!(metrics.function_calls >= 200);
    assert!(heat_u.iter().all(|v| v.is_finite()));
    assert!(lap_out.iter().all(|v| v.is_finite()));
    assert!(mean_curv.iter().all(|v| v.is_finite()));
    for row in &flow.surface.points {
        for p in row {
            assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
        }
    }
}
