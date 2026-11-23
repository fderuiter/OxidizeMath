//! Zero-allocation `VerifiedAllocator` integration tests for flat-grid `MeanCurvatureFlow`,
//! `HeatEquationSolver`, and curvilinear `helmholtz_decompose`.

use nalgebra::{Point3, Vector3};
use pure_math::pure_math::differential_geometry::buckled_flow::SurfaceTopology;
use pure_math::pure_math::differential_geometry::heat_equation::HeatEquationSolver;
use pure_math::pure_math::differential_geometry::mean_curvature_flow::{
    DiscreteSurface, MeanCurvatureFlow,
};
use pure_math::pure_math::differential_geometry::surface::{KleinBottle, Sphere, Torus};
use pure_math::pure_math::vector_calculus::coordinates::{Cylindrical, Spherical};
use pure_math::pure_math::vector_calculus::theorems::{
    Domain, HelmholtzWorkspace, helmholtz_decompose,
};
use std::f64::consts::PI;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_mean_curvature_flow_zero_alloc_steps() {
    let surface = DiscreteSurface::from_parametric(
        (0.0, 2.0 * PI),
        (0.1, PI - 0.1),
        16,
        12,
        true,
        false,
        |u, v| {
            Point3::new(
                2.0 * v.sin() * u.cos(),
                2.0 * v.sin() * u.sin(),
                2.0 * v.cos(),
            )
        },
    );
    let mut flow = MeanCurvatureFlow::new(surface);
    let dt = flow.recommended_cfl_dt();

    let (area, metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..8 {
            flow.step(dt);
        }
        flow.surface.surface_area()
    });
    assert!(metrics.function_calls > 0);
    assert!(area.is_finite() && area > 0.0);
}

#[test]
#[verified_engine::verified]
fn test_heat_equation_solver_zero_alloc_steps() {
    let torus = Torus {
        major_radius: 3.0,
        minor_radius: 1.0,
    };
    let mut solver = HeatEquationSolver::new(
        &torus,
        0.05,
        (0.0, 2.0 * PI),
        (0.0, 2.0 * PI),
        (18, 18),
        |u, v| (-((u - PI).powi(2) + (v - PI).powi(2))).exp(),
    );
    let dt = solver.recommended_cfl_dt();

    let (heat, metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..10 {
            solver.step(dt);
        }
        solver.total_heat()
    });
    assert!(metrics.function_calls > 0);
    assert!(heat.is_finite() && heat > 0.0);
}

#[test]
#[verified_engine::verified]
fn test_helmholtz_decompose_zero_alloc_curvilinear() {
    let cyl = Cylindrical;
    let sph = Spherical;
    let domain = Domain {
        min: Vector3::new(1.2, 0.3, 0.2),
        max: Vector3::new(2.2, 1.3, 1.2),
    };
    let mut ws = HelmholtzWorkspace::new(10, 10, 10);
    let mut samples = vec![Vector3::zeros(); ws.len()];
    for k in 0..ws.nz {
        for j in 0..ws.ny {
            for i in 0..ws.nx {
                let p = ws.cell_center(&domain, i, j, k);
                samples[ws.index(i, j, k)] = Vector3::new(p[0] * p[0], p[1].sin(), p[2]);
            }
        }
    }

    let ((e1, e2), metrics) = VerifiedEngine::run_verified(|| {
        let s1 = helmholtz_decompose(&cyl, &samples, &domain, &mut ws, 40);
        let s2 = helmholtz_decompose(&sph, &samples, &domain, &mut ws, 40);
        (s1.max_reconstruction_error, s2.max_reconstruction_error)
    });
    assert!(metrics.function_calls > 0);
    assert!(e1 < 1e-12 && e2 < 1e-12);
}

#[test]
#[verified_engine::verified]
fn test_heat_equation_topologies_zero_alloc_steps() {
    let torus = Torus {
        major_radius: 2.5,
        minor_radius: 0.75,
    };
    let mut torus_solver = HeatEquationSolver::new(
        &torus,
        0.04,
        (0.0, 2.0 * PI),
        (0.0, 2.0 * PI),
        (12, 12),
        |u, v| 1.0 + 0.5 * u.cos() * v.sin(),
    )
    .with_topology(SurfaceTopology::PeriodicTorus);

    let sphere = Sphere { radius: 1.5 };
    let mut sphere_solver = HeatEquationSolver::new(
        &sphere,
        0.03,
        (0.0, 2.0 * PI),
        (0.0, PI),
        (12, 11),
        |u, v| 1.0 + 0.4 * v.cos() + 0.1 * u.sin(),
    )
    .with_topology(SurfaceTopology::PolarSphere);

    let klein = KleinBottle { radius: 2.0 };
    let mut klein_solver = HeatEquationSolver::new(
        &klein,
        0.02,
        (0.0, 2.0 * PI),
        (0.0, 2.0 * PI),
        (12, 12),
        |u, v| 1.0 + 0.3 * (u - PI).sin() * v.cos(),
    )
    .with_topology(SurfaceTopology::TwistedKlein);

    let ((h_torus, h_sphere, h_klein), metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..5 {
            torus_solver.step(0.002);
            sphere_solver.step(0.002);
            klein_solver.step(0.001);
        }
        (
            torus_solver.total_heat(),
            sphere_solver.total_heat(),
            klein_solver.total_heat(),
        )
    });
    assert!(metrics.function_calls > 0);
    assert!(h_torus.is_finite() && h_torus > 0.0);
    assert!(h_sphere.is_finite() && h_sphere > 0.0);
    assert!(h_klein.is_finite() && h_klein > 0.0);
}
