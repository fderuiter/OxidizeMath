//! Integration tests for test differential geometry.
use nalgebra::Point3;
use pure_math::pure_math::analysis::pde::{PdeClassification, SecondOrderLinearPde2D};
use pure_math::pure_math::differential_geometry::heat_equation::HeatEquationSolver;
use pure_math::pure_math::differential_geometry::mean_curvature_flow::{
    DiscreteSurface, MeanCurvatureFlow,
};
use pure_math::pure_math::differential_geometry::operators::laplace_beltrami;
use pure_math::pure_math::differential_geometry::surface::{Sphere, SurfaceAnalysis, Torus};
use std::f64::consts::PI;

#[test]
#[verified_engine::verified]
fn test_sphere_properties() {
    let radius = 2.0;
    let sphere = Sphere { radius };

    // Test curvature at equator
    let u = 0.0;
    let v = PI / 2.0; // Equator (polar angle)

    let k = sphere.gaussian_curvature(u, v);
    let h = sphere.mean_curvature(u, v);

    // K = 1/R^2
    let expected_k = 1.0 / (radius * radius);
    let expected_h = 1.0 / radius;

    assert!(
        (k - expected_k).abs() < 1e-4,
        "Gaussian curvature mismatch: {} vs {}",
        k,
        expected_k
    );
    assert!(
        (h.abs() - expected_h).abs() < 1e-4,
        "Mean curvature mismatch: {} vs {}",
        h,
        expected_h
    );
}

#[test]
#[verified_engine::verified]
fn test_torus_curvature_sign() {
    let major = 3.0;
    let minor = 1.0;
    let torus = Torus {
        major_radius: major,
        minor_radius: minor,
    };

    // Outer edge (u=0, v=0) -> Positive Gaussian curvature
    let k_out = torus.gaussian_curvature(0.0, 0.0);
    assert!(k_out > 0.0);

    // Inner edge (u=0, v=pi) -> Negative Gaussian curvature
    let k_in = torus.gaussian_curvature(0.0, PI);
    assert!(k_in < 0.0);
}

#[test]
#[verified_engine::verified]
fn test_pde_classification() {
    let laplace = SecondOrderLinearPde2D {
        a: 1.0,
        b: 0.0,
        c: 1.0,
    };
    assert_eq!(laplace.classify(), PdeClassification::Elliptic);

    let wave = SecondOrderLinearPde2D {
        a: 1.0,
        b: 0.0,
        c: -1.0,
    };
    assert_eq!(wave.classify(), PdeClassification::Hyperbolic);
}

#[test]
#[verified_engine::verified]
fn test_laplace_beltrami_constant() {
    let sphere = Sphere { radius: 1.0 };
    let constant_func = |_: f64, _: f64| 1.0;

    let lb = laplace_beltrami(&sphere, 0.0, PI / 2.0, &constant_func);
    assert!(lb.abs() < math_commons::registry::TOLERANCE_FAST);
}

#[test]
#[verified_engine::verified]
fn test_heat_equation_smoothing() {
    let radius = 1.0;
    let sphere = Sphere { radius };

    // Initial condition: Hot spot at equator (v = PI/2)
    // u from 0 to 2pi, v from 0 to pi
    // Let's verify heat diffuses.
    let initial = |_: f64, v: f64| {
        if (v - PI / 2.0).abs() < 0.5 { 1.0 } else { 0.0 }
    };

    let mut solver = HeatEquationSolver::new(
        &sphere,
        0.1, // alpha
        (0.0, 2.0 * PI),
        (0.1, PI - 0.1), // Avoid poles singularity for numerical grid
        (20, 20),
        initial,
    );

    // Measure max temp
    let max_temp_init = solver
        .u_grid
        .iter()
        .flatten()
        .fold(0.0f64, |a, &b| a.max(b));

    // Run steps
    for _ in 0..50 {
        solver.step(0.01);
    }

    let max_temp_final = solver
        .u_grid
        .iter()
        .flatten()
        .fold(0.0f64, |a, &b| a.max(b));

    // Heat should spread, lowering the max temp
    assert!(
        max_temp_final < max_temp_init,
        "Max temp should decrease: {} -> {}",
        max_temp_init,
        max_temp_final
    );
}

#[test]
#[verified_engine::verified]
fn test_mean_curvature_flow_shrinkage() {
    // Create a discrete sphere
    let nu = 20;
    let nv = 10;
    let radius = 2.0;

    let mut points = vec![vec![Point3::origin(); nv]; nu];

    #[allow(clippy::needless_range_loop)]
    for i in 0..nu {
        #[allow(clippy::needless_range_loop)]
        for j in 0..nv {
            let u = 2.0 * PI * i as f64 / nu as f64;
            // Avoid exact 0 and PI to prevent singularities in parameterization derivatives
            let v_frac = j as f64 / (nv as f64 - 1.0);
            let v = 0.05 + v_frac * (PI - 0.1);

            let x = radius * v.sin() * u.cos();
            let y = radius * v.sin() * u.sin();
            let z = radius * v.cos();
            points[i][j] = Point3::new(x, y, z);
        }
    }

    let discrete_sphere = DiscreteSurface::new(points, true, false);
    let mut flow = MeanCurvatureFlow::new(discrete_sphere);

    // Compute initial average radius
    let avg_radius = |surf: &DiscreteSurface| -> f64 {
        let mut sum = 0.0;
        let mut count = 0;
        for row in &surf.points {
            for p in row {
                sum += p.coords.norm(); // distance from origin
                count += 1;
            }
        }
        sum / count as f64
    };

    let r_init = avg_radius(&flow.surface);

    // Step
    // H = 1/R = 0.5. Speed ~ 0.5.
    // dt = 0.01. Change ~ 0.005.
    for _ in 0..10 {
        flow.step(0.01);
    }

    let r_final = avg_radius(&flow.surface);

    assert!(
        r_final < r_init,
        "Sphere should shrink under MCF: {} -> {}",
        r_init,
        r_final
    );
}

#[test]
#[verified_engine::verified]
fn test_heat_equation_sphere_eigenmode_grid_convergence() {
    use pure_math::pure_math::differential_geometry::buckled_flow::SurfaceTopology;

    // Exact spherical harmonic eigenmode u(u, v, t) = cos(v) * exp(-2 * alpha * t) on unit sphere
    let sphere = Sphere { radius: 1.0 };
    let alpha = 0.05;
    let t_final = 0.04;
    let grids = [17_usize, 33, 65];
    let mut errors = [0.0_f64; 3];
    let mut spacings = [0.0_f64; 3];

    for (idx, &nv) in grids.iter().enumerate() {
        let nu = 8;
        let du = 2.0 * PI / nu as f64;
        let dv = PI / (nv - 1) as f64;
        spacings[idx] = dv;
        let h_eff_sq = dv.powi(2).min((dv.sin() * du).powi(2));
        let dt_target = 0.15 * h_eff_sq / alpha;
        let steps = ((t_final / dt_target).ceil() as usize).max(8);
        let dt = t_final / steps as f64;

        let mut solver = HeatEquationSolver::new(
            &sphere,
            alpha,
            (0.0, 2.0 * PI),
            (0.0, PI),
            (nu, nv),
            |_u, v| v.cos(),
        )
        .with_topology(SurfaceTopology::PolarSphere);

        for _ in 0..steps {
            solver.step(dt);
        }

        let decay = (-2.0 * alpha * t_final).exp();
        let mut max_err = 0.0_f64;
        for j in 2..(nv - 2) {
            let v = j as f64 * dv;
            let exact = decay * v.cos();
            let err = (solver.u_grid[0][j] - exact).abs();
            max_err = max_err.max(err);
        }
        errors[idx] = max_err;
    }

    let order_coarse = (errors[0] / errors[1]).ln() / (spacings[0] / spacings[1]).ln();
    let order_fine = (errors[1] / errors[2]).ln() / (spacings[1] / spacings[2]).ln();
    assert!(
        errors[2] < errors[1] && errors[1] < errors[0],
        "HeatEquationSolver error must decrease monotonically: {errors:?}"
    );
    assert!(
        order_coarse > 1.75 && order_fine > 1.85,
        "HeatEquationSolver expected 2nd-order spatial convergence, got ({order_coarse:.3}, {order_fine:.3})"
    );
}

#[test]
#[verified_engine::verified]
fn test_mean_curvature_flow_sphere_exact_radius_convergence() {
    // Under dr/dt = H * n_inward = -(1/R) e_r, a sphere of radius R0 shrinks as R(t) = sqrt(R0^2 - 2*t).
    let r0 = 2.0_f64;
    let t_final = 0.02_f64;
    let r_exact = (r0 * r0 - 2.0 * t_final).sqrt();
    let resolutions = [16_usize, 32, 64];
    let mut errors = [0.0_f64; 3];

    for (idx, &n) in resolutions.iter().enumerate() {
        let mut points = vec![vec![Point3::origin(); n]; n];
        for (i, row) in points.iter_mut().enumerate().take(n) {
            let u = 2.0 * PI * i as f64 / n as f64;
            for (j, pt) in row.iter_mut().enumerate().take(n) {
                let v = 0.2 + (PI - 0.4) * (j as f64 / (n - 1) as f64);
                *pt = Point3::new(r0 * v.sin() * u.cos(), r0 * v.sin() * u.sin(), r0 * v.cos());
            }
        }
        let mut flow = MeanCurvatureFlow::new(DiscreteSurface::new(points, true, false));
        let steps = (n * n) / 8;
        let dt = t_final / steps as f64;
        for _ in 0..steps {
            flow.step(dt);
        }
        let mid = n / 2;
        let r_num = flow.surface.points[0][mid].coords.norm();
        errors[idx] = (r_num - r_exact).abs();
    }

    let order_1 = (errors[0] / errors[1]).ln() / 2.0_f64.ln();
    let order_2 = (errors[1] / errors[2]).ln() / 2.0_f64.ln();
    assert!(
        errors[2] < errors[1] && errors[1] < errors[0],
        "MeanCurvatureFlow radius error must decrease on refined grids: {errors:?}"
    );
    assert!(
        order_1 > 1.75 && order_2 > 1.85,
        "MeanCurvatureFlow expected 2nd-order convergence to sqrt(R0^2 - 2t), got ({order_1:.3}, {order_2:.3})"
    );
}
