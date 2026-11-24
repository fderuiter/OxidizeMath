#![cfg_attr(any(), verified(opt_out = "verification_harness"))]

use crate::mms::{SolverCertificate, ols_convergence_order};
use math_commons::math_kernel::types::{StepSize, flatten_2d_index};
use nalgebra::{Point3, SVector};
use oxidize_core::rng::OxidizeRng;
use pure_math::pure_math::analysis::integration::Trapezoidal;
use pure_math::pure_math::analysis::ode::{ArrayState, RungeKutta4};
use pure_math::pure_math::analysis::pde::fused_stepper::FusedStencilStepper;
use pure_math::pure_math::analysis::pde::greens::GreenFunctionSolver1D;
use pure_math::pure_math::analysis::pde::laplace::RectangularLaplaceSolver;
use pure_math::pure_math::analysis::pde::wave::{WaveField2D, WaveFieldConfig2D};
use pure_math::pure_math::analysis::pde::wave_2d::{
    BoundaryMode2D, Wave2DMedium, Wave2DParams, Wave2DSolver,
};
use pure_math::pure_math::differential_geometry::buckled_flow::{
    BuckledSurfaceFlow, SurfaceTopology,
};
use pure_math::pure_math::differential_geometry::heat_equation::HeatEquationSolver;
use pure_math::pure_math::differential_geometry::mean_curvature_flow::{
    DiscreteSurface, MeanCurvatureFlow,
};
use pure_math::pure_math::differential_geometry::surface::Sphere;
use pure_math::pure_math::statistics::ou_process::{
    EulerMaruyama, ExactOuSolver, OuParams, TimeStep,
};
use pure_math::pure_math::tensor::TwoSphereMetric;
use pure_math::pure_math::tensor::geodesic::{GeodesicState, GeodesicSystem};
use std::f64::consts::{FRAC_PI_2, PI};

fn pm_cert(
    name: &'static str,
    min_p: f64,
    hs: &[f64],
    errs: &[f64],
    max_err: f64,
) -> SolverCertificate {
    SolverCertificate::evaluate(
        name,
        "pure_math",
        min_p,
        ols_convergence_order(hs, errs),
        errs[2],
        max_err,
    )
}

fn certify_green_and_laplace() -> Vec<SolverCertificate> {
    let ns = [32usize, 64, 128];
    let (mut hs_g, mut err_g) = ([0.0; 3], [0.0; 3]);
    let green = |x: f64, xi: f64| {
        if xi <= x {
            xi * (1.0 - x)
        } else {
            x * (1.0 - xi)
        }
    };
    for (k, &n) in ns.iter().enumerate() {
        hs_g[k] = 1.0 / n as f64;
        let solver = GreenFunctionSolver1D::with_integrator(Trapezoidal { steps: n });
        let u_mid = solver.solve_at(0.5, (0.0, 1.0), green, |xi| PI * PI * (PI * xi).sin());
        err_g[k] = (u_mid - 1.0).abs();
    }

    let lap = RectangularLaplaceSolver {
        width: 1.0,
        height: 1.0,
        coefficients: vec![1.0],
    };
    let hs_l = [0.02_f64, 0.01, 0.005];
    let err_l = hs_l.map(|h| {
        let u0 = lap.evaluate(0.5, 0.5);
        ((lap.evaluate(0.5 + h, 0.5)
            + lap.evaluate(0.5 - h, 0.5)
            + lap.evaluate(0.5, 0.5 + h)
            + lap.evaluate(0.5, 0.5 - h)
            - 4.0 * u0)
            / (h * h))
            .abs()
    });
    vec![
        pm_cert("GreenFunctionSolver1D", 1.95, &hs_g, &err_g, 1e-4),
        pm_cert("RectangularLaplaceSolver", 1.95, &hs_l, &err_l, 5e-3),
    ]
}

fn certify_wave_solvers() -> Vec<SolverCertificate> {
    let params = Wave2DParams {
        medium: Wave2DMedium::AcousticWave,
        boundary: BoundaryMode2D::Dirichlet,
        wave_speed: 1.0,
        plate_stiffness: 0.0,
        damping: 0.0,
    };
    let (mut hs_w, mut err_w) = ([0.0; 3], [0.0; 3]);
    for (k, &m) in [17usize, 33, 65].iter().enumerate() {
        let dx = 1.0 / (m - 1) as f64;
        hs_w[k] = dx;
        let mut solver = Wave2DSolver::new(m, m, dx, dx);
        let mut init = vec![0.0; m * m];
        for j in 0..m {
            for i in 0..m {
                init[flatten_2d_index(i, j, m)] =
                    (PI * i as f64 * dx).sin() * (PI * j as f64 * dx).sin();
            }
        }
        solver.set_stationary_displacement(&init);
        let sub = (0.12 / (0.25 * dx)).ceil() as usize;
        for _ in 0..sub {
            solver.step(&params, 0.12 / sub as f64);
        }
        let exact = (2.0_f64.sqrt() * PI * 0.12).cos();
        err_w[k] = (solver.displacement()[flatten_2d_index(m / 2, m / 2, m)] - exact).abs();
    }

    let run_fdtd = |steps: usize| -> f64 {
        let n = 32usize;
        let dx = 1.0 / n as f64;
        let mut wf = WaveField2D::new(WaveFieldConfig2D {
            width: n,
            height: n,
            dx,
            dy: dx,
            sponge_width: 0,
            sponge_max_damping: 0.0,
            ..WaveFieldConfig2D::default()
        });
        let src = pure_math::pure_math::analysis::pde::wave::WaveSource2D {
            kind: pure_math::pure_math::analysis::pde::wave::WaveSourceKind::PointSource {
                x: n / 2,
                y: n / 2,
            },
            frequency: 2.0,
            amplitude: 1.0,
            phase: 0.0,
        };
        for _ in 0..steps {
            wf.step(0.08 / steps as f64, Some(&src));
        }
        wf.field()[flatten_2d_index(n / 2, n / 2, n)]
    };
    let ref_fdtd = run_fdtd(256);
    let hs_wf = [0.08 / 8.0, 0.08 / 16.0, 0.08 / 32.0];
    let err_wf = [8usize, 16, 32].map(|s| (run_fdtd(s) - ref_fdtd).abs());
    vec![
        pm_cert("Wave2DSolver", 1.80, &hs_w, &err_w, 5e-3),
        pm_cert("WaveField2D", 1.50, &hs_wf, &err_wf, 5e-3),
    ]
}

fn certify_fused_stencil() -> SolverCertificate {
    let (mut hs_f, mut err_f) = ([0.0; 3], [0.0; 3]);
    let (diff, t_f) = (0.05_f64, 0.04_f64);
    for (k, &n) in [81usize, 161, 321].iter().enumerate() {
        let dx = 1.0 / (n - 1) as f64;
        hs_f[k] = dx;
        let stepper = FusedStencilStepper::new(StepSize(dx));
        let mut u: Vec<f64> = (0..n).map(|i| (PI * (i as f64 * dx)).cos()).collect();
        let mut next = vec![0.0; n];
        let steps = ((t_f / (0.2 * dx * dx / diff)).ceil() as usize).max(8);
        let (dt, inv_dx2) = (t_f / steps as f64, 1.0 / (dx * dx));
        for _ in 0..steps {
            stepper.step_1d_coupled_neumann(
                n,
                [u.as_slice()],
                [&mut next[..]],
                dt,
                1.0,
                |_i, prev, curr, nxt, _ops| [diff * (prev[0] - 2.0 * curr[0] + nxt[0]) * inv_dx2],
            );
            std::mem::swap(&mut u, &mut next);
        }
        err_f[k] = (u[0] - (-PI * PI * diff * t_f).exp()).abs();
    }
    pm_cert("FusedStencilStepper", 0.80, &hs_f, &err_f, 2e-3)
}

fn certify_curvature_flows() -> Vec<SolverCertificate> {
    let sphere = Sphere { radius: 1.0 };
    let (mut hs_h, mut err_h) = ([0.0; 3], [0.0; 3]);
    for (k, &nv) in [17usize, 33, 65].iter().enumerate() {
        let (nu, dv) = (8usize, PI / (nv - 1) as f64);
        hs_h[k] = dv;
        let h_eff_sq = dv.powi(2).min((dv.sin() * (2.0 * PI / nu as f64)).powi(2));
        let steps = ((0.04 / (0.15 * h_eff_sq / 0.05)).ceil() as usize).max(8);
        let mut solver = HeatEquationSolver::new(
            &sphere,
            0.05,
            (0.0, 2.0 * PI),
            (0.0, PI),
            (nu, nv),
            |_u, v| v.cos(),
        )
        .with_topology(SurfaceTopology::PolarSphere);
        for _ in 0..steps {
            solver.step(0.04 / steps as f64);
        }
        let decay = (-2.0 * 0.05 * 0.04_f64).exp();
        err_h[k] = (2..(nv - 2))
            .map(|j| (solver.u_grid[0][j] - decay * (j as f64 * dv).cos()).abs())
            .fold(0.0_f64, f64::max);
    }

    let (mut hs_m, mut err_m) = ([0.0; 3], [0.0; 3]);
    let r_exact = (4.0_f64 - 0.04).sqrt();
    for (k, &n) in [16usize, 32, 64].iter().enumerate() {
        hs_m[k] = 1.0 / n as f64;
        let mut pts = vec![vec![Point3::origin(); n]; n];
        for (i, row) in pts.iter_mut().enumerate() {
            let u = 2.0 * PI * i as f64 / n as f64;
            for (j, pt) in row.iter_mut().enumerate() {
                let v = 0.2 + (PI - 0.4) * (j as f64 / (n - 1) as f64);
                *pt = Point3::new(
                    2.0 * v.sin() * u.cos(),
                    2.0 * v.sin() * u.sin(),
                    2.0 * v.cos(),
                );
            }
        }
        let mut mcf = MeanCurvatureFlow::new(DiscreteSurface::new(pts, true, false));
        let steps = (n * n) / 8;
        for _ in 0..steps {
            mcf.step(0.02 / steps as f64);
        }
        err_m[k] = (mcf.surface.points[0][n / 2].coords.norm() - r_exact).abs();
    }
    vec![
        pm_cert("HeatEquationSolver", 1.80, &hs_h, &err_h, 1e-4),
        pm_cert("MeanCurvatureFlow", 1.80, &hs_m, &err_m, 2e-4),
    ]
}

fn certify_buckled_and_geodesic() -> Result<Vec<SolverCertificate>, String> {
    let sphere = Sphere { radius: 1.0 };
    let morphogen = vec![0.4_f64; 144];
    let eval_buckled = |steps: usize| -> f64 {
        let mut flow = BuckledSurfaceFlow::from_parametric(
            &sphere,
            SurfaceTopology::PolarSphere,
            (0.0, 2.0 * PI),
            (0.0, PI),
            12,
            12,
        );
        for _ in 0..steps {
            let _ = flow.step_buckled(&morphogen, 0.0, 0.2, 0.1, 0.05, 0.004 / steps as f64);
        }
        flow.displacements()[flatten_2d_index(6, 6, 12)]
    };
    let ref_bf = eval_buckled(128);
    let hs_bf = [0.004 / 8.0, 0.004 / 16.0, 0.004 / 32.0];
    let err_bf = [8usize, 16, 32].map(|s| (eval_buckled(s) - ref_bf).abs());

    let geo = GeodesicSystem::new(TwoSphereMetric::new(1.0).map_err(|e| format!("{e:?}"))?);
    let (mut hs_geo, mut err_geo) = ([0.0; 3], [0.0; 3]);
    let s0 = ArrayState::<4>::from_pos_vel(
        &SVector::<f64, 2>::new(1.1, 0.0),
        &SVector::<f64, 2>::new(0.2, 0.8),
    );
    let mut ref_s = s0;
    let mut ref_rk = RungeKutta4::new(&ref_s);
    for i in 0..1000usize {
        let _ = geo.step_rk4(&mut ref_rk, i as f64 * 0.0005, &mut ref_s, 0.0005);
    }
    let (p_ref, v_ref) = ref_s.split_pos_vel();
    for (k, &n) in [10usize, 20, 40].iter().enumerate() {
        let dt = 0.5 / n as f64;
        hs_geo[k] = dt;
        let (mut st, mut rk) = (s0, RungeKutta4::new(&s0));
        for i in 0..n {
            let _ = geo.step_rk4(&mut rk, i as f64 * dt, &mut st, dt);
        }
        let (p_n, v_n) = st.split_pos_vel();
        err_geo[k] = (p_n - p_ref).norm().max((v_n - v_ref).norm());
    }
    let _ = FRAC_PI_2;
    Ok(vec![
        pm_cert("BuckledSurfaceFlow", 0.90, &hs_bf, &err_bf, 1e-4),
        pm_cert("GeodesicSystem", 3.90, &hs_geo, &err_geo, 1e-6),
    ])
}

fn certify_ou_solvers() -> Result<Vec<SolverCertificate>, String> {
    let params = OuParams::from_values(2.0, 1.5, 0.0).map_err(|e| format!("{e:?}"))?;
    let exact_ou = 2.0 + (0.5 - 2.0) * (-1.5 * 0.8_f64).exp();
    let (mut hs_ou, mut err_ou) = ([0.0; 3], [0.0; 3]);
    let mut err_exact_ou = 0.0_f64;
    let mut rng = OxidizeRng::from_seed(99);
    for (k, &n) in [20usize, 40, 80].iter().enumerate() {
        let dt = 0.8 / n as f64;
        hs_ou[k] = dt;
        let step = TimeStep::new(dt).map_err(|e| format!("{e:?}"))?;
        let em = EulerMaruyama::new(params, step);
        let exact_solver = ExactOuSolver::new(params, step);
        err_ou[k] = (*em.simulate(0.5, n, &mut rng).last().unwrap() - exact_ou).abs();
        err_exact_ou = err_exact_ou
            .max((*exact_solver.simulate(0.5, n, &mut rng).last().unwrap() - exact_ou).abs());
    }
    let noisy = OuParams::from_values(2.0, 1.5, 0.8).map_err(|e| format!("{e:?}"))?;
    let exact_noisy = ExactOuSolver::new(noisy, TimeStep::new(0.4).map_err(|e| format!("{e:?}"))?);
    let exact_var = (0.64 / 3.0) * (1.0 - (-2.4_f64).exp());
    err_exact_ou = err_exact_ou
        .max((exact_noisy.conditional_mean(0.5, 0.8) - exact_ou).abs())
        .max((exact_noisy.conditional_variance(0.8) - exact_var).abs());
    Ok(vec![
        pm_cert("EulerMaruyama", 0.95, &hs_ou, &err_ou, 1e-2),
        SolverCertificate::evaluate("ExactOuSolver", "pure_math", 0.0, 1.0, err_exact_ou, 1e-12),
    ])
}

/// Certifies all 11 PDE, curvature-flow, geodesic, and OU solvers in `pure_math`.
pub fn certify_pure_pde_and_geom() -> Result<Vec<SolverCertificate>, String> {
    let mut out = certify_green_and_laplace();
    out.extend(certify_wave_solvers());
    out.push(certify_fused_stencil());
    out.extend(certify_curvature_flows());
    out.extend(certify_buckled_and_geodesic()?);
    out.extend(certify_ou_solvers()?);
    Ok(out)
}
