//! Runnable CLI showcase for curvilinear Stokes & Helmholtz decomposition, flat-buffer
//! Mean Curvature & Laplace-Beltrami heat flows, and multi-sheet Riemann monodromy.

use math_explorer::nalgebra::Vector3;
use math_explorer::num_complex::Complex64;
use math_explorer::pure_math::analysis::complex::integration::residue_via_integral;
use math_explorer::pure_math::analysis::complex::surfaces::{NthRootSurface, RiemannSurface};
use math_explorer::pure_math::differential_geometry::heat_equation::HeatEquationSolver;
use math_explorer::pure_math::differential_geometry::mean_curvature_flow::{
    DiscreteSurface, MeanCurvatureFlow,
};
use math_explorer::pure_math::differential_geometry::surface::{ParametricSurface, Sphere, Torus};
use math_explorer::pure_math::vector_calculus::coordinates::Cartesian;
use math_explorer::pure_math::vector_calculus::theorems::{
    Domain, HelmholtzWorkspace, StokesPatch, helmholtz_decompose, verify_stokes_theorem,
};
use std::f64::consts::{PI, TAU};

fn demo_vector_calculus() {
    let domain = Domain {
        min: Vector3::new(-1.0, -1.0, -1.0),
        max: Vector3::new(1.0, 1.0, 1.0),
    };
    let patch = StokesPatch::new(2, 0.25, (-0.8, 0.8), (-0.8, 0.8));
    let field = |p: &Vector3<f64>| Vector3::new(p[0] - p[1], p[1] + p[0], p[2]);
    let (surf_curl, loop_circ, diff) = verify_stokes_theorem(&Cartesian, &patch, field, 20);

    let mut ws = HelmholtzWorkspace::new(9, 9, 9);
    let mut samples = vec![Vector3::zeros(); 9 * 9 * 9];
    for k in 0..9 {
        for j in 0..9 {
            for i in 0..9 {
                let p = ws.cell_center(&domain, i, j, k);
                samples[ws.index(i, j, k)] = field(&p);
            }
        }
    }
    let summary = helmholtz_decompose(&Cartesian, &samples, &domain, &mut ws, 90);
    println!(
        "[Vector Calculus] Stokes curl={surf_curl:.4}, loop={loop_circ:.4}, |diff|={diff:.2e}; Helmholtz max|div F_sol|={:.2e}",
        summary.max_solenoidal_divergence
    );
}

fn demo_geometric_flows() {
    let sphere = Sphere { radius: 1.5 };
    let discrete_sphere = DiscreteSurface::from_parametric(
        (0.0, TAU),
        (0.14, PI - 0.14),
        18,
        12,
        true,
        false,
        |u, v| sphere.position(u, v),
    );
    let mut mcf = MeanCurvatureFlow::new(discrete_sphere);
    let area0 = mcf.surface.surface_area();
    let dt = mcf.recommended_cfl_dt() * 0.5;
    for _ in 0..8 {
        mcf.step(dt);
    }
    let area1 = mcf.surface.surface_area();

    let torus = Torus {
        major_radius: 2.0,
        minor_radius: 0.7,
    };
    let mut heat =
        HeatEquationSolver::new(&torus, 0.1, (0.0, TAU), (0.0, TAU), (16, 16), |u, v| {
            (-((u - PI).powi(2) + (v - PI).powi(2))).exp()
        });
    let q0 = heat.total_heat();
    heat.step(heat.recommended_cfl_dt() * 0.5);
    println!(
        "[Manifold Flows] Sphere MCF area {area0:.4} -> {area1:.4}; Torus heat Q0={q0:.4} -> Q1={:.4}",
        heat.total_heat()
    );
}

fn demo_riemann_and_residues() {
    let cube_root = NthRootSurface::new(3);
    let z = Complex64::new(1.0, 1.0);
    let s1 = cube_root.next_sheet(0, 1);
    let w0 = cube_root.evaluate(z, 0);
    let w1 = cube_root.evaluate(z, s1);
    let pole = Complex64::new(0.5, -0.25);
    let res = residue_via_integral(|w| (w + 1.0) / (w - pole), pole, 0.1, 180);
    println!(
        "[Complex Analysis] z^(1/3) sheet 0={w0:.4} -> sheet {s1}={w1:.4}; Res((z+1)/(z-z_p))={res:.4}"
    );
}

fn main() {
    demo_vector_calculus();
    demo_geometric_flows();
    demo_riemann_and_residues();
}
