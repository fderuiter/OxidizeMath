#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Initial condition builders and analytical chart verification helpers for [`super::solver::RiemannianSimulationEngine`].

use super::mesh_gen::ManifoldGeometry;
use super::state::{ManifoldKind, RiemannianStudioConfig};
use math_explorer::pure_math::differential_geometry::heat_equation::HeatEquationSolver;
use math_explorer::pure_math::differential_geometry::operators::laplace_beltrami;
use math_explorer::pure_math::differential_geometry::surface::{Sphere, Torus};

#[cfg(feature = "biology")]
use math_explorer::biology::diffusion::CotangentMeshDiffusion;
#[cfg(feature = "biology")]
use math_explorer::biology::morphogenesis::{
    DiffusionCoeff, GrayScottKinetics, SchnakenbergKinetics, TuringSystem,
};
#[cfg(feature = "biology")]
use math_explorer::biology::neuroscience::CorticalSheetRunner;
#[cfg(feature = "biology")]
use math_explorer::math_kernel::types::Dimension;

pub(super) fn seed_heat_field(geom: &ManifoldGeometry, out: &mut [f64]) {
    for (idx, pos) in geom.positions.iter().enumerate() {
        out[idx] = pos[2] + 0.65 * (-2.5 * ((pos[0] - 0.8).powi(2) + pos[1].powi(2))).exp();
    }
}

pub(super) fn evaluate_analytical_chart_verification(
    geom: &ManifoldGeometry,
    cfg: &RiemannianStudioConfig,
    cursor_vertex: usize,
) -> (f64, f64, f64) {
    let idx = cursor_vertex.min(geom.uvs.len().saturating_sub(1));
    let (u_cur, v_cur) = geom.uvs.get(idx).copied().unwrap_or((0.5, 0.8));

    let sphere = Sphere { radius: 1.25 };
    let y10 = |_u: f64, v: f64| v.cos();
    let cursor_lb = if cfg.manifold == ManifoldKind::Torus {
        let torus = Torus {
            major_radius: 1.65,
            minor_radius: 0.62,
        };
        laplace_beltrami(&torus, u_cur, v_cur, &|u, v| u.cos() * v.sin())
    } else {
        laplace_beltrami(&sphere, u_cur, v_cur, &y10)
    };

    let mut solver = HeatEquationSolver::new(
        &sphere,
        cfg.heat_alpha,
        (0.0, std::f64::consts::TAU),
        (0.0, std::f64::consts::PI),
        (12, 12),
        |_u, v| v.cos(),
    );
    solver.step(1e-4);
    let heat_peak = solver
        .u_grid
        .iter()
        .flatten()
        .copied()
        .fold(0.0_f64, f64::max);

    let field_z: Vec<f64> = geom.positions.iter().map(|p| p[2]).collect();
    let mut cot_out = vec![0.0; field_z.len()];
    geom.laplacian.apply(&field_z, &mut cot_out);
    let mut err_sq = 0.0;
    let mut norm_sq = 1e-12;
    for (v, (&approx, &z_val)) in cot_out.iter().zip(field_z.iter()).enumerate() {
        let exact = (-2.0 / (1.25 * 1.25)) * z_val;
        let a = geom.laplacian.vertex_area(v);
        err_sq += a * (approx - exact).powi(2);
        norm_sq += a * exact.powi(2);
    }
    ((err_sq / norm_sq).sqrt(), cursor_lb, heat_peak)
}

#[cfg(feature = "biology")]
pub(super) fn build_schnakenberg_system(
    geom: &ManifoldGeometry,
    cfg: &RiemannianStudioConfig,
) -> TuringSystem<2, SchnakenbergKinetics, CotangentMeshDiffusion> {
    let n = geom.positions.len();
    let u_eq = cfg.schnak_a + cfg.schnak_b;
    let v_eq = cfg.schnak_b / (u_eq * u_eq).max(1e-6);
    let mut sys = TuringSystem::new_with_kinetics(
        Dimension(n),
        DiffusionCoeff(cfg.d_u),
        DiffusionCoeff(cfg.d_v),
        SchnakenbergKinetics::new(cfg.schnak_a, cfg.schnak_b),
        CotangentMeshDiffusion::new(geom.laplacian.clone()),
    );
    for (i, pos) in geom.positions.iter().enumerate() {
        let pert = 0.08 * (6.0 * pos[0]).sin() * (5.0 * pos[1]).cos() + 0.05 * (4.0 * pos[2]).sin();
        sys.u_mut()[i] = (u_eq + pert).max(0.01);
        sys.v_mut()[i] = (v_eq - 0.5 * pert).max(0.01);
    }
    sys
}

#[cfg(feature = "biology")]
pub(super) fn build_gray_scott_system(
    geom: &ManifoldGeometry,
    cfg: &RiemannianStudioConfig,
) -> TuringSystem<2, GrayScottKinetics, CotangentMeshDiffusion> {
    let n = geom.positions.len();
    let mut sys = TuringSystem::new_with_kinetics(
        Dimension(n),
        DiffusionCoeff(cfg.d_u),
        DiffusionCoeff(cfg.d_v * 0.5),
        GrayScottKinetics::new(cfg.gs_f, cfg.gs_k),
        CotangentMeshDiffusion::new(geom.laplacian.clone()),
    );
    let seeds: [[f64; 3]; 4] = [
        [0.0, 0.0, 1.15],
        [0.95, 0.25, 0.25],
        [-0.75, 0.65, -0.25],
        [0.15, -0.95, -0.45],
    ];
    for (i, pos) in geom.positions.iter().enumerate() {
        let near_seed = seeds.iter().any(|s| {
            let d2 = (pos[0] - s[0]).powi(2) + (pos[1] - s[1]).powi(2) + (pos[2] - s[2]).powi(2);
            d2 < 0.16
        });
        if near_seed {
            sys.u_mut()[i] = 0.50;
            sys.v_mut()[i] = 0.25;
        } else {
            sys.u_mut()[i] = 1.0;
            sys.v_mut()[i] = 0.0;
        }
    }
    sys
}

#[cfg(feature = "biology")]
pub(super) fn build_cortical_sheet(
    geom: &ManifoldGeometry,
    cfg: &RiemannianStudioConfig,
    thickness_mm: &[f64],
) -> CorticalSheetRunner {
    let mut runner = CorticalSheetRunner::new(
        geom.laplacian.clone(),
        thickness_mm,
        2.5,
        cfg.hh_conductance,
        -65.0,
    );
    for (i, pos) in geom.positions.iter().enumerate() {
        let cam_p = cfg.camera.rotate_to_camera_space(pos);
        let norm = (cam_p[0] * cam_p[0] + cam_p[1] * cam_p[1] + cam_p[2] * cam_p[2])
            .sqrt()
            .max(1e-6);
        let cos_front = (cam_p[0] * 0.18 + cam_p[1] * 0.15 + cam_p[2] * 0.97) / norm;
        if cos_front > 0.84 {
            runner.depolarize_vertex(i, -50.0);
            runner.set_stimulus(i, cfg.hh_stimulus);
        }
    }
    runner
}
