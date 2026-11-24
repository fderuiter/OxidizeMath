//! Integration tests verifying zero-allocation hot-path stepping under `VerifiedAllocator`
//! across 100 steps of `CotangentMeshDiffusion` (`SchnakenbergKinetics` & `GrayScottKinetics`)
//! and `CorticalSheetRunner` (`HodgkinHuxleyNeuron` + FreeSurfer `cortical_thickness_batch`).

use domain_applied::applied::freesurfer::{Surface, cortical_thickness_batch};
use domain_biology::biology::diffusion::{CotangentMeshDiffusion, SpatialDiffusion};
use domain_biology::biology::morphogenesis::{
    DiffusionCoeff, GrayScottKinetics, SchnakenbergKinetics, TuringSystem,
};
use domain_biology::biology::neuroscience::CorticalSheetRunner;
use domain_biology::biology::reaction_diffusion::{ChemicalState, DiffusionModel};
use math_commons::math_kernel::types::Dimension;
use pure_math::pure_math::differential_geometry::CotangentLaplacian;
use std::f64::consts::PI;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

struct CorticalShellFixture {
    white_verts: Vec<[f64; 3]>,
    pial_verts: Vec<[f64; 3]>,
    tris: Vec<[usize; 3]>,
}

fn build_cortical_shell_fixture() -> CorticalShellFixture {
    let (nu, nv) = (14, 10);
    let mut white_verts = vec![[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
    let mut pial_verts = vec![[0.0, 0.0, 1.03], [0.0, 0.0, -1.02]];
    for j in 1..(nv - 1) {
        let (sv, cv) = (PI * (j as f64) / ((nv - 1) as f64)).sin_cos();
        for i in 0..nu {
            let u = 2.0 * PI * (i as f64) / (nu as f64);
            let (su, cu) = u.sin_cos();
            let dir = [sv * cu, sv * su, cv];
            let scale = 1.025 + 0.01 * (3.0 * u).cos();
            white_verts.push(dir);
            pial_verts.push([dir[0] * scale, dir[1] * scale, dir[2] * scale]);
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
    CorticalShellFixture {
        white_verts,
        pial_verts,
        tris,
    }
}

#[test]
fn test_cotangent_diffusion_and_cortical_sheet_100_steps_verified_allocator() {
    let fx = build_cortical_shell_fixture();
    let n = fx.white_verts.len();
    let lap = CotangentLaplacian::from_indexed(&fx.white_verts, &fx.tris);
    let diff_mesh = CotangentMeshDiffusion::new(lap.clone());

    let mut sys_schnak = TuringSystem::new_with_kinetics(
        Dimension(n),
        DiffusionCoeff(0.004),
        DiffusionCoeff(0.08),
        SchnakenbergKinetics::new(0.1, 0.9),
        diff_mesh.clone(),
    );
    let mut sys_gs = TuringSystem::new_with_kinetics(
        Dimension(n),
        DiffusionCoeff(0.002),
        DiffusionCoeff(0.001),
        GrayScottKinetics::new(0.0367, 0.0649),
        diff_mesh.clone(),
    );
    for (idx, p) in fx.white_verts.iter().enumerate() {
        sys_schnak.u_mut()[idx] = 1.0 + 0.05 * p[2];
        sys_schnak.v_mut()[idx] = 0.9 - 0.04 * p[0];
        sys_gs.u_mut()[idx] = 1.0;
        sys_gs.v_mut()[idx] = if p[2] > 0.7 { 0.25 } else { 0.0 };
    }

    let chem_in = ChemicalState::new(2, n);
    let mut chem_out = ChemicalState::new(2, n);
    let white_surf = Surface::from_vertices(fx.white_verts);
    let pial_surf = Surface::from_vertices(fx.pial_verts);
    let thickness = cortical_thickness_batch(&white_surf, &pial_surf);
    let mut sheet = CorticalSheetRunner::new(lap, &thickness, 0.025, 0.28, -65.0);
    sheet.depolarize_vertex(0, 15.0);
    sheet.set_stimulus(0, 18.0);

    let (accum_lap, metrics) = VerifiedEngine::run_verified(|| {
        DiffusionModel::apply(&diff_mesh, &chem_in, &mut chem_out, &[0.01, 0.02]);
        let mut sum_lap = 0.0;
        diff_mesh.map_diffusion(
            [sys_schnak.u(), sys_schnak.v()],
            [0.004, 0.08],
            |_v, _cur, diffs| {
                sum_lap += diffs[0] + diffs[1];
            },
        );
        for _ in 0..100 {
            sys_schnak.step(1e-3);
            sys_gs.step(1e-2);
            sheet.step(0.02);
        }
        sum_lap
    });

    assert!(accum_lap.is_finite());
    assert!(metrics.function_calls >= 300);
    assert!(sys_schnak.u().iter().all(|v| v.is_finite()));
    assert!(sys_gs.v().iter().all(|v| v.is_finite()));
    assert!(sheet.potentials().iter().all(|v| v.is_finite()));
}
