//! Integration tests verifying zero-allocation hot-path execution of FreeSurfer
//! `internal_energy`, `external_energy`, and `evolve_surface` under `VerifiedAllocator` across 100 steps.

use domain_applied::applied::freesurfer::{
    GlmDesignQr, Surface, ThicknessWorkspace, cortical_thickness_batch,
    cortical_thickness_map_into, estimate_beta_with_workspace, evolve_surface, external_energy,
    glm_workspace_bytes, internal_energy, t_statistic_with_workspace,
};
use math_memory::OwnedByteArena;
use nalgebra::{DMatrix, DVector};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_freesurfer_internal_energy_and_surface_evolution_100_steps_verified_allocator() {
    let mut white_verts = Vec::with_capacity(64);
    let mut pial_verts = Vec::with_capacity(64);
    let mut faces = Vec::with_capacity(62);
    for i in 0..64 {
        let theta = (i as f64) * 0.1;
        white_verts.push([10.0 * theta.cos(), 10.0 * theta.sin(), (i as f64) * 0.05]);
        pial_verts.push([12.5 * theta.cos(), 12.5 * theta.sin(), (i as f64) * 0.05]);
        if i + 2 < 64 {
            faces.push([i, i + 1, i + 2]);
        }
    }
    let white_surface = Surface::new(white_verts, faces.clone());
    let mut pial_surface = Surface::new(pial_verts, faces);
    let batch = cortical_thickness_batch(&white_surface, &pial_surface);
    assert_eq!(batch.len(), 64);
    assert!(batch.iter().all(|&t| t > 2.0 && t.is_finite()));

    let mut workspace = ThicknessWorkspace::with_max_vertices(64);
    let mut thick_buf = [0.0_f64; 64];

    let (energy_accum, metrics) = VerifiedEngine::run_verified(|| {
        let mut total = 0.0;
        for _ in 0..100 {
            evolve_surface(&mut pial_surface, 1e-4, 0.25);
            let n = cortical_thickness_map_into(
                &white_surface,
                &pial_surface,
                &mut workspace,
                &mut thick_buf,
            );
            total += internal_energy(&pial_surface, 0.4, 0.2)
                + external_energy(&pial_surface, 0.25)
                + thick_buf[n.saturating_sub(1)];
        }
        total
    });

    assert!(energy_accum.is_finite());
    assert!(metrics.function_calls >= 300);
}

#[test]
fn test_glm_with_workspace_zero_heap_allocations() {
    let x = DMatrix::from_row_slice(5, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let y = DVector::from_column_slice(&[2.0, 3.5, 5.0, 6.5, 8.0]);
    let c = DVector::from_column_slice(&[0.0, 1.0]);
    let mut out_beta = DVector::zeros(2);
    let mut design_qr = GlmDesignQr::<f64>::with_capacity(5, 2);
    let mut arena = OwnedByteArena::with_capacity(glm_workspace_bytes::<f64>(5, 2));

    let ((t_direct, t_cached), _) = VerifiedEngine::run_verified(|| {
        estimate_beta_with_workspace(&x, &y, &mut out_beta, &mut arena).unwrap();
        let t_direct = t_statistic_with_workspace(&c, &out_beta, &x, 0.01, &mut arena).unwrap();
        design_qr
            .factor_into_with_workspace(&x, &mut arena)
            .unwrap();
        design_qr
            .estimate_beta_with_workspace(&y, &mut out_beta, &mut arena)
            .unwrap();
        let t_cached = design_qr
            .t_statistic_with_workspace(&c, &out_beta, 0.01, &mut arena)
            .unwrap();
        (t_direct, t_cached)
    });
    assert!((out_beta[0] - 2.0).abs() < 1e-12);
    assert!((out_beta[1] - 1.5).abs() < 1e-12);
    assert!((t_direct - t_cached).abs() < 1e-12);
}
