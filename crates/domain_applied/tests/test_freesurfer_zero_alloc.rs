//! Integration tests verifying zero-allocation hot-path execution of FreeSurfer
//! `internal_energy`, `external_energy`, and `evolve_surface` under `VerifiedAllocator` across 100 steps.

use domain_applied::applied::freesurfer::{
    Surface, ThicknessWorkspace, cortical_thickness_batch, cortical_thickness_map_into,
    evolve_surface, external_energy, internal_energy,
};
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
