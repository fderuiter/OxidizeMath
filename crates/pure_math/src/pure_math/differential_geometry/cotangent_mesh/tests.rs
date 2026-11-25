use super::*;
use std::f64::consts::PI;
use verified_engine::engine::VerifiedEngine;

fn build_welded_unit_sphere(nu: usize, nv: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut positions = Vec::with_capacity(2 + (nv - 2) * nu);
    positions.push([0.0, 0.0, 1.0]); // North pole (v = 0)
    positions.push([0.0, 0.0, -1.0]); // South pole (v = PI)
    for j in 1..(nv - 1) {
        let v = PI * (j as f64) / ((nv - 1) as f64);
        let (sv, cv) = v.sin_cos();
        for i in 0..nu {
            let u = 2.0 * PI * (i as f64) / (nu as f64);
            let (su, cu) = u.sin_cos();
            positions.push([sv * cu, sv * su, cv]);
        }
    }
    let ring_idx = |i: usize, j: usize| -> usize { 2 + (j - 1) * nu + (i % nu) };
    let mut tris = Vec::new();
    for i in 0..nu {
        tris.push([0, ring_idx(i, 1), ring_idx(i + 1, 1)]);
    }
    for j in 1..(nv - 2) {
        for i in 0..nu {
            let a = ring_idx(i, j);
            let b = ring_idx(i + 1, j);
            let c = ring_idx(i + 1, j + 1);
            let d = ring_idx(i, j + 1);
            tris.push([a, d, c]);
            tris.push([a, c, b]);
        }
    }
    let last_ring = nv - 2;
    for i in 0..nu {
        tris.push([ring_idx(i, last_ring), 1, ring_idx(i + 1, last_ring)]);
    }
    (positions, tris)
}

fn spherical_harmonic_l2_error(nu: usize, nv: usize, l: usize) -> f64 {
    let (pos, tris) = build_welded_unit_sphere(nu, nv);
    let lap = CotangentLaplacian::from_indexed(&pos, &tris);
    let eigenval = -((l * (l + 1)) as f64);
    let field: Vec<f64> = pos
        .iter()
        .map(
            |&[_x, _y, z]| {
                if l == 1 { z } else { 0.5 * (3.0 * z * z - 1.0) }
            },
        )
        .collect();
    let mut out = vec![0.0; pos.len()];
    lap.apply(&field, &mut out);

    let mut err_sq = 0.0;
    let mut norm_sq = 0.0;
    for (i, (&approx, &u)) in out.iter().zip(field.iter()).enumerate() {
        assert!(approx.is_finite(), "Non-finite value at vertex {i}");
        let area = lap.vertex_area(i);
        let exact = eigenval * u;
        err_sq += area * (approx - exact).powi(2);
        norm_sq += area * exact.powi(2);
    }
    (err_sq / norm_sq).sqrt()
}

#[test]
fn test_spherical_harmonics_l2_convergence_and_polar_stability() {
    let err_l1_coarse = spherical_harmonic_l2_error(24, 16, 1);
    let err_l1_fine = spherical_harmonic_l2_error(48, 32, 1);
    assert!(
        err_l1_fine < err_l1_coarse * 0.5,
        "l=1 failed to converge: {err_l1_coarse} -> {err_l1_fine}"
    );
    assert!(
        err_l1_fine < 0.02,
        "l=1 fine relative L2 error too high: {err_l1_fine}"
    );

    let err_l2_coarse = spherical_harmonic_l2_error(24, 16, 2);
    let err_l2_fine = spherical_harmonic_l2_error(48, 32, 2);
    assert!(
        err_l2_fine < err_l2_coarse * 0.5,
        "l=2 failed to converge: {err_l2_coarse} -> {err_l2_fine}"
    );
    assert!(
        err_l2_fine < 0.04,
        "l=2 fine relative L2 error too high: {err_l2_fine}"
    );
}

#[test]
fn test_cotangent_laplacian_100_steps_zero_alloc_and_mass_conservation() {
    let (pos, tris) = build_welded_unit_sphere(24, 16);
    let lap = CotangentLaplacian::from_indexed(&pos, &tris);
    let mut u: Vec<f64> = pos.iter().map(|p| p[2] + 0.5 * p[0] * p[1]).collect();
    let mut next = vec![0.0; pos.len()];
    let mut mean_curv = vec![0.0; pos.len()];

    let initial_mass: f64 = u
        .iter()
        .enumerate()
        .map(|(i, &val)| lap.vertex_area(i) * val)
        .sum();

    let ((), _metrics) = VerifiedEngine::run_verified(|| {
        lap.compute_mean_curvature(&pos, &mut mean_curv);
        for _ in 0..100 {
            lap.step_heat_euler(&u, &mut next, 0.05, 1e-3);
            std::mem::swap(&mut u, &mut next);
        }
    });

    let final_mass: f64 = u
        .iter()
        .enumerate()
        .map(|(i, &val)| lap.vertex_area(i) * val)
        .sum();
    assert!((final_mass - initial_mass).abs() < 1e-10);
    assert!(
        mean_curv
            .iter()
            .all(|h| h.is_finite() && (*h - 1.0).abs() < 0.1)
    );
}

#[test]
fn test_degenerate_obtuse_and_isolated_vertices_clamp_without_nan() {
    let pos = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.05, 0.0], // obtuse at vertex 2 (> 160 deg)
        [2.0, 0.0, 0.0],  // collinear with 0 and 1
        [9.0, 9.0, 9.0],  // isolated vertex
    ];
    let tris = [[0, 1, 2], [0, 1, 3], [0, 0, 1]];
    let lap = CotangentLaplacian::from_indexed(&pos, &tris);
    for v in 0..pos.len() {
        assert!(lap.vertex_area(v) >= 1e-12 && lap.vertex_area(v).is_finite());
    }
    let field = [1.0, 2.0, -0.5, 0.25, 3.0];
    let mut out = [0.0; 5];
    let mut next = [0.0; 5];
    let mut curv = [0.0; 5];
    lap.apply(&field, &mut out);
    lap.step_heat_euler(&field, &mut next, 0.1, 1e-3);
    lap.compute_mean_curvature(&pos, &mut curv);
    assert!(out.iter().all(|v| v.is_finite()) && out[4] == 0.0);
    assert!(next.iter().all(|v| v.is_finite()));
    assert!(curv.iter().all(|v| v.is_finite()));
}
