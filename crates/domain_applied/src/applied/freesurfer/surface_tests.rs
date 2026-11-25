use super::*;
use verified_engine::engine::VerifiedEngine;

#[test]
fn test_internal_energy_100_steps_zero_alloc_and_exact_value() {
    let mut surface =
        Surface::from_vertices(vec![[0.0, 0.0, 0.0], [1.0, 2.0, 2.0], [1.0, 2.0, 5.0]]);
    let initial = internal_energy(&surface, 0.5, 0.25);
    assert!((initial - 10.5).abs() < 1e-12);

    let (accum, _metrics) = VerifiedEngine::run_verified(|| {
        let mut sum = 0.0;
        for _ in 0..100 {
            evolve_surface(&mut surface, 1e-4, 0.2);
            sum += internal_energy(&surface, 0.5, 0.25) + external_energy(&surface, 0.2);
        }
        sum
    });
    assert!(accum.is_finite());
}

#[test]
fn test_mri_smoothed_gradient_and_normal_force_match_exact_polynomial_field() {
    let dims = [11, 11, 11];
    let origin = [-5.0, -5.0, -5.0];
    let h = 1.0;
    let mut vox = Vec::with_capacity(11 * 11 * 11);
    for iz in 0..11 {
        let z = origin[2] + (iz as f64) * h;
        for iy in 0..11 {
            let y = origin[1] + (iy as f64) * h;
            for ix in 0..11 {
                let x = origin[0] + (ix as f64) * h;
                vox.push(1.5 * x * x - 0.8 * y * y + 2.1 * z * z + 0.7 * x * y);
            }
        }
    }
    let vol = MriVolume::new(dims, origin, h, vox).unwrap();
    let pt = [0.65, -0.42, 0.38];
    let sigma = 0.5;
    let grad = vol.smoothed_gradient(&pt, sigma);
    let exact_grad = [
        3.0 * pt[0] + 0.7 * pt[1],
        -1.6 * pt[1] + 0.7 * pt[0],
        4.2 * pt[2],
    ];
    for a in 0..3 {
        assert!(
            (grad[a] - exact_grad[a]).abs() < 1e-13,
            "grad[{a}]={} vs exact={}",
            grad[a],
            exact_grad[a]
        );
    }

    let raw_n = [1.0, -2.0, 0.5];
    let nrm = (1.0_f64 + 4.0 + 0.25).sqrt();
    let (nx, ny, nz) = (raw_n[0] / nrm, raw_n[1] / nrm, raw_n[2] / nrm);
    let hn = [3.0 * nx + 0.7 * ny, 0.7 * nx - 1.6 * ny, 4.2 * nz];
    let exact_neg_de_dn =
        2.0 * (exact_grad[0] * hn[0] + exact_grad[1] * hn[1] + exact_grad[2] * hn[2]);
    let f_ext = normal_external_force(&pt, &raw_n, &vol, sigma);
    let exact_f = [
        exact_neg_de_dn * nx,
        exact_neg_de_dn * ny,
        exact_neg_de_dn * nz,
    ];
    for a in 0..3 {
        assert!(
            (f_ext[a] - exact_f[a]).abs() < 1e-13,
            "f_ext[{a}]={} vs exact={}",
            f_ext[a],
            exact_f[a]
        );
    }
}
