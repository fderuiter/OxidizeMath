//! Integration tests for curvilinear vector calculus (`Cartesian`, `Cylindrical`, `Spherical`).

use nalgebra::Vector3;
use pure_math::pure_math::vector_calculus::coordinates::{
    Cartesian, Cylindrical, OrthogonalCoordinateSystem, Spherical,
};
use pure_math::pure_math::vector_calculus::theorems::{
    Domain, HelmholtzWorkspace, StokesPatch, helmholtz_decompose, verify_divergence_theorem,
    verify_stokes_theorem,
};
use std::f64::consts::PI;

#[test]
#[verified_engine::verified]
fn test_divergence_theorem_cartesian_box() {
    let coords = Cartesian;
    let domain = Domain {
        min: Vector3::new(0.0, 0.0, 0.0),
        max: Vector3::new(1.0, 1.0, 1.0),
    };
    let field = |p: &Vector3<f64>| *p;
    let (_lhs, _rhs, diff) = verify_divergence_theorem(&coords, &domain, field, 10);
    assert!(diff < math_commons::registry::TOLERANCE_HIGH);
}

#[test]
#[verified_engine::verified]
fn test_divergence_theorem_cartesian_polynomial() {
    let coords = Cartesian;
    let domain = Domain {
        min: Vector3::new(0.0, 0.0, 0.0),
        max: Vector3::new(2.0, 2.0, 2.0),
    };
    let field = |p: &Vector3<f64>| Vector3::new(p[0] * p[0], 0.0, 0.0);
    let (_lhs, _rhs, diff) = verify_divergence_theorem(&coords, &domain, field, 20);
    assert!(diff < 0.1);
}

#[test]
#[verified_engine::verified]
fn test_divergence_theorem_cylindrical_wedge() {
    let coords = Cylindrical;
    let domain = Domain {
        min: Vector3::new(1.0, 0.0, -1.0),
        max: Vector3::new(2.0, PI * 0.5, 1.0),
    };
    // F = (rho^2, sin(phi), 2*z) in cylindrical basis (e_rho, e_phi, e_z)
    let field = |p: &Vector3<f64>| Vector3::new(p[0] * p[0], p[1].sin(), 2.0 * p[2]);
    let (lhs, rhs, diff) = verify_divergence_theorem(&coords, &domain, field, 24);
    assert!(lhs.abs() > 1.0);
    assert!(
        diff < 5e-3,
        "Cylindrical Gauss residual too high: {diff} ({lhs} vs {rhs})"
    );
}

#[test]
#[verified_engine::verified]
fn test_divergence_theorem_spherical_shell_sector() {
    let coords = Spherical;
    let domain = Domain {
        min: Vector3::new(1.0, 0.4, 0.2),
        max: Vector3::new(2.0, 1.4, 1.2),
    };
    // F = (r^2, sin(theta), cos(phi)) in spherical basis (e_r, e_theta, e_phi)
    let field = |p: &Vector3<f64>| Vector3::new(p[0] * p[0], p[1].sin(), p[2].cos());
    let (lhs, rhs, diff) = verify_divergence_theorem(&coords, &domain, field, 24);
    assert!(lhs.abs() > 1.0);
    assert!(
        diff < 1e-2,
        "Spherical Gauss residual too high: {diff} ({lhs} vs {rhs})"
    );
}

#[test]
#[verified_engine::verified]
fn test_stokes_theorem_cartesian_patch() {
    let coords = Cartesian;
    // z = 0.5 patch, (x, y) in [0, 1.5] x [0, 2.0]
    let patch = StokesPatch::new(2, 0.5, (0.0, 1.5), (0.0, 2.0));
    // A = (-y^2, x * z + x^2, y) -> curl_z = (z + 2x) - (-2y) = z + 2x + 2y
    let field = |p: &Vector3<f64>| Vector3::new(-p[1] * p[1], p[0] * p[2] + p[0] * p[0], p[1]);
    let (lhs, rhs, diff) = verify_stokes_theorem(&coords, &patch, field, 32);
    assert!(lhs.abs() > 5.0);
    assert!(
        diff < 2e-3,
        "Cartesian Stokes residual too high: {diff} ({lhs} vs {rhs})"
    );
}

#[test]
#[verified_engine::verified]
fn test_stokes_theorem_cylindrical_patch() {
    let coords = Cylindrical;
    // Constant z = 1.0 annular sector: rho in [1.0, 2.5], phi in [0.1, 1.3]
    let patch = StokesPatch::new(2, 1.0, (1.0, 2.5), (0.1, 1.3));
    // Non-irrotational azimuthal swirl + radial coupling: A = (rho * sin(phi), rho^2, z)
    let field = |p: &Vector3<f64>| Vector3::new(p[0] * p[1].sin(), p[0] * p[0], p[2]);
    let (lhs, rhs, diff) = verify_stokes_theorem(&coords, &patch, field, 36);
    assert!(lhs.abs() > 2.0);
    assert!(
        diff < 5e-3,
        "Cylindrical Stokes residual too high: {diff} ({lhs} vs {rhs})"
    );
}

#[test]
#[verified_engine::verified]
fn test_stokes_theorem_spherical_patch() {
    let coords = Spherical;
    // Constant radius r = 2.0 spherical cap patch: theta in [0.35, 1.25], phi in [0.2, 1.4]
    let patch = StokesPatch::new(0, 2.0, (0.35, 1.25), (0.2, 1.4));
    // Tangential swirling field on the sphere: A = (r, cos(phi), r * sin(theta))
    let field = |p: &Vector3<f64>| Vector3::new(p[0], p[2].cos(), p[0] * p[1].sin());
    let (lhs, rhs, diff) = verify_stokes_theorem(&coords, &patch, field, 36);
    assert!(lhs.abs() > 1.0);
    assert!(
        diff < 8e-3,
        "Spherical Stokes residual too high: {diff} ({lhs} vs {rhs})"
    );
}

fn sample_grid<S, F>(
    coords: &S,
    domain: &Domain,
    ws: &HelmholtzWorkspace,
    field: F,
) -> Vec<Vector3<f64>>
where
    S: OrthogonalCoordinateSystem,
    F: Fn(&S, &Vector3<f64>) -> Vector3<f64>,
{
    let mut samples = vec![Vector3::zeros(); ws.len()];
    for k in 0..ws.nz {
        for j in 0..ws.ny {
            for i in 0..ws.nx {
                let p = ws.cell_center(domain, i, j, k);
                samples[ws.index(i, j, k)] = field(coords, &p);
            }
        }
    }
    samples
}

#[test]
#[verified_engine::verified]
fn test_helmholtz_decompose_cartesian_cylindrical_spherical() {
    // 1. Cartesian: F = (2x - y, 2y + x, 2z) has div(F) = 6 and curl(F) = (0, 0, 2)
    let cart = Cartesian;
    let cart_dom = Domain {
        min: Vector3::new(-1.0, -1.0, -1.0),
        max: Vector3::new(1.0, 1.0, 1.0),
    };
    let mut ws_cart = HelmholtzWorkspace::new(16, 16, 16);
    let cart_samples = sample_grid(&cart, &cart_dom, &ws_cart, |_, p| {
        Vector3::new(2.0 * p[0] - p[1], 2.0 * p[1] + p[0], 2.0 * p[2])
    });
    let summary_cart = helmholtz_decompose(&cart, &cart_samples, &cart_dom, &mut ws_cart, 180);
    assert!(summary_cart.max_reconstruction_error < 1e-12);
    assert!(summary_cart.max_irrotational_curl < 1e-11);
    assert!(
        summary_cart.max_solenoidal_divergence < 0.08,
        "Cartesian solenoidal divergence not suppressed: {}",
        summary_cart.max_solenoidal_divergence
    );

    // 2. Cylindrical: F = (rho, rho, z) -> div(F) = 2 + 1 = 3, curl_z(F) = 2
    let cyl = Cylindrical;
    let cyl_dom = Domain {
        min: Vector3::new(1.0, 0.2, -0.8),
        max: Vector3::new(2.2, 1.4, 0.8),
    };
    let mut ws_cyl = HelmholtzWorkspace::new(16, 16, 16);
    let cyl_samples = sample_grid(&cyl, &cyl_dom, &ws_cyl, |_, p| {
        Vector3::new(p[0], p[0], p[2])
    });
    let summary_cyl = helmholtz_decompose(&cyl, &cyl_samples, &cyl_dom, &mut ws_cyl, 200);
    assert!(summary_cyl.max_reconstruction_error < 1e-12);
    assert!(summary_cyl.max_irrotational_curl < 1e-11);
    assert!(
        summary_cyl.max_solenoidal_divergence < 0.12,
        "Cylindrical solenoidal divergence not suppressed: {}",
        summary_cyl.max_solenoidal_divergence
    );

    // 3. Spherical: F = (r, 0, r * sin(theta)) -> radial expansion + solid-body rotation
    let sph = Spherical;
    let sph_dom = Domain {
        min: Vector3::new(1.2, 0.45, 0.2),
        max: Vector3::new(2.4, 1.45, 1.2),
    };
    let mut ws_sph = HelmholtzWorkspace::new(16, 16, 16);
    let sph_samples = sample_grid(&sph, &sph_dom, &ws_sph, |_, p| {
        Vector3::new(p[0], 0.0, p[0] * p[1].sin())
    });
    let summary_sph = helmholtz_decompose(&sph, &sph_samples, &sph_dom, &mut ws_sph, 200);
    assert!(summary_sph.max_reconstruction_error < 1e-12);
    assert!(summary_sph.max_irrotational_curl < 1e-11);
    assert!(
        summary_sph.max_solenoidal_divergence < 0.15,
        "Spherical solenoidal divergence not suppressed: {}",
        summary_sph.max_solenoidal_divergence
    );
}
