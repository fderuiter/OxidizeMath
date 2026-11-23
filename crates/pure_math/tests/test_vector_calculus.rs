//! Integration tests for test vector calculus.
use nalgebra::Vector3;
use pure_math::pure_math::vector_calculus::coordinates::Cartesian;
use pure_math::pure_math::vector_calculus::theorems::{Domain, verify_divergence_theorem};

#[test]
#[verified_engine::verified]
fn test_divergence_theorem_cartesian_box() {
    let coords = Cartesian;
    let domain = Domain {
        min: Vector3::new(0.0, 0.0, 0.0),
        max: Vector3::new(1.0, 1.0, 1.0),
    };

    // Field F = (x, y, z). Divergence = 1+1+1 = 3.
    // Volume Integral = 3 * Volume = 3 * 1 = 3.
    // Flux:
    // x=1: F=(1,y,z). n=(1,0,0). F.n = 1. Area=1. Flux=1.
    // x=0: F=(0,y,z). n=(-1,0,0). F.n = 0.
    // y=1: F=(x,1,z). n=(0,1,0). F.n = 1. Area=1. Flux=1.
    // y=0: F=(x,0,z). n=(0,-1,0). F.n = 0.
    // z=1: F=(x,y,1). n=(0,0,1). F.n = 1. Area=1. Flux=1.
    // z=0: F=(x,y,0). n=(0,0,-1). F.n = 0.
    // Total Flux = 3.

    let field = |p: &Vector3<f64>| *p;

    let (lhs, rhs, diff) = verify_divergence_theorem(&coords, &domain, field, 10);

    println!("LHS (Vol): {}, RHS (Surf): {}, Diff: {}", lhs, rhs, diff);
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

    // Field F = (x^2, 0, 0). Div = 2x.
    // Vol Int = \int_0^2 \int_0^2 \int_0^2 2x dx dy dz = 2 * 2 * [x^2]_0^2 = 4 * 4 = 16.

    // Flux:
    // x=2: F=(4, 0, 0). n=(1,0,0). Flux = 4 * Area = 4 * 4 = 16.
    // x=0: F=(0, 0, 0). Flux = 0.
    // y, z faces: F.n = 0.
    // Total Flux = 16.

    let field = |p: &Vector3<f64>| Vector3::new(p[0] * p[0], 0.0, 0.0);

    let (lhs, rhs, diff) = verify_divergence_theorem(&coords, &domain, field, 20);
    println!("LHS (Vol): {}, RHS (Surf): {}, Diff: {}", lhs, rhs, diff);

    // Numerical integration error will exist.
    assert!(diff < 0.1);
}

#[test]
#[verified_engine::verified]
fn test_spherical_gradient_divergence_curl_and_laplacian_autodiff() {
    use math_commons::ast::{Dual, Jet2};
    use pure_math::pure_math::vector_calculus::coordinates::Spherical;
    use pure_math::pure_math::vector_calculus::operators::{
        curl_dual, divergence_dual, gradient_dual, laplacian_jet2,
    };

    let coords = Spherical;
    let pt = Vector3::new(2.5, 0.9, 1.3);
    let (r, theta, phi) = (pt[0], pt[1], pt[2]);

    // Harmonic monopole 1/r has exact spherical Laplacian = 0 for r > 0
    let lap_monopole = laplacian_jet2(&coords, |p: &[Jet2<3>; 3]| p[0].recip(), &pt);
    assert!(lap_monopole.abs() < 1e-14, "got {lap_monopole:e}");

    // Gradient of r^2 * cos(theta) = [2 r cos(theta), -r sin(theta), 0]
    let g = gradient_dual(&coords, |p: &[Dual<f64>; 3]| p[0] * p[0] * p[1].cos(), &pt);
    assert!((g[0] - 2.0 * r * theta.cos()).abs() < 1e-14);
    assert!((g[1] + r * theta.sin()).abs() < 1e-14);
    assert!(g[2].abs() < 1e-14);

    // Curl of gradient of Phi = r^2 cos(theta) sin(phi) is identically zero to machine precision
    let irrot = curl_dual(
        &coords,
        |p: &[Dual<f64>; 3]| {
            let (pr, pt_a, pp) = (p[0], p[1], p[2]);
            [
                Dual::constant(2.0) * pr * pt_a.cos() * pp.sin(),
                -pr * pt_a.sin() * pp.sin(),
                pr * (pt_a.cos() / pt_a.sin()) * pp.cos(),
            ]
        },
        &pt,
    );
    assert!(irrot.norm() < 1e-13, "curl(grad) = {irrot:?}");

    // Radial field A = (r, 0, 0) in spherical coordinates has div(A) = (1/r^2) d/dr(r^3) = 3
    let div_radial = divergence_dual(
        &coords,
        |p: &[Dual<f64>; 3]| [p[0], Dual::ZERO, Dual::ZERO],
        &pt,
    );
    assert!(
        (div_radial - 3.0).abs() < 1e-14,
        "div(r e_r) = {div_radial}"
    );
    let _ = phi;
}
