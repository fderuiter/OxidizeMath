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
