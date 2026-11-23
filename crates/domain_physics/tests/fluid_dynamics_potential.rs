//! Integration tests for fluid dynamics potential.
use domain_physics::physics::fluid_dynamics::potential_flow::{
    Doublet, FlowElement, PotentialFlowField, UniformFlow,
};
use nalgebra::Vector2;
use std::f64::consts::PI;

#[test]
#[verified_engine::verified]
fn test_cylinder_flow() {
    let u_inf = 10.0;
    let radius = 2.0;

    // Create flow field
    let mut field = PotentialFlowField::new();

    // 1. Uniform Flow in +X direction
    field.add(Box::new(UniformFlow::new(u_inf, 0.0)));

    // 2. Doublet at origin to simulate cylinder
    // Strength kappa = 2 * pi * U * R^2
    let kappa = 2.0 * PI * u_inf * radius * radius;
    field.add(Box::new(Doublet::new(kappa, 0.0, 0.0)));

    // Check stagnation points at (-R, 0) and (R, 0)
    let stagnation_point = Vector2::new(-radius, 0.0);
    let velocity = field.velocity(stagnation_point.x, stagnation_point.y);

    println!("Velocity at stagnation point (-R, 0): {:?}", velocity);
    assert!(
        velocity.norm() < 1e-5,
        "Velocity at stagnation point should be zero"
    );

    let stagnation_point_2 = Vector2::new(radius, 0.0);
    let velocity_2 = field.velocity(stagnation_point_2.x, stagnation_point_2.y);

    println!("Velocity at stagnation point (R, 0): {:?}", velocity_2);
    assert!(
        velocity_2.norm() < 1e-5,
        "Velocity at stagnation point should be zero"
    );

    // Check point on top of cylinder (0, R)
    // Velocity should be 2 * U_inf in X direction (tangential)
    // Actually, v_theta = - (1/r) dphi/dtheta
    // phi = U(r + R^2/r)cos(theta)
    // dphi/dtheta = -U(r + R^2/r)sin(theta)
    // v_theta = U(1 + R^2/r^2)sin(theta)
    // At r=R, theta=pi/2 (top): v_theta = U(1+1)*1 = 2U.
    // Direction is -X because flow goes around.
    // Wait, let's check vector.
    // At (0, R), Uniform gives (U, 0).
    // Doublet: u = kappa/2pi * (x^2-y^2)/r^4 = kappa/2pi * (-R^2)/R^4 = -kappa/(2pi R^2)
    // kappa = 2 pi U R^2 -> u = - (2 pi U R^2) / (2 pi R^2) = -U.
    // Total u = U - U = 0?
    // Wait. My Doublet u formula:
    // u = kappa/(2pi) * (x^2 - y^2)/r^4 (from previous thought process, let's verify code)

    // Code says:
    // u_num = strength * (dx*dx - dy*dy)
    // u = -u_num / (2pi r^4)
    // At (0, R): dx=0, dy=R. u_num = strength * (-R^2).
    // u = - (strength * -R^2) / (2pi R^4) = strength / (2pi R^2).
    // strength = 2 pi U R^2.
    // u = (2 pi U R^2) / (2 pi R^2) = U.
    // Total u = U (uniform) + U (doublet) = 2U.
    // This seems correct for inviscid flow over a cylinder.

    let top_point = Vector2::new(0.0, radius);
    let v_top = field.velocity(top_point.x, top_point.y);
    println!("Velocity at top point (0, R): {:?}", v_top);

    assert!(
        (v_top.x - 2.0 * u_inf).abs() < 1e-5,
        "Tangential velocity at top should be 2*U"
    );
    assert!(v_top.y.abs() < 1e-5, "Radial velocity at top should be 0");
}
