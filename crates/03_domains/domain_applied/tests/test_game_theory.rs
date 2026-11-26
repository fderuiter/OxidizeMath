//! Integration tests for test game theory.
use domain_applied::applied::game_theory::equilibrium::{
    BestResponseCorrespondence, BoxSet, ConvexSet, is_fixed_point,
};
use domain_applied::applied::game_theory::evolutionary::ReplicatorDynamics;
use domain_applied::applied::game_theory::mean_field::{
    Density, FixedPointSolver, MFGConfigBuilder, MFGSolver, Position,
};
use domain_applied::applied::game_theory::mechanism_design::optimal_reserve_price;
use nalgebra::{DMatrix, DVector};
use statrs::distribution::Uniform;
use std::num::NonZeroUsize;

#[test]
#[verified_engine::verified]
fn test_equilibrium_integration() {
    let box_set = BoxSet::new(vec![0.0], vec![1.0]);
    assert!(box_set.contains(&DVector::from_vec(vec![0.5])));

    // Check trivial fixed point
    let correspondence = BestResponseCorrespondence {
        mapping: Box::new(|x| x.clone()),
        tolerance: math_commons::registry::TOLERANCE_FAST,
    };
    assert!(is_fixed_point(
        &correspondence,
        &DVector::from_vec(vec![0.5])
    ));
}

#[test]
#[verified_engine::verified]
fn test_mean_field_integration() {
    let config = MFGConfigBuilder::new()
        .viscosity(0.1)
        .time_horizon(1.0)
        .grid_points(NonZeroUsize::new(10).unwrap())
        .time_steps(NonZeroUsize::new(10).unwrap())
        .space_bounds(-1.0, 1.0)
        .expect("Space bounds must be valid")
        .build()
        .expect("Config should build");

    let solver = FixedPointSolver::new(2);
    let _res = solver.solve(
        &config,
        &|_p: Position, d: Density| d.0 * d.0,
        &|_p: Position, _d: Density| 0.0,
        &|_p: Position| 1.0,
    );
}

#[test]
#[verified_engine::verified]
fn test_evolutionary_integration() {
    let payoff = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 5.0, 1.0]); // Prisoner's Dilemma-ish
    let system = ReplicatorDynamics::new(payoff).unwrap();
    let traj = system.simulate(DVector::from_vec(vec![0.5, 0.5]), 1.0, 0.1);
    assert!(!traj.is_empty());
}

#[test]
#[verified_engine::verified]
fn test_mechanism_integration() {
    let dist = Uniform::new(0.0, 1.0).unwrap();
    let r = optimal_reserve_price(&dist, 0.0, 1.0);
    assert!((r - 0.5).abs() < 1e-2);
}
