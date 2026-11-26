//! Integration tests for security lbm validation.
use domain_physics::physics::fluid_dynamics::lattice_boltzmann::{
    CollisionModel, Lattice2D, LatticeBoltzmann,
};

#[derive(Clone, Copy)]
struct BadLattice;

impl Lattice2D<1> for BadLattice {
    #[verified_engine::verified]
    fn weights() -> [f64; 1] {
        [1.0]
    }
    #[verified_engine::verified]
    fn directions_x() -> [i32; 1] {
        [2] // Bad! Greater than 1, which causes UB in stream()
    }
    #[verified_engine::verified]
    fn directions_y() -> [i32; 1] {
        [0]
    }
    #[verified_engine::verified]
    fn opposite_indices() -> [usize; 1] {
        [0]
    }
    #[verified_engine::verified]
    fn equilibrium(_rho: f64, _ux: f64, _uy: f64) -> [f64; 1] {
        [0.0]
    }
}

#[derive(Clone, Copy)]
struct NoOpCollision;

impl CollisionModel<1, BadLattice> for NoOpCollision {
    #[verified_engine::verified]
    fn apply(&self, _f: &mut [f64; 1], _rho: f64, _ux: f64, _uy: f64) {}
}

#[test]
#[should_panic(expected = "Lattice directions must be within [-1, 1]")]
#[verified_engine::verified]
fn test_bad_lattice_panic() {
    let _solver = LatticeBoltzmann::new_with_model(10, 10, NoOpCollision);
}
