use super::state::LatticeState;
use std::marker::PhantomData;
use verified_engine::Theory;

/// Trait defining the lattice geometry and weights.
///
/// This uses the **Strategy Pattern** to decouple the solver logic from the
/// specific lattice arrangement (D2Q9, D2Q5, etc.).
pub trait Lattice2D<const Q: usize>: Copy + Clone + Send + Sync + 'static {
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn weights() -> [f64; Q];
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn directions_x() -> [i32; Q];
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn directions_y() -> [i32; Q];
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn opposite_indices() -> [usize; Q];
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn equilibrium(rho: f64, ux: f64, uy: f64) -> [f64; Q];
}

/// D2Q9 Lattice Model.
///
/// Standard 9-velocity lattice for 2D fluid simulation.
#[derive(Debug, Clone, Copy)]
pub struct D2Q9;

impl Lattice2D<9> for D2Q9 {
    #[inline(always)]
    #[verified_engine::verified]
    fn weights() -> [f64; 9] {
        [
            4.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
        ]
    }

    #[inline(always)]
    #[verified_engine::verified]
    fn directions_x() -> [i32; 9] {
        [0, 1, 0, -1, 0, 1, -1, -1, 1]
    }

    #[inline(always)]
    #[verified_engine::verified]
    fn directions_y() -> [i32; 9] {
        [0, 0, 1, 0, -1, 1, 1, -1, -1]
    }

    #[inline(always)]
    #[verified_engine::verified]
    fn opposite_indices() -> [usize; 9] {
        [0, 3, 4, 1, 2, 7, 8, 5, 6]
    }

    #[inline(always)]
    #[verified_engine::verified]
    fn equilibrium(rho: f64, ux: f64, uy: f64) -> [f64; 9] {
        let mut eq = [0.0; 9];
        let u2 = ux * ux + uy * uy;
        let cx = Self::directions_x();
        let cy = Self::directions_y();
        let w = Self::weights();

        for k in 0..9 {
            let cu = (cx[k] as f64 * ux) + (cy[k] as f64 * uy);
            eq[k] = rho * w[k] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        }
        eq
    }
}

/// Trait defining the collision operator strategy.
pub trait CollisionModel<const Q: usize, L: Lattice2D<Q>> {
    /// Applies the collision operator to the distribution function `f`.
    #[verified_engine::verified]
    fn apply(&self, f: &mut [f64; Q], rho: f64, ux: f64, uy: f64);
}

/// BGK Collision Model.
#[derive(Debug, Clone, Copy, Theory)]
#[theory(
    description = "Bhatnagar-Gross-Krook (BGK) collision operator for single relaxation time.",
    citation = "Bhatnagar, P. L., Gross, E. P., & Krook, M. (1954)."
)]
pub struct BgkCollision {
    /// Relaxation time.
    #[theory(min = 0.515, max = 1.1, step = 0.005)]
    pub tau: f64,
}

impl<const Q: usize, L: Lattice2D<Q>> CollisionModel<Q, L> for BgkCollision {
    #[inline(always)]
    #[verified_engine::verified]
    fn apply(&self, f: &mut [f64; Q], rho: f64, ux: f64, uy: f64) {
        let omega = 1.0 / self.tau;
        let eq = L::equilibrium(rho, ux, uy);
        for k in 0..Q {
            f[k] = (1.0 - omega) * f[k] + omega * eq[k];
        }
    }
}

/// Generic Lattice Boltzmann Solver.
///
/// Implements the discrete streaming and collision steps to simulate fluid dynamics.
///
/// # Examples
///
/// ```rust
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{LatticeBoltzmannD2Q9, BgkCollision};
///
/// // Create a 20x10 lattice with a relaxation time (tau) of 1.0.
/// // Tau determines the kinematic viscosity: nu = (tau - 0.5) / 3.
/// let mut solver = LatticeBoltzmannD2Q9::new(20, 10, 1.0);
///
/// // Set an obstacle at coordinates (10, 5)
/// solver.set_obstacle(10, 5, true);
///
/// // Set a continuous inlet velocity on the left edge (x=0)
/// solver.set_inlet(0, 4, 1, 2, 0.1, 0.0);
///
/// // Perform a single simulation step
/// solver.step();
///
/// // Velocity inside an obstacle is strictly enforced to 0.0
/// let (ux, uy) = solver.get_velocity(10, 5);
/// assert_eq!(ux, 0.0);
/// assert_eq!(uy, 0.0);
/// ```
#[derive(Theory)]
#[theory(
    description = "The Lattice Boltzmann Method (LBM) is a discrete computational fluid dynamics approach that simulates Newtonian fluid flows by modeling the microscopic behavior of fictitious fluid particles on a regular grid. It approximates the Navier-Stokes equations and the material derivative D/Dt.",
    phonetic = "The Lattice Boltzmann Method is a discrete computational fluid dynamics approach that simulates Newtonian fluid flows by modeling the microscopic behavior of fictitious fluid particles on a regular grid. It approximates the Navier-Stokes equations and the Material derivative with respect to time.",
    citation = "Lattice Boltzmann equation for fluid dynamics and beyond (Succi, 2001)"
)]
pub struct LatticeBoltzmann<const Q: usize, L: Lattice2D<Q>, C: CollisionModel<Q, L>> {
    /// The simulation state (grids, obstacles).
    pub state: LatticeState<Q>,
    /// Collision Strategy
    pub collision_model: C,
    #[allow(missing_docs)]
    pub _marker: PhantomData<L>,
}

/// Type Alias for Backward Compatibility.
pub type LatticeBoltzmannD2Q9<C> = LatticeBoltzmann<9, D2Q9, C>;
