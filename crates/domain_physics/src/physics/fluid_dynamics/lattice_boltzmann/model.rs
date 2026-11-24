use super::state::LatticeState;
use std::marker::PhantomData;
use verified_engine::Theory;

/// Strategy trait defining a $Q$-velocity 2D quadrature lattice $\{(\mathbf{c}_k, w_k)\}_{k=0}^{Q-1}$ with $\sum_k w_k = 1$ (`@qian1992lattice`, `spec:fluid_dynamics_v1`).
/// Numerical stability: low-Mach polynomial equilibrium requires $|\mathbf{u}| < c_s = 1/\sqrt{3}$ for non-negative populations. Domain of validity: isothermal weakly compressible 2D flows ($Q \ge 5$).
pub trait Lattice2D<const Q: usize>: Copy + Clone + Send + Sync + 'static {
    /// Returns quadrature weights $w_k > 0$ satisfying $\sum_{k=0}^{Q-1} w_k = 1$ (`@qian1992lattice`, `spec:fluid_dynamics_v1`). Numerical stability: compile-time rational constants rounded to IEEE-754 `f64`. Domain of validity: $0 \le k < Q$.
    #[verified_engine::verified]
    fn weights() -> [f64; Q];
    /// Returns discrete horizontal lattice velocities $c_{k,x} \in \{-1, 0, 1\}$ (`@qian1992lattice`, `spec:fluid_dynamics_v1`). Numerical stability: exact `i32` integer shifts. Domain of validity: $0 \le k < Q$.
    #[verified_engine::verified]
    fn directions_x() -> [i32; Q];
    /// Returns discrete vertical lattice velocities $c_{k,y} \in \{-1, 0, 1\}$ (`@qian1992lattice`, `spec:fluid_dynamics_v1`). Numerical stability: exact `i32` integer shifts. Domain of validity: $0 \le k < Q$.
    #[verified_engine::verified]
    fn directions_y() -> [i32; Q];
    /// Returns bounce-back involutive index map $\bar{k}$ such that $\mathbf{c}_{\bar{k}} = -\mathbf{c}_k$ (`@qian1992lattice`, `spec:fluid_dynamics_v1`). Numerical stability: exact permutation in `0..Q`. Domain of validity: $0 \le k < Q$.
    #[verified_engine::verified]
    fn opposite_indices() -> [usize; Q];
    /// Evaluates second-order Maxwell-Boltzmann equilibrium $f_k^{\text{eq}} = w_k \rho (1 + 3 \mathbf{c}_k \cdot \mathbf{u} + \tfrac{9}{2}(\mathbf{c}_k \cdot \mathbf{u})^2 - \tfrac{3}{2}|\mathbf{u}|^2)$ (`@qian1992lattice`, `spec:fluid_dynamics_v1`).
    /// Numerical stability: positivity of $f_k^{\text{eq}}$ holds when $|\mathbf{u}| \ll c_s = 1/\sqrt{3}$. Domain of validity: finite density $\rho > 0$ and Mach number $\text{Ma} = |\mathbf{u}|/c_s < 0.3$.
    #[verified_engine::verified]
    fn equilibrium(rho: f64, ux: f64, uy: f64) -> [f64; Q];
}

/// Standard 9-velocity 2D lattice (`D2Q9`) with speed of sound $c_s = 1/\sqrt{3}$ and weights $(4/9, 1/9, 1/36)$ (`@qian1992lattice`, `spec:fluid_dynamics_v1`).
/// Numerical stability: isotropic up to fourth-order velocity moments for $|\mathbf{u}| < 0.2$. Domain of validity: 2D weakly compressible Navier-Stokes regimes ($\text{Ma} < 0.3$).
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

/// Collision operator relaxing populations $f_k \mapsto f_k + \Omega_k(f)$ while conserving $\rho = \sum_k f_k$ and $\rho \mathbf{u} = \sum_k \mathbf{c}_k f_k$ (`@bhatnagar1954model`, `spec:fluid_dynamics_v1`).
/// Numerical stability: relaxation spectrum must lie inside $(-2, 0)$ to avoid post-collision over-relaxation blow-up. Domain of validity: finite $\rho > 0$ and $|\mathbf{u}| < c_s$.
pub trait CollisionModel<const Q: usize, L: Lattice2D<Q>> {
    /// Applies in-place kinetic relaxation $f_k^\star = f_k + \Omega_k(f; \rho, u_x, u_y)$ (`@bhatnagar1954model`, `spec:fluid_dynamics_v1`).
    /// Numerical stability: conserves mass and momentum up to $O(Q \varepsilon_{\text{mach}})$. Domain of validity: finite `f`, $\rho > 0$, and subsonic $(u_x, u_y)$.
    #[verified_engine::verified]
    fn apply(&self, f: &mut [f64; Q], rho: f64, ux: f64, uy: f64);
}

/// Single-relaxation-time Bhatnagar-Gross-Krook operator $f_k^\star = (1 - 1/\tau) f_k + (1/\tau) f_k^{\text{eq}}$ with $\nu = c_s^2(\tau - 1/2)$ (`@bhatnagar1954model`, `spec:fluid_dynamics_v1`).
/// Numerical stability: strictly requires $\tau > 0.5$ ($\nu > 0$); $\tau \ge 0.515$ avoids under-resolved grid oscillations. Domain of validity: $\tau \in [0.515, 1.1]$ and $|\mathbf{u}| < 0.2$.
#[derive(Debug, Clone, Copy, Theory)]
#[theory(
    description = "Bhatnagar-Gross-Krook (BGK) collision operator for single relaxation time.",
    citation = "bhatnagar1954model"
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
///
/// # Governing Equation & Verification
///
/// Evolves discrete populations via $f_k(\mathbf{x} + \mathbf{c}_k \Delta t, t + \Delta t) - f_k(\mathbf{x}, t) = -\frac{1}{\tau}(f_k - f_k^{\text{eq}})$.
/// - **Citation**: `@qian1992lattice` (`spec:fluid_dynamics_v1`).
/// - **Numerical stability**: Courant-locked streaming ($\Delta x = \Delta t = 1$) is stable when $\tau > 0.5$ and lattice Mach number $\text{Ma} = |\mathbf{u}| / c_s < 0.3$.
/// - **Domain of validity**: Non-empty 2D grids (`width >= 1`, `height >= 1`), positive relaxation $\tau > 0.5$, and subsonic boundary velocities.
#[derive(Theory)]
#[theory(
    description = "The Lattice Boltzmann Method (LBM) is a discrete computational fluid dynamics approach that simulates Newtonian fluid flows by modeling the microscopic behavior of fictitious fluid particles on a regular grid. It approximates the Navier-Stokes equations and the material derivative D/Dt.",
    phonetic = "The Lattice Boltzmann Method is a discrete computational fluid dynamics approach that simulates Newtonian fluid flows by modeling the microscopic behavior of fictitious fluid particles on a regular grid. It approximates the Navier-Stokes equations and the Material derivative with respect to time.",
    citation = "qian1992lattice"
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
