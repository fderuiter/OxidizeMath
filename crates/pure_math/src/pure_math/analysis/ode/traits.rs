use std::ops::{Add, AddAssign, Mul, MulAssign};

/// A trait defining the vector operations required by numerical integrators.
///
/// This trait allows the solver to be agnostic of the underlying storage (Heap vs Stack).
/// It requires the type to support addition and scalar multiplication.
pub trait VectorOperations:
    Sized + Clone + Add<Output = Self> + Mul<f64, Output = Self> + AddAssign + MulAssign<f64>
{
    /// Adds `other` scaled by `scale` to `self`.
    /// `self += other * scale`
    #[verified_engine::verified]
    fn scale_add(&mut self, other: &Self, scale: f64);

    /// Copies the content of `other` into `self`.
    /// This allows reusing allocated buffers.
    #[verified_engine::verified]
    fn copy_from(&mut self, other: &Self);

    /// Fused operation: `self = source + other * scale`.
    ///
    /// Copies `source` into `self` while adding `other` scaled by `scale`.
    /// This avoids intermediate writes and reads, reducing memory traffic.
    ///
    /// # Default Implementation
    /// Calls `copy_from` then `scale_add`. Override for better performance.
    #[verified_engine::verified]
    fn copy_from_scaled(&mut self, source: &Self, other: &Self, scale: f64) {
        self.copy_from(source);
        self.scale_add(other, scale);
    }

    /// Extracts `self` for an in-place solver step while leaving a valid placeholder behind.
    ///
    /// Heap-backed state vectors override this with a zero-capacity replacement so
    /// [`super::stepper::TimeStepper::step_with`] can split borrows without heap allocations.
    #[verified_engine::verified]
    fn take_for_step(&mut self) -> Self {
        self.clone()
    }
}

/// A trait defining a system of Ordinary Differential Equations.
///
/// $$ \frac{d\vec{y}}{dt} = f(t, \vec{y}) $$
///
/// The state type `State` must implement `VectorOperations`.
pub trait OdeSystem<State: VectorOperations> {
    /// Computes the time derivative of the system state.
    ///
    /// # Arguments
    /// * `t` - The current time.
    /// * `state` - The current state vector $\vec{y}$.
    ///
    /// # Returns
    /// The derivative vector $\frac{d\vec{y}}{dt}$.
    ///
    /// # Performance
    /// Default implementation allocates a new State. Implement `derivative_in_place` for better performance.
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &State) -> State;

    /// Computes the time derivative of the system state in-place.
    ///
    /// # Arguments
    /// * `t` - The current time.
    /// * `state` - The current state vector $\vec{y}$.
    /// * `out` - The output vector to store $\frac{d\vec{y}}{dt}$.
    #[verified_engine::verified]
    fn derivative_in_place(&self, t: f64, state: &State, out: &mut State) {
        *out = self.derivative(t, state);
    }
}

/// A trait defining a numerical ODE solver strategy.
pub trait Solver<State: VectorOperations> {
    /// Advances the system state by one time step `dt` in-place.
    #[verified_engine::verified]
    fn step<S>(&mut self, system: &S, t: f64, state: &mut State, dt: f64)
    where
        S: OdeSystem<State> + ?Sized;
}

/// Extension trait for `Solver` providing derived computational methods.
pub trait SolverExt<State: VectorOperations>: Solver<State> {
    /// Advances the system state by one time step `dt`.
    #[verified_engine::verified]
    fn solve<S>(&mut self, system: &S, t: f64, state: &State, dt: f64) -> State
    where
        S: OdeSystem<State> + ?Sized;
}

impl<State: VectorOperations, T: Solver<State> + ?Sized> SolverExt<State> for T {
    #[verified_engine::verified]
    fn solve<S>(&mut self, system: &S, t: f64, state: &State, dt: f64) -> State
    where
        S: OdeSystem<State> + ?Sized,
    {
        let mut new_state = state.clone();
        self.step(system, t, &mut new_state, dt);
        new_state
    }
}

/// A canonical Hamiltonian dynamical system with conserved scalar energy $H(q, p)$.
///
/// Implementing systems evolve according to Hamilton's canonical equations
/// $\dot{q} = \nabla_p H(q, p)$ and $\dot{p} = -\nabla_q H(q, p)$ via [`OdeSystem`].
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::ode::{
///     ArrayState, GaussLegendre4, HamiltonianSystem, OdeSystem, Solver,
/// };
///
/// struct HarmonicOscillator;
///
/// impl OdeSystem<ArrayState<2>> for HarmonicOscillator {
///     fn derivative(&self, _t: f64, s: &ArrayState<2>) -> ArrayState<2> {
///         ArrayState([s.0[1], -s.0[0]])
///     }
/// }
///
/// impl HamiltonianSystem<ArrayState<2>> for HarmonicOscillator {
///     fn hamiltonian(&self, s: &ArrayState<2>) -> f64 {
///         0.5 * (s.0[0] * s.0[0] + s.0[1] * s.0[1])
///     }
/// }
///
/// let sys = HarmonicOscillator;
/// let mut state = ArrayState([1.0, 0.0]);
/// let e0 = sys.hamiltonian(&state);
/// let mut solver = GaussLegendre4::new(&state);
/// solver.step(&sys, 0.0, &mut state, 0.05);
/// assert!(sys.relative_energy_error(&state, e0) < 1e-12);
/// ```
pub trait HamiltonianSystem<State: VectorOperations>: OdeSystem<State> {
    /// Evaluates the scalar Hamiltonian (total energy) $H(q, p)$ at `state`.
    #[verified_engine::verified]
    fn hamiltonian(&self, state: &State) -> f64;

    /// Computes relative energy drift $|H(y) - H_0| / |H_0|$ (falling back to $|H(y) - H_0|$ near zero).
    #[verified_engine::verified]
    fn relative_energy_error(&self, state: &State, initial_energy: f64) -> f64 {
        let current = self.hamiltonian(state);
        let denom = initial_energy.abs();
        if denom > 1e-15 {
            (current - initial_energy).abs() / denom
        } else {
            (current - initial_energy).abs()
        }
    }
}

/// A separable Hamiltonian system $H(q, p) = T(p) + V(q)$ supporting explicit symplectic splitting.
///
/// Partitions the full phase-space `State` into generalized coordinates `q` and conjugate
/// momenta `p` of type [`Self::Coord`], enabling zero-allocation Störmer-Verlet and Yoshida
/// composition integrators without implicit root-finding.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::ode::{
///     ArrayState, HamiltonianSystem, OdeSystem, SeparableHamiltonian, Yoshida4,
/// };
///
/// struct SeparableOscillator;
///
/// impl OdeSystem<ArrayState<2>> for SeparableOscillator {
///     fn derivative(&self, _t: f64, s: &ArrayState<2>) -> ArrayState<2> {
///         ArrayState([s.0[1], -s.0[0]])
///     }
/// }
///
/// impl HamiltonianSystem<ArrayState<2>> for SeparableOscillator {
///     fn hamiltonian(&self, s: &ArrayState<2>) -> f64 {
///         0.5 * (s.0[0] * s.0[0] + s.0[1] * s.0[1])
///     }
/// }
///
/// impl SeparableHamiltonian<ArrayState<2>> for SeparableOscillator {
///     type Coord = ArrayState<1>;
///     fn split_canonical(&self, s: &ArrayState<2>, q: &mut Self::Coord, p: &mut Self::Coord) {
///         q.0[0] = s.0[0];
///         p.0[0] = s.0[1];
///     }
///     fn join_canonical(&self, q: &Self::Coord, p: &Self::Coord, s: &mut ArrayState<2>) {
///         s.0[0] = q.0[0];
///         s.0[1] = p.0[0];
///     }
///     fn kinetic_energy(&self, p: &Self::Coord) -> f64 { 0.5 * p.0[0] * p.0[0] }
///     fn potential_energy(&self, q: &Self::Coord) -> f64 { 0.5 * q.0[0] * q.0[0] }
///     fn grad_kinetic_in_place(&self, p: &Self::Coord, dq: &mut Self::Coord) { dq.0[0] = p.0[0]; }
///     fn neg_grad_potential_in_place(&self, q: &Self::Coord, dp: &mut Self::Coord) { dp.0[0] = -q.0[0]; }
/// }
///
/// let sys = SeparableOscillator;
/// let mut state = ArrayState([1.0, 0.0]);
/// let mut solver = Yoshida4::new(&ArrayState([0.0]));
/// solver.step_separable(&sys, &mut state, 0.05);
/// assert!((sys.hamiltonian(&state) - 0.5).abs() < 1e-8);
/// ```
pub trait SeparableHamiltonian<State: VectorOperations>: HamiltonianSystem<State> {
    /// Coordinate/momentum half-dimension vector type (e.g. `ArrayState<2>` for 2-DOF `ArrayState<4>`).
    type Coord: VectorOperations;

    /// Extracts canonical coordinates `q` and conjugate momenta `p` from `state` in-place.
    #[verified_engine::verified]
    fn split_canonical(&self, state: &State, q: &mut Self::Coord, p: &mut Self::Coord);

    /// Writes canonical coordinates `q` and conjugate momenta `p` back into `state` in-place.
    #[verified_engine::verified]
    fn join_canonical(&self, q: &Self::Coord, p: &Self::Coord, state: &mut State);

    /// Evaluates the kinetic energy $T(p)$.
    #[verified_engine::verified]
    fn kinetic_energy(&self, p: &Self::Coord) -> f64;

    /// Evaluates the potential energy $V(q)$.
    #[verified_engine::verified]
    fn potential_energy(&self, q: &Self::Coord) -> f64;

    /// Computes canonical velocity $\dot{q} = \nabla_p T(p)$ in-place.
    #[verified_engine::verified]
    fn grad_kinetic_in_place(&self, p: &Self::Coord, dq_dt: &mut Self::Coord);

    /// Computes canonical force $\dot{p} = -\nabla_q V(q)$ in-place.
    #[verified_engine::verified]
    fn neg_grad_potential_in_place(&self, q: &Self::Coord, dp_dt: &mut Self::Coord);
}
