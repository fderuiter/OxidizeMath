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
