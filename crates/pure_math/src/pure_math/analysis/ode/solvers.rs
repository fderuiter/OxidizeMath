use super::traits::{OdeSystem, Solver, VectorOperations};

/// Shared kernel for Runge-Kutta 4th Order logic.
///
/// This function encapsulates the core RK4 arithmetic to ensure consistency
/// between the static (allocating) method and the struct-based (buffered) method.
///
/// # Arguments
/// * `system`: The ODE system.
/// * `t`: Current time.
/// * `dt`: Time step.
/// * `y_n`: The state at the beginning of the step ($y_n$).
/// * `y_acc`: The accumulator for the new state ($y_{n+1}$).
/// * `k`: Scratch buffer for derivatives.
/// * `tmp`: Scratch buffer for intermediate states.
#[verified_engine::verified]
fn rk4_kernel<State, S>(
    system: &S,
    t: f64,
    dt: f64,
    y_n: &State,
    y_acc: &mut State,
    k: &mut State,
    tmp: &mut State,
) where
    State: VectorOperations,
    S: OdeSystem<State> + ?Sized,
{
    // k1 = f(t, y_n)
    system.derivative_in_place(t, y_n, k);
    // y_acc += k1 * dt/6
    y_acc.scale_add(k, dt / 6.0);

    // k2 = f(t + dt/2, y_n + k1 * dt/2)
    // tmp = y_n + k1 * dt/2
    tmp.copy_from_scaled(y_n, k, dt / 2.0);
    system.derivative_in_place(t + dt / 2.0, tmp, k);
    // y_acc += k2 * dt/3
    y_acc.scale_add(k, dt / 3.0);

    // k3 = f(t + dt/2, y_n + k2 * dt/2)
    // tmp = y_n + k2 * dt/2
    tmp.copy_from_scaled(y_n, k, dt / 2.0);
    system.derivative_in_place(t + dt / 2.0, tmp, k);
    // y_acc += k3 * dt/3
    y_acc.scale_add(k, dt / 3.0);

    // k4 = f(t + dt, y_n + k3 * dt)
    // tmp = y_n + k3 * dt
    tmp.copy_from_scaled(y_n, k, dt);
    system.derivative_in_place(t + dt, tmp, k);
    // y_acc += k4 * dt/6
    y_acc.scale_add(k, dt / 6.0);
}

/// Euler's Method Solver.
///
/// A simple first-order numerical integrator.
/// Maintains an internal buffer to avoid allocations during steps.
///
/// # Parse, Don't Validate
/// This solver requires an example state at construction to pre-allocate buffers,
/// ensuring that invalid (uninitialized) states are unrepresentable during simulation.
#[derive(Debug, Clone)]
pub struct Euler<State> {
    buffer: State,
}

impl<State: Clone> Euler<State> {
    /// Creates a new Euler solver.
    ///
    /// # Arguments
    /// * `example_state` - A reference to a state vector used to determine buffer size/type.
    #[verified_engine::verified]
    pub fn new(example_state: &State) -> Self {
        Self {
            buffer: example_state.clone(),
        }
    }
}

impl<State: VectorOperations> Solver<State> for Euler<State> {
    #[verified_engine::verified]
    fn step<S>(&mut self, system: &S, t: f64, state: &mut State, dt: f64)
    where
        S: OdeSystem<State> + ?Sized,
    {
        // derivative = f(t, y)
        system.derivative_in_place(t, state, &mut self.buffer);

        // y += derivative * dt
        state.scale_add(&self.buffer, dt);
    }
}

/// Runge-Kutta 4th Order Solver.
///
/// A classic fixed-step integrator for ODEs.
/// Maintains internal buffers to avoid allocations.
///
/// # Parse, Don't Validate
/// This solver requires an example state at construction to pre-allocate buffers,
/// ensuring that invalid (uninitialized) states are unrepresentable during simulation.
#[derive(Debug, Clone)]
pub struct RungeKutta4<State> {
    k: State,
    tmp: State,
    initial_state: State,
}

impl<State: Clone> RungeKutta4<State> {
    /// Creates a new Runge-Kutta 4 solver.
    ///
    /// # Arguments
    /// * `example_state` - A reference to a state vector used to determine buffer size/type.
    #[verified_engine::verified]
    pub fn new(example_state: &State) -> Self {
        Self {
            k: example_state.clone(),
            tmp: example_state.clone(),
            initial_state: example_state.clone(),
        }
    }
}

impl<State: VectorOperations> Solver<State> for RungeKutta4<State> {
    #[verified_engine::verified]
    fn step<S>(&mut self, system: &S, t: f64, state: &mut State, dt: f64)
    where
        S: OdeSystem<State> + ?Sized,
    {
        // Copy current state to initial_state buffer to preserve y_n
        self.initial_state.copy_from(state);

        // state acts as y_acc (accumulator)
        rk4_kernel(
            system,
            t,
            dt,
            &self.initial_state,
            state,
            &mut self.k,
            &mut self.tmp,
        );
    }
}
