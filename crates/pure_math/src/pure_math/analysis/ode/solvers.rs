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

/// Explicit first-order forward Euler integrator $\mathbf{y}_{n+1} = \mathbf{y}_n + \Delta t\,\mathbf{f}(t_n, \mathbf{y}_n)$.
///
/// Reference: Hairer, Norsett & Wanner (1993), DOI: 10.1007/978-3-540-78862-1; [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Conditionally A-stable inside the unit disk $|1 + \lambda \Delta t| \le 1$; unconditionally unstable for purely imaginary eigenvalues $\lambda = i\omega \neq 0$.
///
/// # Domain of Validity
/// Valid for non-stiff dissipative systems with small time steps $\Delta t < 2 / |\lambda|_{\max}$.
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

/// Explicit second-order Heun predictor-corrector (RK2) solver $\mathbf{y}_{n+1} = \mathbf{y}_n + \frac{\Delta t}{2}\bigl(\mathbf{k}_1 + \mathbf{f}(t_n + \Delta t, \mathbf{y}_n + \Delta t\,\mathbf{k}_1)\bigr)$.
///
/// Reference: Hairer, Norsett & Wanner (1993), DOI: 10.1007/978-3-540-78862-1; [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Strong-stability-preserving (SSP) 2-stage Runge-Kutta method with stability polynomial $R(z) = 1 + z + z^2/2$ ($|R(z)| \le 1$).
///
/// # Domain of Validity
/// Valid for non-stiff ODEs with $C^2$ solutions and step sizes within $|1 + z + z^2/2| \le 1$.
#[derive(Debug, Clone)]
pub struct Heun<State> {
    k: State,
    tmp: State,
    initial_state: State,
}

impl<State: Clone> Heun<State> {
    /// Creates a new Heun solver.
    ///
    /// # Arguments
    /// * `example_state` - A reference to a state vector used to size scratch buffers.
    #[verified_engine::verified]
    pub fn new(example_state: &State) -> Self {
        Self {
            k: example_state.clone(),
            tmp: example_state.clone(),
            initial_state: example_state.clone(),
        }
    }
}

impl<State: VectorOperations> Solver<State> for Heun<State> {
    #[verified_engine::verified]
    fn step<S>(&mut self, system: &S, t: f64, state: &mut State, dt: f64)
    where
        S: OdeSystem<State> + ?Sized,
    {
        // Copy current state to initial_state buffer to retain y_n
        self.initial_state.copy_from(state);

        // k1 = f(t, y_n)
        system.derivative_in_place(t, &self.initial_state, &mut self.k);

        // Predictor step: tmp = y_n + dt * k1
        self.tmp.copy_from_scaled(&self.initial_state, &self.k, dt);

        // Accumulate first stage: state = y_n + 0.5 * dt * k1
        state.scale_add(&self.k, dt * 0.5);

        // Corrector step: k2 = f(t + dt, tmp)
        system.derivative_in_place(t + dt, &self.tmp, &mut self.k);

        // Accumulate second stage: state += 0.5 * dt * k2
        state.scale_add(&self.k, dt * 0.5);
    }
}

/// Classical explicit 4-stage 4th-order Runge-Kutta integrator $\mathbf{y}_{n+1} = \mathbf{y}_n + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)$.
///
/// Reference: Hairer, Norsett & Wanner (1993), DOI: 10.1007/978-3-540-78862-1; [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Global truncation error $O(\Delta t^4)$ with stability polynomial $R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24$ covering imaginary-axis modes up to $|\omega \Delta t| \le 2\sqrt{2}$.
///
/// # Domain of Validity
/// Valid for non-stiff $C^4$ dynamical systems where $|\lambda \Delta t| \lesssim 2.78$.
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
