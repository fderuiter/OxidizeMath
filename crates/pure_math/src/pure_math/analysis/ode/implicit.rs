//! Stiff A-stable / L-damped implicit Runge-Kutta ODE solvers (`BackwardEuler`, `ImplicitTrapezoidal`).
//!
//! Solves nonlinear stage equations via finite-difference Newton iterations with in-place
//! Gaussian elimination over pre-allocated buffers for zero post-init heap allocations.

use super::traits::{IndexedState, OdeSystem, Solver};
use crate::pure_math::analysis::dense_linear::solve_linear_in_place;

const FD_REL_STEP: f64 = 1.490_116_119_384_765_6e-8;
const PIVOT_TOL: f64 = 1e-14;
const TRAP_STIFF_CUTOFF: f64 = 1.8;

#[derive(Debug, Clone)]
struct ImplicitWorkspace<State: IndexedState> {
    y_iter: State,
    y_pert: State,
    f_base: State,
    f_iter: State,
    f_pert: State,
    jac_mat: Vec<f64>,
    rhs: Vec<f64>,
    scratch: Vec<f64>,
    pivots: Vec<usize>,
    newton_tol: f64,
    max_newton_iters: usize,
}

impl<State: IndexedState> ImplicitWorkspace<State> {
    #[verified_engine::verified]
    fn new(template: &State) -> Self {
        let n = template.len();
        Self {
            y_iter: template.clone(),
            y_pert: template.clone(),
            f_base: template.clone(),
            f_iter: template.clone(),
            f_pert: template.clone(),
            jac_mat: vec![0.0; n * n],
            rhs: vec![0.0; n],
            scratch: vec![0.0; n],
            pivots: vec![0; n],
            newton_tol: 1e-11,
            max_newton_iters: 12,
        }
    }

    #[verified_engine::verified]
    fn ensure_capacity(&mut self, n: usize) {
        if self.jac_mat.len() < n * n {
            self.jac_mat.resize(n * n, 0.0);
        }
        if self.rhs.len() < n {
            self.rhs.resize(n, 0.0);
        }
        if self.scratch.len() < n {
            self.scratch.resize(n, 0.0);
        }
        if self.pivots.len() < n {
            self.pivots.resize(n, 0);
        }
    }

    /// Evaluates the Jacobian infinity norm `||J(t, y)||_inf` using forward finite differences.
    #[verified_engine::verified]
    fn jacobian_inf_norm<S>(&mut self, system: &S, t: f64, state: &State) -> f64
    where
        S: OdeSystem<State> + ?Sized,
    {
        let n = state.len();
        self.ensure_capacity(n);
        for val in self.rhs.iter_mut().take(n) {
            *val = 0.0;
        }
        self.y_pert.copy_from(state);
        for j in 0..n {
            let yj = state.get(j);
            let h_fd = FD_REL_STEP * yj.abs().max(1.0);
            self.y_pert.set(j, yj + h_fd);
            system.derivative_in_place(t, &self.y_pert, &mut self.f_pert);
            self.y_pert.set(j, yj);
            let inv_h = 1.0 / h_fd;
            for i in 0..n {
                let df = ((self.f_pert.get(i) - self.f_base.get(i)) * inv_h).abs();
                self.rhs[i] += df;
            }
        }
        let mut max_row = 0.0;
        for &row_sum in self.rhs.iter().take(n) {
            if row_sum > max_row {
                max_row = row_sum;
            }
        }
        max_row
    }

    /// Solves `y - state - dt * ((1 - theta) * f_base + theta * f(t_next, y)) = 0` in-place.
    #[verified_engine::verified]
    fn solve_theta_step<S>(
        &mut self,
        system: &S,
        t_next: f64,
        state: &mut State,
        dt: f64,
        theta: f64,
    ) where
        S: OdeSystem<State> + ?Sized,
    {
        let n = state.len();
        self.ensure_capacity(n);
        self.y_iter.copy_from(state);
        let beta = theta * dt;
        let alpha = (1.0 - theta) * dt;

        for _ in 0..self.max_newton_iters {
            system.derivative_in_place(t_next, &self.y_iter, &mut self.f_iter);
            let mut max_res = 0.0;
            for i in 0..n {
                let res = self.y_iter.get(i)
                    - state.get(i)
                    - alpha * self.f_base.get(i)
                    - beta * self.f_iter.get(i);
                if !res.is_finite() {
                    return;
                }
                self.rhs[i] = -res;
                let a = res.abs();
                if a > max_res {
                    max_res = a;
                }
            }
            if max_res <= self.newton_tol {
                state.copy_from(&self.y_iter);
                return;
            }
            self.assemble_system_matrix(system, t_next, beta, n);
            if !solve_linear_in_place(
                &mut self.jac_mat,
                &mut self.rhs,
                &mut self.pivots,
                &mut self.scratch,
                n,
                PIVOT_TOL,
            ) {
                return;
            }
            let mut max_delta = 0.0;
            for i in 0..n {
                let d = self.rhs[i];
                self.y_iter.set(i, self.y_iter.get(i) + d);
                if d.abs() > max_delta {
                    max_delta = d.abs();
                }
            }
            if max_delta <= self.newton_tol {
                state.copy_from(&self.y_iter);
                return;
            }
        }
        state.copy_from(&self.y_iter);
    }

    #[verified_engine::verified]
    fn assemble_system_matrix<S>(&mut self, system: &S, t_next: f64, beta: f64, n: usize)
    where
        S: OdeSystem<State> + ?Sized,
    {
        self.y_pert.copy_from(&self.y_iter);
        for j in 0..n {
            let yj = self.y_iter.get(j);
            let h_fd = FD_REL_STEP * yj.abs().max(1.0);
            self.y_pert.set(j, yj + h_fd);
            system.derivative_in_place(t_next, &self.y_pert, &mut self.f_pert);
            self.y_pert.set(j, yj);
            let inv_h = 1.0 / h_fd;
            for i in 0..n {
                let j_ij = (self.f_pert.get(i) - self.f_iter.get(i)) * inv_h;
                let diag = if i == j { 1.0 } else { 0.0 };
                let idx = math_commons::math_kernel::types::flatten_2d_index(j, i, n);
                self.jac_mat[idx] = diag - beta * j_ij;
            }
        }
    }
}

/// 1st-order L-stable implicit Backward Euler ODE solver (`y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})`).
///
/// Unconditionally A-stable, L-stable, and positivity-preserving for arbitrarily stiff linear decay.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::ode::{
///     ArrayState, BackwardEuler, OdeModel, OdeSystem, TimeStepper,
/// };
///
/// struct StiffDecay;
/// impl OdeSystem<ArrayState<1>> for StiffDecay {
///     fn derivative(&self, _t: f64, s: &ArrayState<1>) -> ArrayState<1> {
///         ArrayState([-1000.0 * s.0[0]])
///     }
/// }
///
/// let s0 = ArrayState([1.0]);
/// let mut model = OdeModel::new(s0, StiffDecay, BackwardEuler::new(&s0));
/// model.step(0.05);
/// assert!(model.get_state().0[0] > 0.0 && model.get_state().0[0] < 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct BackwardEuler<State: IndexedState> {
    ws: ImplicitWorkspace<State>,
}

impl<State: IndexedState> BackwardEuler<State> {
    /// Creates a `BackwardEuler` solver with pre-allocated Newton buffers matching `template`.
    #[verified_engine::verified]
    pub fn new(template: &State) -> Self {
        Self {
            ws: ImplicitWorkspace::new(template),
        }
    }

    /// Configures the nonlinear Newton residual tolerance and iteration cap.
    #[verified_engine::verified]
    pub fn with_newton_params(mut self, newton_tol: f64, max_newton_iters: usize) -> Self {
        self.ws.newton_tol = newton_tol.max(1e-15);
        self.ws.max_newton_iters = max_newton_iters.max(1);
        self
    }
}

impl<State: IndexedState> Solver<State> for BackwardEuler<State> {
    #[verified_engine::verified]
    fn step<S>(&mut self, system: &S, t: f64, state: &mut State, dt: f64)
    where
        S: OdeSystem<State> + ?Sized,
    {
        if !dt.is_finite() || dt == 0.0 || state.is_empty() {
            return;
        }
        self.ws.solve_theta_step(system, t + dt, state, dt, 1.0);
    }
}

/// 2nd-order A-stable Implicit Trapezoidal (Crank-Nicolson) ODE solver with exponential-fitting
/// stiff positivity stabilization.
///
/// Executes pure 2nd-order Crank-Nicolson (`theta = 0.5`) whenever `|dt| * ||J||_inf <= 1.8`, and
/// smoothly shifts `theta = 1 - 0.9 / rho` in ultra-stiff regimes (`rho > 1.8`) to eliminate
/// spurious Crank-Nicolson sign oscillations while keeping solutions bounded and positive.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::ode::{
///     ArrayState, ImplicitTrapezoidal, OdeModel, OdeSystem, TimeStepper,
/// };
///
/// struct StiffDecay;
/// impl OdeSystem<ArrayState<1>> for StiffDecay {
///     fn derivative(&self, _t: f64, s: &ArrayState<1>) -> ArrayState<1> {
///         ArrayState([-1000.0 * s.0[0]])
///     }
/// }
///
/// let s0 = ArrayState([1.0]);
/// let mut model = OdeModel::new(s0, StiffDecay, ImplicitTrapezoidal::new(&s0));
/// model.step(0.05);
/// assert!(model.get_state().0[0] > 0.0 && model.get_state().0[0] < 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct ImplicitTrapezoidal<State: IndexedState> {
    ws: ImplicitWorkspace<State>,
}

impl<State: IndexedState> ImplicitTrapezoidal<State> {
    /// Creates an `ImplicitTrapezoidal` solver with pre-allocated Newton buffers matching `template`.
    #[verified_engine::verified]
    pub fn new(template: &State) -> Self {
        Self {
            ws: ImplicitWorkspace::new(template),
        }
    }

    /// Configures the nonlinear Newton residual tolerance and iteration cap.
    #[verified_engine::verified]
    pub fn with_newton_params(mut self, newton_tol: f64, max_newton_iters: usize) -> Self {
        self.ws.newton_tol = newton_tol.max(1e-15);
        self.ws.max_newton_iters = max_newton_iters.max(1);
        self
    }
}

impl<State: IndexedState> Solver<State> for ImplicitTrapezoidal<State> {
    #[verified_engine::verified]
    fn step<S>(&mut self, system: &S, t: f64, state: &mut State, dt: f64)
    where
        S: OdeSystem<State> + ?Sized,
    {
        if !dt.is_finite() || dt == 0.0 || state.is_empty() {
            return;
        }
        system.derivative_in_place(t, state, &mut self.ws.f_base);
        let j_norm = self.ws.jacobian_inf_norm(system, t, state);
        let rho = dt.abs() * j_norm;
        let theta = if rho.is_finite() && rho > TRAP_STIFF_CUTOFF {
            (1.0 - 0.9 / rho).clamp(0.5, 1.0)
        } else {
            0.5
        };
        self.ws.solve_theta_step(system, t + dt, state, dt, theta);
    }
}
