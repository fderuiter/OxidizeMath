//! Adaptive L-stable modified Rosenbrock triple of order (2)3 (`Rosenbrock23` / `ode23s`).
//!
//! Solves linearly implicit Rosenbrock stages using one finite-difference Jacobian and one
//! in-place `PA = LU` factorization per substep with zero post-initialization heap allocations.

use super::traits::{IndexedState, OdeSystem, Solver};
use crate::pure_math::analysis::dense_linear::{
    lu_factor_in_place, lu_solve_in_place, solve_linear_in_place,
};

const FD_REL_STEP: f64 = 1.490_116_119_384_765_6e-8;
const PIVOT_TOL: f64 = 1e-14;
const ROS_D: f64 = 0.292_893_218_813_452_5; // 1 / (2 + sqrt(2))
const ROS_E32: f64 = 7.414_213_562_373_095; // 6 + sqrt(2)
const POSITIVITY_RHO_LIMIT: f64 = 2.2;

/// Adaptive L-stable linearly-implicit Rosenbrock 2(3) ODE solver (`ode23s`).
///
/// Particularly suited to stiff and non-autonomous ODEs where explicit Runge-Kutta methods
/// suffer severe Courant-like time-step restrictions.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::ode::{
///     ArrayState, OdeModel, OdeSystem, Rosenbrock23, TimeStepper,
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
/// let mut model = OdeModel::new(s0, StiffDecay, Rosenbrock23::new(&s0));
/// model.step(0.05);
/// assert!(model.get_state().0[0] > 0.0 && model.get_state().0[0] < 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct Rosenbrock23<State: IndexedState> {
    f0: State,
    f1: State,
    f2: State,
    f_pert: State,
    f_t_minus: State,
    y_stage: State,
    y_next: State,
    jac: Vec<f64>,
    lu: Vec<f64>,
    pivots: Vec<usize>,
    dfdt: Vec<f64>,
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    scratch: Vec<f64>,
    rtol: f64,
    atol: f64,
    max_substeps: usize,
    last_error: f64,
    last_substeps: usize,
}

impl<State: IndexedState> Rosenbrock23<State> {
    /// Creates a `Rosenbrock23` solver with pre-allocated stage and LU buffers matching `template`.
    #[verified_engine::verified]
    pub fn new(template: &State) -> Self {
        let n = template.len();
        Self {
            f0: template.clone(),
            f1: template.clone(),
            f2: template.clone(),
            f_pert: template.clone(),
            f_t_minus: template.clone(),
            y_stage: template.clone(),
            y_next: template.clone(),
            jac: vec![0.0; n * n],
            lu: vec![0.0; n * n],
            pivots: vec![0; n],
            dfdt: vec![0.0; n],
            k1: vec![0.0; n],
            k2: vec![0.0; n],
            k3: vec![0.0; n],
            scratch: vec![0.0; n],
            rtol: 1e-6,
            atol: 1e-9,
            max_substeps: 2048,
            last_error: 0.0,
            last_substeps: 0,
        }
    }

    /// Configures relative and absolute local error tolerances.
    #[verified_engine::verified]
    pub fn with_tolerances(mut self, rtol: f64, atol: f64) -> Self {
        self.rtol = rtol.max(1e-14);
        self.atol = atol.max(1e-15);
        self
    }

    /// Configures the upper bound on adaptive substeps per outer `step` call.
    #[verified_engine::verified]
    pub fn with_max_substeps(mut self, max_substeps: usize) -> Self {
        self.max_substeps = max_substeps.max(1);
        self
    }

    /// Returns the maximum scaled error norm observed across accepted substeps in the last `step`.
    #[verified_engine::verified]
    pub fn last_error(&self) -> f64 {
        self.last_error
    }

    /// Returns the number of accepted adaptive substeps taken in the last `step`.
    #[verified_engine::verified]
    pub fn last_substeps(&self) -> usize {
        self.last_substeps
    }

    #[verified_engine::verified]
    fn compute_derivatives<S>(&mut self, system: &S, t: f64, state: &State, n: usize) -> f64
    where
        S: OdeSystem<State> + ?Sized,
    {
        system.derivative_in_place(t, state, &mut self.f0);
        self.y_stage.copy_from(state);
        let mut max_row_sum = 0.0;
        for i in 0..n {
            self.scratch[i] = 0.0;
        }
        for j in 0..n {
            let yj = state.get(j);
            let h_fd = FD_REL_STEP * yj.abs().max(1.0);
            self.y_stage.set(j, yj + h_fd);
            system.derivative_in_place(t, &self.y_stage, &mut self.f_pert);
            self.y_stage.set(j, yj);
            let inv_h = 1.0 / h_fd;
            for i in 0..n {
                let j_ij = (self.f_pert.get(i) - self.f0.get(i)) * inv_h;
                let idx = math_commons::math_kernel::types::flatten_2d_index(j, i, n);
                self.jac[idx] = j_ij;
                self.scratch[i] += j_ij.abs();
            }
        }
        for i in 0..n {
            if self.scratch[i] > max_row_sum {
                max_row_sum = self.scratch[i];
            }
        }
        let dt_fd = FD_REL_STEP * t.abs().max(1.0);
        system.derivative_in_place(t + dt_fd, state, &mut self.f_pert);
        system.derivative_in_place(t - dt_fd, state, &mut self.f_t_minus);
        let inv_2dt = 0.5 / dt_fd;
        for i in 0..n {
            self.dfdt[i] = (self.f_pert.get(i) - self.f_t_minus.get(i)) * inv_2dt;
        }
        max_row_sum
    }

    #[verified_engine::verified]
    fn try_substep<S>(&mut self, system: &S, t: f64, state: &State, h: f64, n: usize) -> Option<f64>
    where
        S: OdeSystem<State> + ?Sized,
    {
        let hd = h * ROS_D;
        for i in 0..n {
            for j in 0..n {
                let diag = if i == j { 1.0 } else { 0.0 };
                let idx = math_commons::math_kernel::types::flatten_2d_index(j, i, n);
                self.lu[idx] = diag - hd * self.jac[idx];
            }
            self.k1[i] = self.f0.get(i) + hd * self.dfdt[i];
        }
        if !lu_factor_in_place(&mut self.lu, &mut self.pivots, n, PIVOT_TOL) {
            return None;
        }
        if !lu_solve_in_place(&self.lu, &self.pivots, &mut self.k1, &mut self.scratch, n) {
            return None;
        }
        for i in 0..n {
            self.y_stage.set(i, state.get(i) + 0.5 * h * self.k1[i]);
        }
        system.derivative_in_place(t + 0.5 * h, &self.y_stage, &mut self.f1);
        for i in 0..n {
            self.k2[i] = self.f1.get(i) - self.k1[i];
        }
        if !lu_solve_in_place(&self.lu, &self.pivots, &mut self.k2, &mut self.scratch, n) {
            return None;
        }
        for i in 0..n {
            self.k2[i] += self.k1[i];
            self.y_next.set(i, state.get(i) + h * self.k2[i]);
        }
        system.derivative_in_place(t + h, &self.y_next, &mut self.f2);
        for i in 0..n {
            self.k3[i] = self.f2.get(i)
                - ROS_E32 * (self.k2[i] - self.f1.get(i))
                - 2.0 * (self.k1[i] - self.f0.get(i))
                + hd * self.dfdt[i];
        }
        if !lu_solve_in_place(&self.lu, &self.pivots, &mut self.k3, &mut self.scratch, n) {
            return None;
        }
        Some(self.compute_error_norm(state, h, n))
    }

    #[verified_engine::verified]
    fn compute_error_norm(&self, state: &State, h: f64, n: usize) -> f64 {
        let mut sum_sq = 0.0;
        for i in 0..n {
            let y0 = state.get(i);
            let y1 = self.y_next.get(i);
            let err = (h / 6.0) * (self.k1[i] - 2.0 * self.k2[i] + self.k3[i]);
            if !y1.is_finite() || !err.is_finite() {
                return f64::INFINITY;
            }
            let scale = self.atol + self.rtol * y0.abs().max(y1.abs());
            let ratio = err / scale;
            sum_sq += ratio * ratio;
        }
        (sum_sq / (n as f64)).sqrt()
    }

    /// L-stable linearly-implicit Euler fallback (`(I - h J) dy = h (f0 + h df/dt)`) used only
    /// when a step is already inside the `atol` equilibrium band (`err_norm <= 1`) at huge `|h|*||J||`.
    #[verified_engine::verified]
    fn apply_equilibrium_l_stable_step(&mut self, state: &State, h: f64, n: usize) -> bool {
        for i in 0..n {
            for j in 0..n {
                let diag = if i == j { 1.0 } else { 0.0 };
                let idx = math_commons::math_kernel::types::flatten_2d_index(j, i, n);
                self.lu[idx] = diag - h * self.jac[idx];
            }
            self.scratch[i] = h * (self.f0.get(i) + h * self.dfdt[i]);
        }
        if !solve_linear_in_place(&mut self.lu, &mut self.scratch, n, PIVOT_TOL) {
            return false;
        }
        for i in 0..n {
            self.y_next.set(i, state.get(i) + self.scratch[i]);
        }
        true
    }
}

impl<State: IndexedState> Solver<State> for Rosenbrock23<State> {
    #[verified_engine::verified]
    fn step<S>(&mut self, system: &S, t: f64, state: &mut State, dt: f64)
    where
        S: OdeSystem<State> + ?Sized,
    {
        self.last_error = 0.0;
        self.last_substeps = 0;
        let n = state.len();
        if !dt.is_finite() || dt == 0.0 || n == 0 {
            return;
        }
        let t_end = t + dt;
        let h_min = 1e-14 * dt.abs().max(1.0);
        let tol_time = 1e-15 * dt.abs().max(1.0);
        let mut t_cur = t;
        let mut h = dt;
        let mut need_derivs = true;
        let mut j_norm = 0.0;

        for _ in 0..self.max_substeps {
            let rem = t_end - t_cur;
            if rem.abs() <= tol_time {
                break;
            }
            if h.abs() > rem.abs() || (h > 0.0) != (rem > 0.0) {
                h = rem;
            }
            if need_derivs {
                j_norm = self.compute_derivatives(system, t_cur, state, n);
                need_derivs = false;
            }
            let Some(err_norm) = self.try_substep(system, t_cur, state, h, n) else {
                h *= 0.25;
                if h.abs() <= h_min {
                    break;
                }
                continue;
            };
            if !err_norm.is_finite() {
                h *= 0.25;
                if h.abs() <= h_min {
                    break;
                }
                continue;
            }
            if err_norm <= 1.0 || h.abs() <= h_min {
                if h.abs() * j_norm > POSITIVITY_RHO_LIMIT {
                    let _ = self.apply_equilibrium_l_stable_step(state, h, n);
                }
                state.copy_from(&self.y_next);
                t_cur += h;
                need_derivs = true;
                self.last_substeps += 1;
                if err_norm > self.last_error {
                    self.last_error = err_norm;
                }
                let growth = if err_norm == 0.0 {
                    4.0
                } else {
                    (0.9 * err_norm.powf(-1.0 / 3.0)).clamp(0.2, 4.0)
                };
                h *= growth;
            } else {
                let shrink = (0.9 * err_norm.powf(-1.0 / 3.0)).clamp(0.1, 0.5);
                h *= shrink;
            }
        }
    }
}
