//! Embedded adaptive Dormand-Prince RK5(4)7M ODE solver (`DormandPrince45`).
//!
//! Pre-allocates all seven Runge-Kutta stage buffers at construction time so [`Solver::step`]
//! executes adaptive substepping with zero heap allocations.

use super::traits::{IndexedState, OdeSystem, Solver};

const C2: f64 = 1.0 / 5.0;
const C3: f64 = 3.0 / 10.0;
const C4: f64 = 4.0 / 5.0;
const C5: f64 = 8.0 / 9.0;

const A21: f64 = 1.0 / 5.0;
const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;
const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;
const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;

const B1: f64 = 35.0 / 384.0;
const B3: f64 = 500.0 / 1113.0;
const B4: f64 = 125.0 / 192.0;
const B5: f64 = -2187.0 / 6784.0;
const B6: f64 = 11.0 / 84.0;

const E1: f64 = 71.0 / 57600.0;
const E3: f64 = -71.0 / 16695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17253.0 / 339200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;

/// Adaptive 5th(4th)-order Dormand-Prince Runge-Kutta integrator (`DOPRI5` / `ode45`).
///
/// Uses seven stage evaluations per substep with embedded 4th/5th-order local error control
/// and pre-allocated stage vectors for zero-allocation stepping.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::ode::{
///     ArrayState, DormandPrince45, OdeModel, OdeSystem, TimeStepper,
/// };
///
/// struct LinearRamp;
/// impl OdeSystem<ArrayState<1>> for LinearRamp {
///     fn derivative(&self, t: f64, _s: &ArrayState<1>) -> ArrayState<1> {
///         ArrayState([t])
///     }
/// }
///
/// let s0 = ArrayState([0.0]);
/// let mut model = OdeModel::new(s0, LinearRamp, DormandPrince45::new(&s0));
/// model.step(10.0);
/// assert!((model.get_state().0[0] - 50.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct DormandPrince45<State: IndexedState> {
    k1: State,
    k2: State,
    k3: State,
    k4: State,
    k5: State,
    k6: State,
    k7: State,
    y_stage: State,
    y_next: State,
    y_err: State,
    rtol: f64,
    atol: f64,
    max_substeps: usize,
    last_error: f64,
    last_substeps: usize,
}

impl<State: IndexedState> DormandPrince45<State> {
    /// Creates a new `DormandPrince45` solver with stage buffers sized from `template`.
    #[verified_engine::verified]
    pub fn new(template: &State) -> Self {
        Self {
            k1: template.clone(),
            k2: template.clone(),
            k3: template.clone(),
            k4: template.clone(),
            k5: template.clone(),
            k6: template.clone(),
            k7: template.clone(),
            y_stage: template.clone(),
            y_next: template.clone(),
            y_err: template.clone(),
            rtol: 1e-8,
            atol: 1e-10,
            max_substeps: 2048,
            last_error: 0.0,
            last_substeps: 0,
        }
    }

    /// Configures relative and absolute local error tolerances.
    #[verified_engine::verified]
    pub fn with_tolerances(mut self, rtol: f64, atol: f64) -> Self {
        self.rtol = rtol.max(1e-15);
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
    fn evaluate_stages<S>(&mut self, system: &S, t: f64, state: &State, h: f64)
    where
        S: OdeSystem<State> + ?Sized,
    {
        system.derivative_in_place(t, state, &mut self.k1);

        self.y_stage.copy_from_scaled(state, &self.k1, h * A21);
        system.derivative_in_place(t + C2 * h, &self.y_stage, &mut self.k2);

        self.y_stage.copy_from_scaled(state, &self.k1, h * A31);
        self.y_stage.scale_add(&self.k2, h * A32);
        system.derivative_in_place(t + C3 * h, &self.y_stage, &mut self.k3);

        self.y_stage.copy_from_scaled(state, &self.k1, h * A41);
        self.y_stage.scale_add(&self.k2, h * A42);
        self.y_stage.scale_add(&self.k3, h * A43);
        system.derivative_in_place(t + C4 * h, &self.y_stage, &mut self.k4);

        self.y_stage.copy_from_scaled(state, &self.k1, h * A51);
        self.y_stage.scale_add(&self.k2, h * A52);
        self.y_stage.scale_add(&self.k3, h * A53);
        self.y_stage.scale_add(&self.k4, h * A54);
        system.derivative_in_place(t + C5 * h, &self.y_stage, &mut self.k5);

        self.y_stage.copy_from_scaled(state, &self.k1, h * A61);
        self.y_stage.scale_add(&self.k2, h * A62);
        self.y_stage.scale_add(&self.k3, h * A63);
        self.y_stage.scale_add(&self.k4, h * A64);
        self.y_stage.scale_add(&self.k5, h * A65);
        system.derivative_in_place(t + h, &self.y_stage, &mut self.k6);

        self.y_next.copy_from_scaled(state, &self.k1, h * B1);
        self.y_next.scale_add(&self.k3, h * B3);
        self.y_next.scale_add(&self.k4, h * B4);
        self.y_next.scale_add(&self.k5, h * B5);
        self.y_next.scale_add(&self.k6, h * B6);
        system.derivative_in_place(t + h, &self.y_next, &mut self.k7);

        self.y_err.copy_from(&self.k1);
        self.y_err *= h * E1;
        self.y_err.scale_add(&self.k3, h * E3);
        self.y_err.scale_add(&self.k4, h * E4);
        self.y_err.scale_add(&self.k5, h * E5);
        self.y_err.scale_add(&self.k6, h * E6);
        self.y_err.scale_add(&self.k7, h * E7);
    }

    #[verified_engine::verified]
    fn scaled_error_norm(&self, state: &State) -> f64 {
        let n = state.len();
        if n == 0 {
            return 0.0;
        }
        let mut sum_sq = 0.0;
        for i in 0..n {
            let y0 = state.get(i);
            let y1 = self.y_next.get(i);
            let err = self.y_err.get(i);
            if !y1.is_finite() || !err.is_finite() {
                return f64::INFINITY;
            }
            let scale = self.atol + self.rtol * y0.abs().max(y1.abs());
            let ratio = err / scale;
            sum_sq += ratio * ratio;
        }
        (sum_sq / (n as f64)).sqrt()
    }
}

impl<State: IndexedState> Solver<State> for DormandPrince45<State> {
    #[verified_engine::verified]
    fn step<S>(&mut self, system: &S, t: f64, state: &mut State, dt: f64)
    where
        S: OdeSystem<State> + ?Sized,
    {
        self.last_error = 0.0;
        self.last_substeps = 0;
        if !dt.is_finite() || dt == 0.0 || state.is_empty() {
            return;
        }

        let t_end = t + dt;
        let h_min = 1e-14 * dt.abs().max(1.0);
        let tol_time = 1e-15 * dt.abs().max(1.0);
        let mut t_cur = t;
        let mut h = dt;

        for _ in 0..self.max_substeps {
            let rem = t_end - t_cur;
            if rem.abs() <= tol_time {
                break;
            }
            if h.abs() > rem.abs() || (h > 0.0) != (rem > 0.0) {
                h = rem;
            }
            self.evaluate_stages(system, t_cur, state, h);
            let err_norm = self.scaled_error_norm(state);
            if !err_norm.is_finite() {
                h *= 0.25;
                if h.abs() <= h_min {
                    break;
                }
                continue;
            }
            if err_norm <= 1.0 || h.abs() <= h_min {
                state.copy_from(&self.y_next);
                t_cur += h;
                self.last_substeps += 1;
                if err_norm > self.last_error {
                    self.last_error = err_norm;
                }
                let growth = if err_norm == 0.0 {
                    5.0
                } else {
                    (0.9 * err_norm.powf(-0.2)).clamp(0.2, 5.0)
                };
                h *= growth;
            } else {
                let shrink = (0.9 * err_norm.powf(-0.25)).clamp(0.1, 0.5);
                h *= shrink;
            }
        }
    }
}
