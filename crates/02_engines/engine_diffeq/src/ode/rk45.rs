//! Adaptive Dormand-Prince `RK45` (`DOPRI5`) solver with proportional-integral (PI) step-size control.
// [cite:engine_diffeq]

use super::traits::{OdeState, OdeSystem, SliceOdeSystem, Solver};
use crate::error::DiffeqError;
use crate::math::{pow_pos_f64, sqrt_f64};
use crate::workspace::SolverWorkspace;

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
const A51: f64 = 19_372.0 / 6561.0;
const A52: f64 = -25_360.0 / 2187.0;
const A53: f64 = 64_448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46_732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18_656.0;

const B1: f64 = 35.0 / 384.0;
const B3: f64 = 500.0 / 1113.0;
const B4: f64 = 125.0 / 192.0;
const B5: f64 = -2187.0 / 6784.0;
const B6: f64 = 11.0 / 84.0;

const E1: f64 = 71.0 / 57_600.0;
const E3: f64 = -71.0 / 16_695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17_253.0 / 339_200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;

const MAX_RK45_SUBSTEPS: usize = 2048;

/// Embedded 7-stage FSAL Dormand-Prince `RK45` adaptive solver with proportional-integral step control.
#[derive(Debug, Clone)]
pub struct DormandPrince45<State> {
    k1: State,
    k2: State,
    k3: State,
    k4: State,
    k5: State,
    k6: State,
    k7: State,
    trial: State,
    err_vec: State,
    rtol: f64,
    atol: f64,
    err_prev: f64,
    h_hint: f64,
}

impl<State: OdeState> DormandPrince45<State> {
    /// Creates a `DormandPrince45` solver with default tolerances (`rtol = 1e-8`, `atol = 1e-10`).
    #[must_use]
    #[verified_engine::verified]
    pub fn new(template: &State) -> Self {
        Self::with_tolerances(template, 1e-8, 1e-10)
    }

    /// Creates a `DormandPrince45` solver with custom relative and absolute tolerances.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_tolerances(template: &State, rtol: f64, atol: f64) -> Self {
        Self {
            k1: template.clone(),
            k2: template.clone(),
            k3: template.clone(),
            k4: template.clone(),
            k5: template.clone(),
            k6: template.clone(),
            k7: template.clone(),
            trial: template.clone(),
            err_vec: template.clone(),
            rtol: rtol.max(1e-15),
            atol: atol.max(1e-15),
            err_prev: 1e-4,
            h_hint: 0.0,
        }
    }

    #[verified_engine::verified]
    fn evaluate_stages<S: OdeSystem<State> + ?Sized>(
        &mut self,
        system: &S,
        t: f64,
        y: &State,
        h: f64,
    ) {
        system.derivative_in_place(t, y, &mut self.k1);

        self.trial.copy_from_scaled(y, &self.k1, h * A21);
        system.derivative_in_place(t + C2 * h, &self.trial, &mut self.k2);

        self.trial.copy_from_scaled(y, &self.k1, h * A31);
        self.trial.scale_add(&self.k2, h * A32);
        system.derivative_in_place(t + C3 * h, &self.trial, &mut self.k3);

        self.trial.copy_from_scaled(y, &self.k1, h * A41);
        self.trial.scale_add(&self.k2, h * A42);
        self.trial.scale_add(&self.k3, h * A43);
        system.derivative_in_place(t + C4 * h, &self.trial, &mut self.k4);

        self.trial.copy_from_scaled(y, &self.k1, h * A51);
        self.trial.scale_add(&self.k2, h * A52);
        self.trial.scale_add(&self.k3, h * A53);
        self.trial.scale_add(&self.k4, h * A54);
        system.derivative_in_place(t + C5 * h, &self.trial, &mut self.k5);

        self.trial.copy_from_scaled(y, &self.k1, h * A61);
        self.trial.scale_add(&self.k2, h * A62);
        self.trial.scale_add(&self.k3, h * A63);
        self.trial.scale_add(&self.k4, h * A64);
        self.trial.scale_add(&self.k5, h * A65);
        system.derivative_in_place(t + h, &self.trial, &mut self.k6);

        self.trial.copy_from_scaled(y, &self.k1, h * B1);
        self.trial.scale_add(&self.k3, h * B3);
        self.trial.scale_add(&self.k4, h * B4);
        self.trial.scale_add(&self.k5, h * B5);
        self.trial.scale_add(&self.k6, h * B6);
        system.derivative_in_place(t + h, &self.trial, &mut self.k7);

        self.err_vec.copy_from(&self.k1);
        self.err_vec *= h * E1;
        self.err_vec.scale_add(&self.k3, h * E3);
        self.err_vec.scale_add(&self.k4, h * E4);
        self.err_vec.scale_add(&self.k5, h * E5);
        self.err_vec.scale_add(&self.k6, h * E6);
        self.err_vec.scale_add(&self.k7, h * E7);
    }

    #[verified_engine::verified]
    fn scaled_error_norm(&self, y: &State) -> f64 {
        let dim = y.state_dim();
        if dim == 0 {
            return 0.0;
        }
        let mut sum_sq = 0.0_f64;
        for i in 0..dim {
            let yi = y.component(i);
            let y_new = self.trial.component(i);
            let ei = self.err_vec.component(i);
            if !y_new.is_finite() || !ei.is_finite() {
                return f64::INFINITY;
            }
            let sc = self.atol + self.rtol * yi.abs().max(y_new.abs());
            let ratio = ei / sc;
            sum_sq += ratio * ratio;
        }
        sqrt_f64(sum_sq / (dim as f64))
    }
}

#[verified_engine::verified]
fn pi_accept_factor(err: f64, err_prev: f64) -> f64 {
    if err <= 1e-16 {
        return 5.0;
    }
    let alpha = 0.14;
    let beta = 0.08;
    let raw = 0.9 * pow_pos_f64(err, -alpha) * pow_pos_f64(err_prev.max(1e-4), beta);
    raw.clamp(0.2, 5.0)
}

#[verified_engine::verified]
fn pi_reject_factor(err: f64) -> f64 {
    if !err.is_finite() {
        return 0.2;
    }
    (0.9 * pow_pos_f64(err, -0.2)).clamp(0.1, 0.5)
}

impl<State: OdeState> Solver<State> for DormandPrince45<State> {
    #[verified_engine::verified]
    fn step<S>(&mut self, system: &S, t: f64, state: &mut State, dt: f64)
    where
        S: OdeSystem<State> + ?Sized,
    {
        if !dt.is_finite() || dt.abs() <= 1e-30 {
            return;
        }
        let sign = if dt >= 0.0 { 1.0 } else { -1.0 };
        let t_end = t + dt;
        let mut t_cur = t;
        let min_h = dt.abs() * 1e-12;
        let mut h_abs = if self.h_hint > min_h && self.h_hint < dt.abs() {
            self.h_hint
        } else {
            dt.abs()
        };

        for _ in 0..MAX_RK45_SUBSTEPS {
            let rem = (t_end - t_cur) * sign;
            if rem <= 1e-15 * dt.abs().max(1.0) {
                break;
            }
            let clipped = h_abs >= rem;
            let h_use = if clipped { rem } else { h_abs };
            self.evaluate_stages(system, t_cur, state, sign * h_use);
            let err = self.scaled_error_norm(state);

            if err <= 1.0 || h_use <= min_h {
                state.copy_from(&self.trial);
                t_cur += sign * h_use;
                let fac = pi_accept_factor(err, self.err_prev);
                self.err_prev = err.max(1e-4);
                let grown = h_use * fac;
                if !clipped || grown < h_abs {
                    h_abs = grown;
                }
            } else {
                h_abs = (h_use * pi_reject_factor(err)).max(min_h);
            }
        }
        self.h_hint = h_abs;
    }
}

struct Rk45SliceStep {
    t: f64,
    h: f64,
    rtol: f64,
    atol: f64,
    n: usize,
}

/// Advances a caller-owned slice `state` over `[t, t + dt]` using Dormand-Prince `RK45` and `workspace` (`>= 9 * dim`).
#[verified_engine::verified]
pub fn step_slice_rk45<S: SliceOdeSystem + ?Sized>(
    system: &S,
    t: f64,
    state: &mut [f64],
    dt: f64,
    rtol: f64,
    atol: f64,
    workspace: &mut SolverWorkspace<'_>,
) -> Result<(), DiffeqError> {
    if !dt.is_finite() || dt == 0.0 {
        return Err(DiffeqError::InvalidStepSize { dt });
    }
    if !rtol.is_finite() || rtol <= 0.0 || !atol.is_finite() || atol <= 0.0 {
        return Err(DiffeqError::InvalidTolerance {
            tol: rtol.min(atol),
        });
    }
    let n = state.len();
    let buf = workspace.try_slice_mut(SolverWorkspace::rk45_scratch_len(n))?;
    let sign = if dt >= 0.0 { 1.0 } else { -1.0 };
    let t_end = t + dt;
    let mut t_cur = t;
    let mut h_abs = dt.abs();
    let mut err_prev = 1e-4_f64;
    let min_h = dt.abs() * 1e-12;

    for _ in 0..MAX_RK45_SUBSTEPS {
        let rem = (t_end - t_cur) * sign;
        if rem <= 1e-15 * dt.abs().max(1.0) {
            return Ok(());
        }
        let h_use = h_abs.min(rem);
        let cfg = Rk45SliceStep {
            t: t_cur,
            h: sign * h_use,
            rtol,
            atol,
            n,
        };
        let err = eval_rk45_slice_trial(system, state, &cfg, buf);
        if err <= 1.0 || h_use <= min_h {
            let trial_off = 7 * n;
            state.copy_from_slice(&buf[trial_off..trial_off + n]);
            t_cur += sign * h_use;
            h_abs = h_use * pi_accept_factor(err, err_prev);
            err_prev = err.max(1e-4);
        } else {
            h_abs = (h_use * pi_reject_factor(err)).max(min_h);
        }
    }
    Err(DiffeqError::MaxIterationsExceeded {
        max_iter: MAX_RK45_SUBSTEPS,
        residual: (t_end - t_cur).abs(),
    })
}

#[verified_engine::verified]
fn eval_rk45_slice_trial<S: SliceOdeSystem + ?Sized>(
    sys: &S,
    y: &[f64],
    cfg: &Rk45SliceStep,
    buf: &mut [f64],
) -> f64 {
    let n = cfg.n;
    let t = cfg.t;
    let h = cfg.h;
    let (k1, rest) = buf.split_at_mut(n);
    let (k2, rest) = rest.split_at_mut(n);
    let (k3, rest) = rest.split_at_mut(n);
    let (k4, rest) = rest.split_at_mut(n);
    let (k5, rest) = rest.split_at_mut(n);
    let (k6, rest) = rest.split_at_mut(n);
    let (k7, rest) = rest.split_at_mut(n);
    let (trial, err_v) = rest.split_at_mut(n);

    sys.derivative_slice(t, y, k1);
    for i in 0..n {
        trial[i] = y[i] + h * A21 * k1[i];
    }
    sys.derivative_slice(t + C2 * h, trial, k2);
    for i in 0..n {
        trial[i] = y[i] + h * (A31 * k1[i] + A32 * k2[i]);
    }
    sys.derivative_slice(t + C3 * h, trial, k3);
    for i in 0..n {
        trial[i] = y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
    }
    sys.derivative_slice(t + C4 * h, trial, k4);
    for i in 0..n {
        trial[i] = y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
    }
    sys.derivative_slice(t + C5 * h, trial, k5);
    for i in 0..n {
        trial[i] = y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
    }
    sys.derivative_slice(t + h, trial, k6);
    for i in 0..n {
        trial[i] = y[i] + h * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] + B6 * k6[i]);
    }
    sys.derivative_slice(t + h, trial, k7);

    let mut sum_sq = 0.0_f64;
    for i in 0..n {
        err_v[i] =
            h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
        if !trial[i].is_finite() || !err_v[i].is_finite() {
            return f64::INFINITY;
        }
        let sc = cfg.atol + cfg.rtol * y[i].abs().max(trial[i].abs());
        let r = err_v[i] / sc;
        sum_sq += r * r;
    }
    if n == 0 {
        0.0
    } else {
        sqrt_f64(sum_sq / (n as f64))
    }
}
