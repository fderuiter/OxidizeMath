//! Zero-allocation Lyapunov exponent $\lambda(r)$ sweep for logistic and custom 1D maps.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:chaos_theory]

use super::{BifurcationDiagram, BifurcationMapMode, CUSTOM_BURN_IN, CUSTOM_ESCAPE_GUARD, CUSTOM_ORBIT_SAMPLES};
use math_commons::ast::CompiledExpr;
use egui_plot::PlotPoint;
use math_explorer::physics::chaos::metrics::logistic_lyapunov;

const LOGISTIC_SEED: f64 = 0.213_456_7;
const LOGISTIC_BURN_IN: usize = 200;
const LOGISTIC_LYAPUNOV_STEPS: usize = 240;
const CUSTOM_FD_EPS: f64 = 1e-6;
const LYAPUNOV_CLAMP: f64 = 4.0;

impl BifurcationDiagram {
    pub(super) fn recompute_lyapunov(&mut self) {
        let steps = self.steps.clamp(2, 2000);
        let span = self.r_max - self.r_min;
        if !span.is_finite() || span <= 0.0 {
            self.lyapunov_points.clear();
            return;
        }
        if self.lyapunov_points.capacity() < steps {
            self.lyapunov_points
                .reserve(steps - self.lyapunov_points.capacity());
        }
        self.lyapunov_points.clear();

        let denom = (steps - 1) as f64;
        match self.map_mode {
            BifurcationMapMode::Logistic => {
                for idx in 0..steps {
                    let r = self.r_min + (idx as f64 / denom) * span;
                    let lam = eval_logistic_lyapunov_at(r);
                    if lam.is_finite() {
                        self.lyapunov_points
                            .push(PlotPoint::new(r, lam.clamp(-LYAPUNOV_CLAMP, LYAPUNOV_CLAMP)));
                    }
                }
            }
            BifurcationMapMode::Custom => {
                let Some(compiled) = self.math_input.active_compiled() else {
                    return;
                };
                let params = self.math_input.param_slice();
                for idx in 0..steps {
                    let r = self.r_min + (idx as f64 / denom) * span;
                    if let Some(lam) = eval_custom_lyapunov_at(compiled, params, r, self.x0) {
                        self.lyapunov_points
                            .push(PlotPoint::new(r, lam.clamp(-LYAPUNOV_CLAMP, LYAPUNOV_CLAMP)));
                    }
                }
            }
        }
    }

    /// Evaluates the maximal Lyapunov exponent $\lambda(r)$ at the current `probe_r` parameter.
    #[must_use]
    pub fn probe_lyapunov(&self) -> Option<f64> {
        match self.map_mode {
            BifurcationMapMode::Logistic => {
                let lam = eval_logistic_lyapunov_at(self.probe_r);
                lam.is_finite().then_some(lam)
            }
            BifurcationMapMode::Custom => {
                let compiled = self.math_input.active_compiled()?;
                eval_custom_lyapunov_at(
                    compiled,
                    self.math_input.param_slice(),
                    self.probe_r,
                    self.x0,
                )
            }
        }
    }
}

pub(super) fn eval_logistic_lyapunov_at(r: f64) -> f64 {
    let mut x = LOGISTIC_SEED;
    for _ in 0..LOGISTIC_BURN_IN {
        x = r * x * (1.0 - x);
    }
    logistic_lyapunov(r, x, LOGISTIC_LYAPUNOV_STEPS)
}

pub(super) fn eval_custom_lyapunov_at(
    compiled: &CompiledExpr,
    params: &[f64],
    r: f64,
    x0: f64,
) -> Option<f64> {
    let mut x = x0;
    if !x.is_finite() || x.abs() > CUSTOM_ESCAPE_GUARD {
        return None;
    }
    for _ in 0..CUSTOM_BURN_IN {
        x = compiled.eval_2d(r, x, params);
        if !x.is_finite() || x.abs() > CUSTOM_ESCAPE_GUARD {
            return None;
        }
    }
    let mut sum_ln = 0.0;
    let mut count = 0_usize;
    for _ in 0..CUSTOM_ORBIT_SAMPLES {
        let f_plus = compiled.eval_2d(r, x + CUSTOM_FD_EPS, params);
        let f_minus = compiled.eval_2d(r, x - CUSTOM_FD_EPS, params);
        if !f_plus.is_finite() || !f_minus.is_finite() {
            break;
        }
        let deriv_abs = ((f_plus - f_minus) / (2.0 * CUSTOM_FD_EPS)).abs();
        sum_ln += deriv_abs.max(1e-15).ln();
        count += 1;
        x = compiled.eval_2d(r, x, params);
        if !x.is_finite() || x.abs() > CUSTOM_ESCAPE_GUARD {
            break;
        }
    }
    (count > 0).then_some(sum_ln / (count as f64))
}
