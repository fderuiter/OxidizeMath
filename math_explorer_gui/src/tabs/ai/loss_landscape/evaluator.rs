#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Unified surface evaluator combining analytical potentials, empirical regression objectives
//! (`MSE`, `MAE`, `Huber`) over preset/formula/table targets, custom compiled `CompiledExpr`
//! predictors, `SolverWorkspace`-backed reverse-mode gradients, and sculpted Gaussian barriers/wells.

use super::data::LossMetric;
use super::surfaces::{LandscapeSurface, SculptBump};
use engine_symbolic::SolverWorkspace;
use math_commons::ast::CompiledExpr;
use nalgebra::Vector2;

const MAX_PREDICTOR_VARS: usize = 32;

/// Zero-allocation empirical dataset and loss configuration for [`LandscapeSurface::LinearMse`].
#[derive(Clone, Copy)]
pub struct EmpiricalConfig<'a> {
    pub targets: &'a [[f64; 2]],
    pub metric: LossMetric,
    pub huber_delta: f64,
    pub is_preset_linear_target: bool,
    pub custom_predictor: Option<(&'a CompiledExpr, &'a [f64])>,
}

impl<'a> EmpiricalConfig<'a> {
    #[must_use]
    pub fn evaluate_base(self, w: f64, b: f64) -> Option<f64> {
        let n = self.targets.len();
        if n == 0 {
            return None;
        }
        let mut total_loss = 0.0;
        match self.custom_predictor {
            None => {
                for &[x, y] in self.targets {
                    let err = (w * x + b - y).clamp(-1e6, 1e6);
                    total_loss += self.metric.point_loss(err, self.huber_delta);
                }
            }
            Some((compiled, params)) => {
                for &[x, y] in self.targets {
                    let pred = compiled.eval_3d(x, w, b, params);
                    if !pred.is_finite() {
                        return None;
                    }
                    let err = (pred - y).clamp(-1e6, 1e6);
                    total_loss += self.metric.point_loss(err, self.huber_delta);
                }
            }
        }
        let loss = total_loss / (n as f64);
        loss.is_finite().then_some(loss)
    }

    #[must_use]
    pub fn gradient_base(self, w: f64, b: f64) -> Vector2<f64> {
        let n = self.targets.len();
        if n == 0 {
            return Vector2::zeros();
        }
        let inv_n = 1.0 / (n as f64);
        let (mut dw_sum, mut db_sum) = (0.0, 0.0);
        match self.custom_predictor {
            None => {
                for &[x, y] in self.targets {
                    let raw = w * x + b - y;
                    if (-1e6..=1e6).contains(&raw) {
                        let dl = self.metric.point_loss_derivative(raw, self.huber_delta);
                        dw_sum += dl * x;
                        db_sum += dl;
                    }
                }
            }
            Some((compiled, params)) => {
                let p_len = params.len().min(MAX_PREDICTOR_VARS - 3);
                let total_vars = 3 + p_len;
                let mut vars = [0.0_f64; MAX_PREDICTOR_VARS];
                let mut grad_vars = [0.0_f64; MAX_PREDICTOR_VARS];
                vars[1] = w;
                vars[2] = b;
                vars[3..total_vars].copy_from_slice(&params[..p_len]);
                let mut ws = SolverWorkspace::new();

                for &[x, y] in self.targets {
                    vars[0] = x;
                    let pred = compiled.eval_grad_into(
                        &vars[..total_vars],
                        &mut ws,
                        &mut grad_vars[..total_vars],
                    );
                    if !pred.is_finite() {
                        return Vector2::zeros();
                    }
                    let raw = pred - y;
                    if (-1e6..=1e6).contains(&raw) {
                        let dl = self.metric.point_loss_derivative(raw, self.huber_delta);
                        if grad_vars[1].is_finite() {
                            dw_sum += dl * grad_vars[1];
                        }
                        if grad_vars[2].is_finite() {
                            db_sum += dl * grad_vars[2];
                        }
                    }
                }
            }
        }
        Vector2::new(
            (dw_sum * inv_n).clamp(-1e6, 1e6),
            (db_sum * inv_n).clamp(-1e6, 1e6),
        )
    }
}

/// Zero-allocation potential and gradient evaluator over benchmark or empirical surfaces.
#[derive(Clone, Copy)]
pub struct SurfaceEvaluator<'a> {
    pub surface: LandscapeSurface,
    pub bumps: &'a [SculptBump],
    pub empirical: Option<EmpiricalConfig<'a>>,
}

impl<'a> SurfaceEvaluator<'a> {
    #[inline]
    #[must_use]
    pub const fn new(
        surface: LandscapeSurface,
        bumps: &'a [SculptBump],
        empirical: Option<EmpiricalConfig<'a>>,
    ) -> Self {
        Self {
            surface,
            bumps,
            empirical,
        }
    }

    #[inline]
    #[must_use]
    pub const fn global_minima(self) -> &'static [[f64; 2]] {
        if matches!(self.surface, LandscapeSurface::LinearMse) {
            if let Some(emp) = self.empirical {
                if emp.custom_predictor.is_some() || !emp.is_preset_linear_target {
                    return &[];
                }
            }
        }
        self.surface.global_minima()
    }

    #[inline]
    #[must_use]
    pub fn evaluate(self, w: f64, b: f64) -> f64 {
        let base = match (self.surface, self.empirical) {
            (LandscapeSurface::LinearMse, Some(emp)) => emp.evaluate_base(w, b).unwrap_or(f64::NAN),
            _ => self.surface.base_loss(w, b),
        };
        if !base.is_finite() {
            return f64::NAN;
        }
        let mut total = base;
        for bump in self.bumps {
            total += bump.evaluate(w, b);
        }
        total
    }

    #[inline]
    #[must_use]
    pub fn gradient(self, w: f64, b: f64) -> Vector2<f64> {
        let mut grad = match (self.surface, self.empirical) {
            (LandscapeSurface::LinearMse, Some(emp)) => emp.gradient_base(w, b),
            _ => self.surface.base_gradient(w, b),
        };
        for bump in self.bumps {
            grad += bump.gradient(w, b);
        }
        grad
    }
}
