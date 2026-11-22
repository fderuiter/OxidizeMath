#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Unified surface evaluator combining analytical potentials, custom compiled `CompiledExpr`
//! predictor MSE, central finite-difference gradients, and sculpted Gaussian barriers/wells.

use super::surfaces::{LandscapeSurface, SculptBump, LINEAR_MSE_X, LINEAR_MSE_Y};
use math_commons::ast::CompiledExpr;
use nalgebra::Vector2;

const CUSTOM_FD_STEP: f64 = 1e-5;

/// Zero-allocation potential and gradient evaluator over benchmark or custom MSE surfaces.
#[derive(Clone, Copy)]
pub struct SurfaceEvaluator<'a> {
    pub surface: LandscapeSurface,
    pub bumps: &'a [SculptBump],
    pub custom_mse: Option<(&'a CompiledExpr, &'a [f64])>,
}

impl<'a> SurfaceEvaluator<'a> {
    #[inline]
    #[must_use]
    pub const fn new(
        surface: LandscapeSurface,
        bumps: &'a [SculptBump],
        custom_mse: Option<(&'a CompiledExpr, &'a [f64])>,
    ) -> Self {
        Self {
            surface,
            bumps,
            custom_mse,
        }
    }

    #[must_use]
    pub fn evaluate_mse_only(
        custom_mse: Option<(&CompiledExpr, &[f64])>,
        w: f64,
        b: f64,
    ) -> Option<f64> {
        let n = LINEAR_MSE_X.len();
        let mut sum_sq = 0.0;
        match custom_mse {
            None => {
                for k in 0..n {
                    let err = (w * LINEAR_MSE_X[k] + b) - LINEAR_MSE_Y[k];
                    sum_sq += err * err;
                }
            }
            Some((compiled, params)) => {
                for k in 0..n {
                    let pred = compiled.eval_3d(LINEAR_MSE_X[k], w, b, params);
                    if !pred.is_finite() {
                        return None;
                    }
                    let err = (pred - LINEAR_MSE_Y[k]).clamp(-1e6, 1e6);
                    sum_sq += err * err;
                }
            }
        }
        let loss = sum_sq / (n as f64);
        loss.is_finite().then_some(loss)
    }

    #[inline]
    #[must_use]
    pub const fn global_minima(self) -> &'static [[f64; 2]] {
        if matches!(self.surface, LandscapeSurface::LinearMse) && self.custom_mse.is_some() {
            &[]
        } else {
            self.surface.global_minima()
        }
    }

    #[inline]
    #[must_use]
    pub fn evaluate(self, w: f64, b: f64) -> f64 {
        let base = if self.surface == LandscapeSurface::LinearMse && self.custom_mse.is_some() {
            Self::evaluate_mse_only(self.custom_mse, w, b).unwrap_or(f64::NAN)
        } else {
            self.surface.base_loss(w, b)
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
        let mut grad = if self.surface == LandscapeSurface::LinearMse && self.custom_mse.is_some() {
            let h = CUSTOM_FD_STEP;
            let dw = match (
                Self::evaluate_mse_only(self.custom_mse, w + h, b),
                Self::evaluate_mse_only(self.custom_mse, w - h, b),
            ) {
                (Some(hi), Some(lo)) => ((hi - lo) / (2.0 * h)).clamp(-1e6, 1e6),
                _ => 0.0,
            };
            let db = match (
                Self::evaluate_mse_only(self.custom_mse, w, b + h),
                Self::evaluate_mse_only(self.custom_mse, w, b - h),
            ) {
                (Some(hi), Some(lo)) => ((hi - lo) / (2.0 * h)).clamp(-1e6, 1e6),
                _ => 0.0,
            };
            Vector2::new(dw, db)
        } else {
            self.surface.base_gradient(w, b)
        };
        for bump in self.bumps {
            grad += bump.gradient(w, b);
        }
        grad
    }
}
