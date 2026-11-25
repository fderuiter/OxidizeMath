// [cite:modular_polynomials_review]
use super::{
    DifferentiableSurface, MAX_SURFACE_PARAMS, ParametricSurface, eval_partial_u_ad,
    eval_partial_v_ad, eval_second_partials_ad,
};
use math_commons::ast::CompiledExpr;
use math_scalars::Scalar;
use nalgebra::{Point3, Vector3};

#[inline]
fn sanitize_surface_coord(val: f64) -> f64 {
    if val.is_finite() {
        val.clamp(-50.0, 50.0)
    } else {
        0.0
    }
}

#[inline]
fn sanitize_surface_scalar<S: Scalar>(primal: f64, val: S) -> S {
    if !primal.is_finite() {
        S::ZERO
    } else if primal < -50.0 {
        S::from_f64(-50.0)
    } else if primal > 50.0 {
        S::from_f64(50.0)
    } else {
        val
    }
}

fn pack_surface_params(params: &[f64]) -> ([f64; MAX_SURFACE_PARAMS], usize) {
    let mut inline = [0.0; MAX_SURFACE_PARAMS];
    let len = params.len().min(MAX_SURFACE_PARAMS);
    inline[..len].copy_from_slice(&params[..len]);
    (inline, len)
}

/// Heightfield surface $r(x, y) = (x, y, f(x, y))$ backed by zero-allocation [`CompiledExpr`] bytecode.
#[derive(Debug, Clone)]
pub struct CompiledHeightfieldSurface {
    compiled: CompiledExpr,
    params: [f64; MAX_SURFACE_PARAMS],
    param_len: usize,
}

impl CompiledHeightfieldSurface {
    /// Constructs a heightfield surface from a [`CompiledExpr`] compiled over axes `["x", "y"]`.
    #[must_use]
    pub fn new(compiled: CompiledExpr, params: &[f64]) -> Self {
        let (packed, param_len) = pack_surface_params(params);
        Self {
            compiled,
            params: packed,
            param_len,
        }
    }

    /// Returns a reference to the underlying compiled height expression.
    #[must_use]
    pub const fn compiled(&self) -> &CompiledExpr {
        &self.compiled
    }

    #[inline]
    fn eval_height_primal(&self, u: f64, v: f64) -> f64 {
        self.compiled.eval_2d(u, v, &self.params[..self.param_len])
    }
}

impl DifferentiableSurface for CompiledHeightfieldSurface {
    #[inline]
    fn position_scalar<S: Scalar>(&self, u: S, v: S) -> [S; 3] {
        let p = &self.params[..self.param_len];
        let z = self.compiled.eval_2d_scalar(u, v, p).unwrap_or(S::ZERO);
        [u, v, z]
    }
}

impl ParametricSurface for CompiledHeightfieldSurface {
    #[verified_engine::verified]
    fn position(&self, u: f64, v: f64) -> Point3<f64> {
        let z = self.eval_height_primal(u, v);
        Point3::new(u, v, sanitize_surface_coord(z))
    }

    #[verified_engine::verified]
    fn partial_u(&self, u: f64, v: f64) -> Vector3<f64> {
        let z0 = self.eval_height_primal(u, v);
        if !z0.is_finite() || !(-50.0..=50.0).contains(&z0) {
            return Vector3::new(1.0, 0.0, 0.0);
        }
        let ru = eval_partial_u_ad(self, u, v);
        if ru.iter().all(|c| c.is_finite()) {
            ru
        } else {
            Vector3::new(1.0, 0.0, 0.0)
        }
    }

    #[verified_engine::verified]
    fn partial_v(&self, u: f64, v: f64) -> Vector3<f64> {
        let z0 = self.eval_height_primal(u, v);
        if !z0.is_finite() || !(-50.0..=50.0).contains(&z0) {
            return Vector3::new(0.0, 1.0, 0.0);
        }
        let rv = eval_partial_v_ad(self, u, v);
        if rv.iter().all(|c| c.is_finite()) {
            rv
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        }
    }

    #[verified_engine::verified]
    fn second_partials(&self, u: f64, v: f64) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        let z0 = self.eval_height_primal(u, v);
        if !z0.is_finite() || !(-50.0..=50.0).contains(&z0) {
            return (Vector3::zeros(), Vector3::zeros(), Vector3::zeros());
        }
        let (ruu, ruv, rvv) = eval_second_partials_ad(self, u, v);
        let clean = |vec: Vector3<f64>| {
            if vec.iter().all(|c| c.is_finite()) {
                vec
            } else {
                Vector3::zeros()
            }
        };
        (clean(ruu), clean(ruv), clean(rvv))
    }
}

/// Parametric 3D surface $r(u, v) = (x(u, v), y(u, v), z(u, v))$ backed by [`CompiledExpr`] triples.
#[derive(Debug, Clone)]
pub struct CompiledParametricSurface {
    x_expr: CompiledExpr,
    y_expr: CompiledExpr,
    z_expr: CompiledExpr,
    params: [f64; MAX_SURFACE_PARAMS],
    param_len: usize,
}

impl CompiledParametricSurface {
    /// Constructs a parametric surface from three [`CompiledExpr`] formulas compiled over `["u", "v"]`.
    #[must_use]
    pub fn new(
        x_expr: CompiledExpr,
        y_expr: CompiledExpr,
        z_expr: CompiledExpr,
        params: &[f64],
    ) -> Self {
        let (packed, param_len) = pack_surface_params(params);
        Self {
            x_expr,
            y_expr,
            z_expr,
            params: packed,
            param_len,
        }
    }

    #[inline]
    fn eval_primals(&self, u: f64, v: f64) -> [f64; 3] {
        let p = &self.params[..self.param_len];
        [
            self.x_expr.eval_2d(u, v, p),
            self.y_expr.eval_2d(u, v, p),
            self.z_expr.eval_2d(u, v, p),
        ]
    }
}

impl DifferentiableSurface for CompiledParametricSurface {
    #[inline]
    fn position_scalar<S: Scalar>(&self, u: S, v: S) -> [S; 3] {
        let p = &self.params[..self.param_len];
        [
            self.x_expr.eval_2d_scalar(u, v, p).unwrap_or(S::ZERO),
            self.y_expr.eval_2d_scalar(u, v, p).unwrap_or(S::ZERO),
            self.z_expr.eval_2d_scalar(u, v, p).unwrap_or(S::ZERO),
        ]
    }
}

impl ParametricSurface for CompiledParametricSurface {
    #[verified_engine::verified]
    fn position(&self, u: f64, v: f64) -> Point3<f64> {
        let [x, y, z] = self.eval_primals(u, v);
        Point3::new(
            sanitize_surface_coord(x),
            sanitize_surface_coord(y),
            sanitize_surface_coord(z),
        )
    }

    #[verified_engine::verified]
    fn partial_u(&self, u: f64, v: f64) -> Vector3<f64> {
        let primals = self.eval_primals(u, v);
        let raw = eval_partial_u_ad(self, u, v);
        Vector3::new(
            if (-50.0..=50.0).contains(&primals[0]) && raw[0].is_finite() {
                sanitize_surface_scalar(primals[0], raw[0])
            } else {
                0.0
            },
            if (-50.0..=50.0).contains(&primals[1]) && raw[1].is_finite() {
                sanitize_surface_scalar(primals[1], raw[1])
            } else {
                0.0
            },
            if (-50.0..=50.0).contains(&primals[2]) && raw[2].is_finite() {
                sanitize_surface_scalar(primals[2], raw[2])
            } else {
                0.0
            },
        )
    }

    #[verified_engine::verified]
    fn partial_v(&self, u: f64, v: f64) -> Vector3<f64> {
        let primals = self.eval_primals(u, v);
        let raw = eval_partial_v_ad(self, u, v);
        Vector3::new(
            if (-50.0..=50.0).contains(&primals[0]) && raw[0].is_finite() {
                raw[0]
            } else {
                0.0
            },
            if (-50.0..=50.0).contains(&primals[1]) && raw[1].is_finite() {
                raw[1]
            } else {
                0.0
            },
            if (-50.0..=50.0).contains(&primals[2]) && raw[2].is_finite() {
                raw[2]
            } else {
                0.0
            },
        )
    }

    #[verified_engine::verified]
    fn second_partials(&self, u: f64, v: f64) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        let primals = self.eval_primals(u, v);
        let (ruu, ruv, rvv) = eval_second_partials_ad(self, u, v);
        let mask = |vec: Vector3<f64>| {
            Vector3::new(
                if (-50.0..=50.0).contains(&primals[0]) && vec[0].is_finite() {
                    vec[0]
                } else {
                    0.0
                },
                if (-50.0..=50.0).contains(&primals[1]) && vec[1].is_finite() {
                    vec[1]
                } else {
                    0.0
                },
                if (-50.0..=50.0).contains(&primals[2]) && vec[2].is_finite() {
                    vec[2]
                } else {
                    0.0
                },
            )
        };
        (mask(ruu), mask(ruv), mask(rvv))
    }
}
