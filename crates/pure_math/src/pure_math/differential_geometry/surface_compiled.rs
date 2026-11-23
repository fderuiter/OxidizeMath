// [cite:modular_polynomials_review]
use engine_symbolic::Jet2;
use math_commons::ast::CompiledExpr;
use nalgebra::{Point3, Vector3};

use super::{MAX_SURFACE_PARAMS, ParametricSurface, sanitize_surface_coord};

#[inline]
#[verified_engine::verified]
fn sanitize_jet2(j: Jet2<2>) -> Jet2<2> {
    if j.is_finite() {
        j
    } else if j.val.is_finite() {
        Jet2::constant(j.val)
    } else {
        Jet2::ZERO
    }
}

#[verified_engine::verified]
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
    #[verified_engine::verified]
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
    #[verified_engine::verified]
    pub const fn compiled(&self) -> &CompiledExpr {
        &self.compiled
    }
}

impl ParametricSurface for CompiledHeightfieldSurface {
    #[verified_engine::verified]
    fn position(&self, u: f64, v: f64) -> Point3<f64> {
        let z = self.compiled.eval_2d(u, v, &self.params[..self.param_len]);
        Point3::new(u, v, sanitize_surface_coord(z))
    }

    #[verified_engine::verified]
    fn position_jet2(&self, u: Jet2<2>, v: Jet2<2>) -> [Jet2<2>; 3] {
        let z = self
            .compiled
            .eval_2d_jet2(u, v, &self.params[..self.param_len]);
        [u, v, sanitize_jet2(z)]
    }

    #[verified_engine::verified]
    fn partial_u(&self, u: f64, v: f64) -> Vector3<f64> {
        let jets = self.position_jet2(Jet2::var(u, 0), Jet2::var(v, 1));
        Vector3::new(jets[0].du(), jets[1].du(), jets[2].du())
    }

    #[verified_engine::verified]
    fn partial_v(&self, u: f64, v: f64) -> Vector3<f64> {
        let jets = self.position_jet2(Jet2::var(u, 0), Jet2::var(v, 1));
        Vector3::new(jets[0].dv(), jets[1].dv(), jets[2].dv())
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
    #[verified_engine::verified]
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
}

impl ParametricSurface for CompiledParametricSurface {
    #[verified_engine::verified]
    fn position(&self, u: f64, v: f64) -> Point3<f64> {
        let p = &self.params[..self.param_len];
        let x = sanitize_surface_coord(self.x_expr.eval_2d(u, v, p));
        let y = sanitize_surface_coord(self.y_expr.eval_2d(u, v, p));
        let z = sanitize_surface_coord(self.z_expr.eval_2d(u, v, p));
        Point3::new(x, y, z)
    }

    #[verified_engine::verified]
    fn position_jet2(&self, u: Jet2<2>, v: Jet2<2>) -> [Jet2<2>; 3] {
        let p = &self.params[..self.param_len];
        [
            sanitize_jet2(self.x_expr.eval_2d_jet2(u, v, p)),
            sanitize_jet2(self.y_expr.eval_2d_jet2(u, v, p)),
            sanitize_jet2(self.z_expr.eval_2d_jet2(u, v, p)),
        ]
    }

    #[verified_engine::verified]
    fn partial_u(&self, u: f64, v: f64) -> Vector3<f64> {
        let jets = self.position_jet2(Jet2::var(u, 0), Jet2::var(v, 1));
        Vector3::new(jets[0].du(), jets[1].du(), jets[2].du())
    }

    #[verified_engine::verified]
    fn partial_v(&self, u: f64, v: f64) -> Vector3<f64> {
        let jets = self.position_jet2(Jet2::var(u, 0), Jet2::var(v, 1));
        Vector3::new(jets[0].dv(), jets[1].dv(), jets[2].dv())
    }
}
