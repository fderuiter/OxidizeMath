// [cite:quasiperfect_alcf]
use engine_symbolic::{Dual, Jet2};
use math_commons::ast::CompiledExpr;
use nalgebra::Vector3;

use super::coordinates::OrthogonalCoordinateSystem;

/// Differentiable 3D scalar field $\Phi(u_1, u_2, u_3)$ supporting forward-mode [`Dual<f64>`]
/// and second-order [`Jet2<3>`] automatic differentiation.
pub trait ScalarField3D {
    /// Evaluates $\Phi$ over [`Dual<f64>`] coordinates.
    fn eval_dual(&self, point: &[Dual<f64>; 3]) -> Dual<f64>;

    /// Evaluates $\Phi$ over second-order [`Jet2<3>`] coordinates.
    fn eval_jet2(&self, point: &[Jet2<3>; 3]) -> Jet2<3>;
}

/// Differentiable 3D vector field $\mathbf{A}(u_1, u_2, u_3)$ supporting [`Dual<f64>`] tangents.
pub trait VectorField3D {
    /// Evaluates curvilinear basis components $[A_1, A_2, A_3]$ over [`Dual<f64>`] coordinates.
    fn eval_dual(&self, point: &[Dual<f64>; 3]) -> [Dual<f64>; 3];
}

/// Wrapper lifting a `Dual<f64>` 3D scalar closure into [`ScalarField3D`].
#[derive(Debug, Clone, Copy)]
pub struct DualScalarField3D<G>(pub G);

impl<G> ScalarField3D for DualScalarField3D<G>
where
    G: Fn(&[Dual<f64>; 3]) -> Dual<f64>,
{
    #[inline]
    fn eval_dual(&self, point: &[Dual<f64>; 3]) -> Dual<f64> {
        (self.0)(point)
    }

    fn eval_jet2(&self, point: &[Jet2<3>; 3]) -> Jet2<3> {
        let p0 = [point[0].val, point[1].val, point[2].val];
        let mut grad = [0.0_f64; 3];
        let mut val = 0.0;
        for axis in 0..3 {
            let mut dpt = [
                Dual::constant(p0[0]),
                Dual::constant(p0[1]),
                Dual::constant(p0[2]),
            ];
            dpt[axis] = Dual::variable(p0[axis]);
            let d = (self.0)(&dpt);
            val = d.re;
            grad[axis] = d.eps;
        }
        let mut hess = [[0.0_f64; 3]; 3];
        let h = 1e-6;
        for axis in 0..3 {
            let mut dp = [
                Dual::constant(p0[0]),
                Dual::constant(p0[1]),
                Dual::constant(p0[2]),
            ];
            let mut dm = dp;
            dp[axis] = Dual::variable(p0[axis] + h);
            dm[axis] = Dual::variable(p0[axis] - h);
            hess[axis][axis] = ((self.0)(&dp).eps - (self.0)(&dm).eps) / (2.0 * h);
        }
        Jet2 { val, grad, hess }
    }
}

/// Wrapper lifting a `Jet2<3>` 3D scalar closure into [`ScalarField3D`] for 1-pass Laplacians.
#[derive(Debug, Clone, Copy)]
pub struct Jet2ScalarField3D<G>(pub G);

impl<G> ScalarField3D for Jet2ScalarField3D<G>
where
    G: Fn(&[Jet2<3>; 3]) -> Jet2<3>,
{
    fn eval_dual(&self, point: &[Dual<f64>; 3]) -> Dual<f64> {
        let jets = [
            Jet2 {
                val: point[0].re,
                grad: [point[0].eps, 0.0, 0.0],
                hess: [[0.0; 3]; 3],
            },
            Jet2 {
                val: point[1].re,
                grad: [point[1].eps, 0.0, 0.0],
                hess: [[0.0; 3]; 3],
            },
            Jet2 {
                val: point[2].re,
                grad: [point[2].eps, 0.0, 0.0],
                hess: [[0.0; 3]; 3],
            },
        ];
        let out = (self.0)(&jets);
        Dual::new(out.val, out.grad[0])
    }

    #[inline]
    fn eval_jet2(&self, point: &[Jet2<3>; 3]) -> Jet2<3> {
        (self.0)(point)
    }
}

/// Wrapper lifting a `Dual<f64>` 3D vector closure into [`VectorField3D`].
#[derive(Debug, Clone, Copy)]
pub struct DualVectorField3D<G>(pub G);

impl<G> VectorField3D for DualVectorField3D<G>
where
    G: Fn(&[Dual<f64>; 3]) -> [Dual<f64>; 3],
{
    #[inline]
    fn eval_dual(&self, point: &[Dual<f64>; 3]) -> [Dual<f64>; 3] {
        (self.0)(point)
    }
}

impl ScalarField3D for CompiledExpr {
    #[inline]
    #[verified_engine::verified]
    fn eval_dual(&self, point: &[Dual<f64>; 3]) -> Dual<f64> {
        self.eval_slice_dual(point)
    }

    #[inline]
    #[verified_engine::verified]
    fn eval_jet2(&self, point: &[Jet2<3>; 3]) -> Jet2<3> {
        self.eval_3d_jet2(point[0], point[1], point[2], &[])
    }
}

impl ScalarField3D for &CompiledExpr {
    #[inline]
    #[verified_engine::verified]
    fn eval_dual(&self, point: &[Dual<f64>; 3]) -> Dual<f64> {
        self.eval_slice_dual(point)
    }

    #[inline]
    #[verified_engine::verified]
    fn eval_jet2(&self, point: &[Jet2<3>; 3]) -> Jet2<3> {
        self.eval_3d_jet2(point[0], point[1], point[2], &[])
    }
}

impl VectorField3D for [CompiledExpr; 3] {
    #[inline]
    #[verified_engine::verified]
    fn eval_dual(&self, point: &[Dual<f64>; 3]) -> [Dual<f64>; 3] {
        [
            self[0].eval_slice_dual(point),
            self[1].eval_slice_dual(point),
            self[2].eval_slice_dual(point),
        ]
    }
}

impl VectorField3D for &[CompiledExpr; 3] {
    #[inline]
    #[verified_engine::verified]
    fn eval_dual(&self, point: &[Dual<f64>; 3]) -> [Dual<f64>; 3] {
        [
            self[0].eval_slice_dual(point),
            self[1].eval_slice_dual(point),
            self[2].eval_slice_dual(point),
        ]
    }
}

impl<F> ScalarField3D for F
where
    F: Fn(&Vector3<f64>) -> f64,
{
    #[verified_engine::verified]
    fn eval_dual(&self, point: &[Dual<f64>; 3]) -> Dual<f64> {
        let p0 = Vector3::new(point[0].re, point[1].re, point[2].re);
        let f0 = self(&p0);
        let h = 1e-6;
        let mut eps = 0.0;
        for axis in 0..3 {
            if point[axis].eps != 0.0 {
                let mut pp = p0;
                let mut pm = p0;
                pp[axis] += h;
                pm[axis] -= h;
                eps += ((self(&pp) - self(&pm)) / (2.0 * h)) * point[axis].eps;
            }
        }
        Dual::new(f0, eps)
    }

    #[verified_engine::verified]
    fn eval_jet2(&self, point: &[Jet2<3>; 3]) -> Jet2<3> {
        let p0 = Vector3::new(point[0].val, point[1].val, point[2].val);
        let f0 = self(&p0);
        let h = 1e-5;
        let mut grad = [0.0_f64; 3];
        let mut hess = [[0.0_f64; 3]; 3];
        for axis in 0..3 {
            let mut pp = p0;
            let mut pm = p0;
            pp[axis] += h;
            pm[axis] -= h;
            let fp = self(&pp);
            let fm = self(&pm);
            grad[axis] = (fp - fm) / (2.0 * h);
            hess[axis][axis] = (fp - 2.0 * f0 + fm) / (h * h);
        }
        Jet2 {
            val: f0,
            grad,
            hess,
        }
    }
}

impl<F> VectorField3D for F
where
    F: Fn(&Vector3<f64>) -> Vector3<f64>,
{
    #[verified_engine::verified]
    fn eval_dual(&self, point: &[Dual<f64>; 3]) -> [Dual<f64>; 3] {
        let p0 = Vector3::new(point[0].re, point[1].re, point[2].re);
        let v0 = self(&p0);
        let h = 1e-6;
        let mut out = [
            Dual::constant(v0[0]),
            Dual::constant(v0[1]),
            Dual::constant(v0[2]),
        ];
        for axis in 0..3 {
            if point[axis].eps != 0.0 {
                let mut pp = p0;
                let mut pm = p0;
                pp[axis] += h;
                pm[axis] -= h;
                let dv = (self(&pp) - self(&pm)) / (2.0 * h);
                out[0].eps += dv[0] * point[axis].eps;
                out[1].eps += dv[1] * point[axis].eps;
                out[2].eps += dv[2] * point[axis].eps;
            }
        }
        out
    }
}

#[verified_engine::verified]
fn seed_axis_dual(point: &Vector3<f64>, axis: usize) -> [Dual<f64>; 3] {
    let mut pt = [
        Dual::constant(point[0]),
        Dual::constant(point[1]),
        Dual::constant(point[2]),
    ];
    pt[axis] = Dual::variable(point[axis]);
    pt
}

/// Computes the gradient of a scalar field $\nabla \Phi$ in local curvilinear basis vectors.
#[verified_engine::verified]
pub fn gradient<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> Vector3<f64>
where
    S: OrthogonalCoordinateSystem,
    F: ScalarField3D,
{
    debug_assert!(point.iter().all(|x| x.is_finite()));
    let factors = coords.scale_factors(point);
    let mut grad = Vector3::zeros();
    for i in 0..3 {
        let dphi = field.eval_dual(&seed_axis_dual(point, i)).eps;
        grad[i] = dphi / factors[i];
    }
    grad
}

/// Computes $\nabla \Phi$ directly from a forward-mode [`Dual<f64>`] closure.
#[verified_engine::verified]
pub fn gradient_dual<S, G>(coords: &S, field: G, point: &Vector3<f64>) -> Vector3<f64>
where
    S: OrthogonalCoordinateSystem,
    G: Fn(&[Dual<f64>; 3]) -> Dual<f64>,
{
    gradient(coords, DualScalarField3D(field), point)
}

/// Computes the divergence of a vector field $\nabla \cdot \mathbf{A}$ via forward-mode `Dual` product differentiation.
#[verified_engine::verified]
pub fn divergence<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> f64
where
    S: OrthogonalCoordinateSystem,
    F: VectorField3D,
{
    debug_assert!(point.iter().all(|x| x.is_finite()));
    let factors = coords.scale_factors(point);
    let h1h2h3 = factors[0] * factors[1] * factors[2];
    if h1h2h3.abs() < 1e-12 {
        return 0.0;
    }

    let mut sum = 0.0;
    for i in 0..3 {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let dpt = seed_axis_dual(point, i);
        let h_dual = coords.scale_factors_dual(&dpt);
        let a_dual = field.eval_dual(&dpt);
        sum += (h_dual[j] * h_dual[k] * a_dual[i]).eps;
    }
    sum / h1h2h3
}

/// Computes $\nabla \cdot \mathbf{A}$ directly from a forward-mode [`Dual<f64>`] vector closure.
#[verified_engine::verified]
pub fn divergence_dual<S, G>(coords: &S, field: G, point: &Vector3<f64>) -> f64
where
    S: OrthogonalCoordinateSystem,
    G: Fn(&[Dual<f64>; 3]) -> [Dual<f64>; 3],
{
    divergence(coords, DualVectorField3D(field), point)
}

/// Computes the curl of a vector field $\nabla \times \mathbf{A}$ in local curvilinear basis vectors.
#[verified_engine::verified]
pub fn curl<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> Vector3<f64>
where
    S: OrthogonalCoordinateSystem,
    F: VectorField3D,
{
    debug_assert!(point.iter().all(|x| x.is_finite()));
    let h = coords.scale_factors(point);
    let mut d_ha = [[0.0_f64; 3]; 3];
    let mut axis = 0;
    while axis < 3 {
        let dpt = seed_axis_dual(point, axis);
        let hd = coords.scale_factors_dual(&dpt);
        let ad = field.eval_dual(&dpt);
        d_ha[0][axis] = (hd[0] * ad[0]).eps;
        d_ha[1][axis] = (hd[1] * ad[1]).eps;
        d_ha[2][axis] = (hd[2] * ad[2]).eps;
        axis += 1;
    }

    let mut result = Vector3::zeros();
    for i in 0..3 {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let denom = h[j] * h[k];
        result[i] = if denom.abs() < 1e-12 {
            0.0
        } else {
            (d_ha[k][j] - d_ha[j][k]) / denom
        };
    }
    result
}

/// Computes $\nabla \times \mathbf{A}$ directly from a forward-mode [`Dual<f64>`] vector closure.
#[verified_engine::verified]
pub fn curl_dual<S, G>(coords: &S, field: G, point: &Vector3<f64>) -> Vector3<f64>
where
    S: OrthogonalCoordinateSystem,
    G: Fn(&[Dual<f64>; 3]) -> [Dual<f64>; 3],
{
    curl(coords, DualVectorField3D(field), point)
}

/// Computes the curvilinear Laplacian $\nabla^2 \Phi$ in a single `Jet2<3>` pass.
#[verified_engine::verified]
pub fn laplacian<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> f64
where
    S: OrthogonalCoordinateSystem,
    F: ScalarField3D,
{
    debug_assert!(point.iter().all(|x| x.is_finite()));
    let jpt = [
        Jet2::var(point[0], 0),
        Jet2::var(point[1], 1),
        Jet2::var(point[2], 2),
    ];
    let hj = coords.scale_factors_jet2(&jpt);
    let h1h2h3 = hj[0].val * hj[1].val * hj[2].val;
    if h1h2h3.abs() < 1e-12 {
        return 0.0;
    }

    let phi = field.eval_jet2(&jpt);
    let mut sum = 0.0;
    for i in 0..3 {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let q = (hj[j] * hj[k]) / hj[i];
        sum += q.grad[i] * phi.grad[i] + q.val * phi.hess[i][i];
    }
    sum / h1h2h3
}

/// Computes $\nabla^2 \Phi$ directly from a `Jet2<3>` closure in **1** pass without stencils.
#[verified_engine::verified]
pub fn laplacian_jet2<S, G>(coords: &S, field: G, point: &Vector3<f64>) -> f64
where
    S: OrthogonalCoordinateSystem,
    G: FnOnce(&[Jet2<3>; 3]) -> Jet2<3>,
{
    let jpt = [
        Jet2::var(point[0], 0),
        Jet2::var(point[1], 1),
        Jet2::var(point[2], 2),
    ];
    let hj = coords.scale_factors_jet2(&jpt);
    let h1h2h3 = hj[0].val * hj[1].val * hj[2].val;
    if h1h2h3.abs() < 1e-12 {
        return 0.0;
    }
    let phi = field(&jpt);
    let mut sum = 0.0;
    for i in 0..3 {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let q = (hj[j] * hj[k]) / hj[i];
        sum += q.grad[i] * phi.grad[i] + q.val * phi.hess[i][i];
    }
    sum / h1h2h3
}
