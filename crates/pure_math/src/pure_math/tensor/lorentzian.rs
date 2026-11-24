//! Stack-allocated 4D Lorentzian connections and closed-form Schwarzschild geometry.
//!
//! Provides [`LorentzianConnection4D`] and [`SchwarzschildConnection4D`] backed by fixed-size
//! `nalgebra` [`Vector4`] and [`Matrix4`] types so relativistic geodesic integrators run with
//! zero heap allocations inside [`verified_engine`] runtime verification loops.

use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector, Matrix4, Vector4};

use super::metric::Metric;
use super::static_metric::StaticMetric;
use super::types::TensorError;

/// Stack-based 4D Lorentzian manifold connection with metric signature $(-, +, +, +)$.
///
/// Coordinates follow $x^\mu = (ct, x^1, x^2, x^3)$ and four-velocities $u^\mu = dx^\mu / d\lambda$
/// are parameterized by an affine parameter $\lambda$ ($c\tau$ for timelike worldlines).
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::nalgebra::Vector4;
/// use math_explorer::pure_math::tensor::{LorentzianConnection4D, SchwarzschildConnection4D};
///
/// let conn = SchwarzschildConnection4D::new(2.0).unwrap();
/// let coords = Vector4::new(0.0, 6.0, std::f64::consts::FRAC_PI_2, 0.0);
/// let gammas = conn.christoffel_tensors(&coords).unwrap();
/// // At r = 3 R_s = 6.0: \Gamma^0_{01} = R_s / (2 r (r - R_s)) = 2 / (12 * 4) = 1 / 24
/// assert!((gammas[0][(0, 1)] - (1.0 / 24.0)).abs() < 1e-14);
/// ```
pub trait LorentzianConnection4D {
    /// Evaluates the covariant 4D Lorentzian metric tensor $g_{\mu\nu}(x)$ on the stack.
    ///
    /// # Errors
    /// Returns [`TensorError::SingularMetric`] if `coords` lies at or inside a horizon or
    /// coordinate singularity, or contains non-finite components.
    fn metric_tensor(&self, coords: &Vector4<f64>) -> Result<Matrix4<f64>, TensorError>;

    /// Evaluates the contravariant inverse metric tensor $g^{\mu\nu}(x)$ on the stack.
    ///
    /// # Errors
    /// Returns [`TensorError::SingularMetric`] if $g_{\mu\nu}(x)$ is singular or non-finite.
    #[verified_engine::verified]
    fn inverse_metric_tensor(&self, coords: &Vector4<f64>) -> Result<Matrix4<f64>, TensorError> {
        let g = self.metric_tensor(coords)?;
        if !g.iter().all(|v| v.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        g.try_inverse().ok_or(TensorError::SingularMetric)
    }

    /// Evaluates Christoffel symbols of the second kind $\Gamma^\mu_{\alpha\beta}(x)$ as four
    /// stack-allocated $4 \times 4$ symmetric matrices (`result[mu][(alpha, beta)]`).
    ///
    /// # Errors
    /// Returns [`TensorError::SingularMetric`] if the connection is singular at `coords`.
    fn christoffel_tensors(&self, coords: &Vector4<f64>) -> Result<[Matrix4<f64>; 4], TensorError>;

    /// Computes geodesic four-acceleration $a^\mu = -\Gamma^\mu_{\alpha\beta}(x) u^\alpha u^\beta$
    /// on the stack without heap allocation.
    ///
    /// # Errors
    /// Returns [`TensorError::SingularMetric`] if `coords` or `four_velocity` is non-finite or
    /// lies on a metric singularity.
    #[verified_engine::verified]
    fn geodesic_acceleration(
        &self,
        coords: &Vector4<f64>,
        four_velocity: &Vector4<f64>,
    ) -> Result<Vector4<f64>, TensorError> {
        if !four_velocity.iter().all(|v| v.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        let gamma = self.christoffel_tensors(coords)?;
        let mut acc = Vector4::zeros();
        for mu in 0..4 {
            let quad = four_velocity.dot(&(gamma[mu] * four_velocity));
            if !quad.is_finite() {
                return Err(TensorError::SingularMetric);
            }
            acc[mu] = -quad;
        }
        Ok(acc)
    }

    /// Computes quadratic Lorentzian tangent norm $g_{\mu\nu}(x) u^\mu u^\nu$ (`-1` timelike, `0` null).
    ///
    /// # Errors
    /// Returns [`TensorError::SingularMetric`] if the metric cannot be evaluated at `coords`.
    #[verified_engine::verified]
    fn interval_norm(
        &self,
        coords: &Vector4<f64>,
        four_velocity: &Vector4<f64>,
    ) -> Result<f64, TensorError> {
        if !four_velocity.iter().all(|v| v.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        let g = self.metric_tensor(coords)?;
        let norm = four_velocity.dot(&(g * four_velocity));
        if norm.is_finite() {
            Ok(norm)
        } else {
            Err(TensorError::SingularMetric)
        }
    }

    /// Returns `true` when `coords` lies safely outside horizon and polar coordinate guards.
    fn is_outside_horizon(&self, coords: &Vector4<f64>) -> bool;
}

/// Closed-form 4D Schwarzschild spacetime connection in Boyer-Lindquist / Droste coordinates
/// $x^\mu = (ct, r, \theta, \varphi)$ with signature $(-, +, +, +)$.
///
/// Unlike finite-difference [`super::christoffel::christoffel_symbols`], which allocates 21 heap
/// buffers per 4D evaluation and breaks down within stencil width $h = 10^{-5}$ of $r = R_s$,
/// [`SchwarzschildConnection4D`] evaluates exact analytical connection coefficients directly into
/// stack [`Matrix4`] registers for every exterior radius $r > R_s$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchwarzschildConnection4D {
    /// Schwarzschild event-horizon radius $R_s = \frac{2GM}{c^2} > 0$.
    pub schwarzschild_radius: f64,
}

impl SchwarzschildConnection4D {
    /// Creates a closed-form Schwarzschild 4D connection for horizon scale `schwarzschild_radius`.
    ///
    /// # Errors
    /// Returns [`TensorError::SingularMetric`] if `schwarzschild_radius <= 0.0` or is non-finite.
    #[verified_engine::verified]
    pub fn new(schwarzschild_radius: f64) -> Result<Self, TensorError> {
        if !schwarzschild_radius.is_finite() || schwarzschild_radius <= 0.0 {
            return Err(TensorError::SingularMetric);
        }
        Ok(Self {
            schwarzschild_radius,
        })
    }

    #[verified_engine::verified]
    fn validate_exterior(
        &self,
        coords: &Vector4<f64>,
    ) -> Result<(f64, f64, f64, f64), TensorError> {
        let rs = self.schwarzschild_radius;
        if !rs.is_finite() || rs <= 0.0 || !coords.iter().all(|v| v.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        let r = coords[1];
        let (sin_t, cos_t) = coords[2].sin_cos();
        if r <= rs || sin_t.abs() <= 1e-12 {
            return Err(TensorError::SingularMetric);
        }
        Ok((rs, r, sin_t, cos_t))
    }
}

impl LorentzianConnection4D for SchwarzschildConnection4D {
    #[verified_engine::verified]
    fn metric_tensor(&self, coords: &Vector4<f64>) -> Result<Matrix4<f64>, TensorError> {
        let rs = self.schwarzschild_radius;
        if !rs.is_finite() || rs <= 0.0 || !coords.iter().all(|v| v.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        let r = coords[1];
        if r <= rs {
            return Err(TensorError::SingularMetric);
        }
        let lapse_sq = 1.0 - rs / r;
        let sin_t = coords[2].sin();
        let r2 = r * r;
        Ok(Matrix4::from_diagonal(&Vector4::new(
            -lapse_sq,
            1.0 / lapse_sq,
            r2,
            r2 * sin_t * sin_t,
        )))
    }

    #[verified_engine::verified]
    fn inverse_metric_tensor(&self, coords: &Vector4<f64>) -> Result<Matrix4<f64>, TensorError> {
        let (rs, r, sin_t, _) = self.validate_exterior(coords)?;
        let lapse_sq = 1.0 - rs / r;
        let inv_r2 = 1.0 / (r * r);
        Ok(Matrix4::from_diagonal(&Vector4::new(
            -1.0 / lapse_sq,
            lapse_sq,
            inv_r2,
            inv_r2 / (sin_t * sin_t),
        )))
    }

    #[verified_engine::verified]
    fn christoffel_tensors(&self, coords: &Vector4<f64>) -> Result<[Matrix4<f64>; 4], TensorError> {
        let (rs, r, sin_t, cos_t) = self.validate_exterior(coords)?;
        let delta = r - rs;
        let g_t_tr = rs / (2.0 * r * delta);
        let inv_r = 1.0 / r;
        let cot_t = cos_t / sin_t;

        let mut gamma = [Matrix4::zeros(); 4];
        gamma[0][(0, 1)] = g_t_tr;
        gamma[0][(1, 0)] = g_t_tr;

        gamma[1][(0, 0)] = (rs * delta) / (2.0 * r * r * r);
        gamma[1][(1, 1)] = -g_t_tr;
        gamma[1][(2, 2)] = -delta;
        gamma[1][(3, 3)] = -delta * sin_t * sin_t;

        gamma[2][(1, 2)] = inv_r;
        gamma[2][(2, 1)] = inv_r;
        gamma[2][(3, 3)] = -sin_t * cos_t;

        gamma[3][(1, 3)] = inv_r;
        gamma[3][(3, 1)] = inv_r;
        gamma[3][(2, 3)] = cot_t;
        gamma[3][(3, 2)] = cot_t;
        Ok(gamma)
    }

    #[verified_engine::verified]
    fn geodesic_acceleration(
        &self,
        coords: &Vector4<f64>,
        u: &Vector4<f64>,
    ) -> Result<Vector4<f64>, TensorError> {
        if !u.iter().all(|v| v.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        let (rs, r, sin_t, cos_t) = self.validate_exterior(coords)?;
        let delta = r - rs;
        let g_t_tr = rs / (2.0 * r * delta);
        let inv_r = 1.0 / r;
        let cot_t = cos_t / sin_t;

        let a_ct = -2.0 * g_t_tr * u[0] * u[1];
        let a_r = -((rs * delta) / (2.0 * r * r * r)) * u[0] * u[0]
            + g_t_tr * u[1] * u[1]
            + delta * (u[2] * u[2] + sin_t * sin_t * u[3] * u[3]);
        let a_theta = -2.0 * inv_r * u[1] * u[2] + sin_t * cos_t * u[3] * u[3];
        let a_phi = -2.0 * (inv_r * u[1] + cot_t * u[2]) * u[3];
        let acc = Vector4::new(a_ct, a_r, a_theta, a_phi);
        if acc.iter().all(|v| v.is_finite()) {
            Ok(acc)
        } else {
            Err(TensorError::SingularMetric)
        }
    }

    #[verified_engine::verified]
    fn is_outside_horizon(&self, coords: &Vector4<f64>) -> bool {
        let rs = self.schwarzschild_radius;
        let r = coords[1];
        let theta = coords[2];
        rs.is_finite()
            && rs > 0.0
            && coords.iter().all(|v| v.is_finite())
            && r > 1.02 * rs
            && theta > 1e-4
            && theta < PI - 1e-4
            && theta.sin() > 1e-4
    }
}

impl StaticMetric<4> for SchwarzschildConnection4D {
    #[verified_engine::verified]
    fn metric_at(&self, point: &Vector4<f64>) -> Result<Matrix4<f64>, TensorError> {
        self.metric_tensor(point)
    }

    #[verified_engine::verified]
    fn inverse_metric_at(&self, point: &Vector4<f64>) -> Result<Matrix4<f64>, TensorError> {
        self.inverse_metric_tensor(point)
    }

    #[verified_engine::verified]
    fn christoffel_at(&self, point: &Vector4<f64>) -> Result<[Matrix4<f64>; 4], TensorError> {
        self.christoffel_tensors(point)
    }

    #[verified_engine::verified]
    fn is_regular_at(&self, point: &Vector4<f64>) -> bool {
        self.is_outside_horizon(point)
    }
}

impl Metric for SchwarzschildConnection4D {
    #[verified_engine::verified]
    fn metric_at(&self, point: &DVector<f64>) -> Result<DMatrix<f64>, TensorError> {
        if point.len() != 4 {
            return Err(TensorError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(4),
                    actual: math_commons::math_kernel::types::Dimension(point.len()),
                },
            ));
        }
        let coords = Vector4::new(point[0], point[1], point[2], point[3]);
        let g = self.metric_tensor(&coords)?;
        Ok(DMatrix::from_row_slice(4, 4, g.as_slice()))
    }
}

// [cite:general_relativity_geodesics]
