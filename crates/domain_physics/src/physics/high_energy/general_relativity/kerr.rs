//! Equatorial Kerr spacetime metric (`StaticMetric<4>`) with Lense-Thirring frame dragging.

use std::f64::consts::PI;

use math_commons::constants::{C, G};
use nalgebra::{SMatrix, SVector};
use pure_math::pure_math::tensor::{StaticMetric, TensorError};

use crate::error::HighEnergyError;

/// Rotating Kerr black hole restricted to the equatorial slice ($\theta = \pi/2$) in
/// Boyer-Lindquist coordinates $(ct, r, \theta, \varphi)$.
///
/// Line element near $\theta = \pi/2$ (with $\Delta = r^2 - R_s r + a^2$):
/// $$ds^2 = -\left(1 - \frac{R_s}{r}\right) d(ct)^2
///        - \frac{2 a R_s}{r} d(ct)\,d\varphi
///        + \frac{r^2}{\Delta} dr^2 + r^2 d\theta^2
///        + \left(r^2 + a^2 + \frac{a^2 R_s}{r}\right) d\varphi^2$$
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct KerrEquatorialMetric {
    /// Schwarzschild mass scale $R_s = \frac{2GM}{c^2} > 0$.
    pub schwarzschild_radius: f64,
    /// Kerr spin length parameter $a = \frac{J}{Mc}$ satisfying $|a| < \frac{1}{2} R_s$.
    pub spin_a: f64,
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for KerrEquatorialMetric {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Wire {
            schwarzschild_radius: f64,
            spin_a: f64,
        }
        let wire = Wire::deserialize(deserializer)?;
        Self::new(wire.schwarzschild_radius, wire.spin_a).map_err(serde::de::Error::custom)
    }
}

impl KerrEquatorialMetric {
    /// Constructs a sub-extremal [`KerrEquatorialMetric`] from $R_s > 0$ and $|a| < 0.5 R_s$.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidRadius`] if $R_s \le 0$ or $|a| \ge 0.5 R_s$.
    #[verified_engine::verified]
    pub fn new(schwarzschild_radius: f64, spin_a: f64) -> Result<Self, HighEnergyError> {
        if !schwarzschild_radius.is_finite() || schwarzschild_radius <= 0.0 {
            return Err(HighEnergyError::InvalidRadius {
                radius: schwarzschild_radius,
                limit: 0.0,
            });
        }
        let max_spin = 0.5 * schwarzschild_radius;
        if !spin_a.is_finite() || spin_a.abs() >= max_spin {
            return Err(HighEnergyError::InvalidRadius {
                radius: spin_a.abs(),
                limit: max_spin,
            });
        }
        Ok(Self {
            schwarzschild_radius,
            spin_a,
        })
    }

    /// Constructs a [`KerrEquatorialMetric`] from physical mass $M$ (kg) and dimensionless spin
    /// $\chi = a / M_{\text{geom}} \in (-1, 1)$.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidMass`] if $M \le 0$ or [`HighEnergyError::InvalidRadius`]
    /// if $|\chi| \ge 1$.
    #[verified_engine::verified]
    pub fn from_mass_and_chi(mass: f64, chi: f64) -> Result<Self, HighEnergyError> {
        if !mass.is_finite() || mass <= 0.0 {
            return Err(HighEnergyError::InvalidMass { mass });
        }
        let rs = (2.0 * G * mass) / (C * C);
        Self::new(rs, 0.5 * rs * chi)
    }

    /// Outer event horizon radius $r_+ = \frac{1}{2}\left(R_s + \sqrt{R_s^2 - 4a^2}\right)$.
    #[verified_engine::verified]
    pub fn outer_horizon_radius(&self) -> f64 {
        let disc = (self.schwarzschild_radius * self.schwarzschild_radius
            - 4.0 * self.spin_a * self.spin_a)
            .max(0.0);
        0.5 * (self.schwarzschild_radius + disc.sqrt())
    }

    /// Equatorial static-limit (ergosphere) radius $r_{\text{ergo}}(\pi/2) = R_s$.
    #[verified_engine::verified]
    pub fn ergosphere_radius(&self) -> f64 {
        self.schwarzschild_radius
    }

    /// Zero-angular-momentum observer (ZAMO) frame-dragging angular rate
    /// $\omega(r) = -\frac{g_{03}}{g_{33}} = \frac{d\varphi}{d(ct)}$.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidRadius`] if $r \le r_+$.
    #[verified_engine::verified]
    pub fn frame_dragging_omega(&self, r: f64) -> Result<f64, HighEnergyError> {
        let r_plus = self.outer_horizon_radius();
        if !r.is_finite() || r <= r_plus {
            return Err(HighEnergyError::InvalidRadius {
                radius: r,
                limit: r_plus,
            });
        }
        let rs = self.schwarzschild_radius;
        let a = self.spin_a;
        let g33 = r * r + a * a + (a * a * rs) / r;
        Ok((a * rs) / (r * g33))
    }

    #[verified_engine::verified]
    fn radial_metric_derivatives(&self, r: f64, delta: f64) -> SMatrix<f64, 4, 4> {
        let rs = self.schwarzschild_radius;
        let a = self.spin_a;
        let inv_r2 = 1.0 / (r * r);
        let mut dg_dr = SMatrix::<f64, 4, 4>::zeros();
        dg_dr[(0, 0)] = -rs * inv_r2;
        let dg_03 = a * rs * inv_r2;
        dg_dr[(0, 3)] = dg_03;
        dg_dr[(3, 0)] = dg_03;
        dg_dr[(1, 1)] = (r * (2.0 * a * a - rs * r)) / (delta * delta);
        dg_dr[(2, 2)] = 2.0 * r;
        dg_dr[(3, 3)] = 2.0 * r - a * a * rs * inv_r2;
        dg_dr
    }
}

impl StaticMetric<4> for KerrEquatorialMetric {
    #[verified_engine::verified]
    fn metric_at(&self, point: &SVector<f64, 4>) -> Result<SMatrix<f64, 4, 4>, TensorError> {
        if !point.iter().all(|v| v.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        let rs = self.schwarzschild_radius;
        let a = self.spin_a;
        let r = point[1];
        if r <= self.outer_horizon_radius() {
            return Err(TensorError::SingularMetric);
        }
        let delta = r * r - rs * r + a * a;
        if delta <= 1e-14 {
            return Err(TensorError::SingularMetric);
        }

        let mut g = SMatrix::<f64, 4, 4>::zeros();
        g[(0, 0)] = -(1.0 - rs / r);
        let g03 = -(a * rs) / r;
        g[(0, 3)] = g03;
        g[(3, 0)] = g03;
        g[(1, 1)] = (r * r) / delta;
        g[(2, 2)] = r * r;
        g[(3, 3)] = r * r + a * a + (a * a * rs) / r;
        Ok(g)
    }

    #[verified_engine::verified]
    fn inverse_metric_at(
        &self,
        point: &SVector<f64, 4>,
    ) -> Result<SMatrix<f64, 4, 4>, TensorError> {
        if !point.iter().all(|v| v.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        let rs = self.schwarzschild_radius;
        let a = self.spin_a;
        let r = point[1];
        if r <= self.outer_horizon_radius() {
            return Err(TensorError::SingularMetric);
        }
        let delta = r * r - rs * r + a * a;
        if delta <= 1e-14 {
            return Err(TensorError::SingularMetric);
        }

        let g33 = r * r + a * a + (a * a * rs) / r;
        let mut g_inv = SMatrix::<f64, 4, 4>::zeros();
        g_inv[(0, 0)] = -g33 / delta;
        let g_inv_03 = -(a * rs) / (r * delta);
        g_inv[(0, 3)] = g_inv_03;
        g_inv[(3, 0)] = g_inv_03;
        g_inv[(1, 1)] = delta / (r * r);
        g_inv[(2, 2)] = 1.0 / (r * r);
        g_inv[(3, 3)] = (1.0 - rs / r) / delta;
        Ok(g_inv)
    }

    #[verified_engine::verified]
    fn christoffel_at(
        &self,
        point: &SVector<f64, 4>,
    ) -> Result<[SMatrix<f64, 4, 4>; 4], TensorError> {
        let g_inv = self.inverse_metric_at(point)?;
        let rs = self.schwarzschild_radius;
        let a = self.spin_a;
        let r = point[1];
        let delta = r * r - rs * r + a * a;
        let dg_dr = self.radial_metric_derivatives(r, delta);

        let mut gamma = [SMatrix::<f64, 4, 4>::zeros(); 4];
        for mu in 0..4 {
            for alpha in 0..4 {
                for beta in alpha..4 {
                    let mut sum = 0.0;
                    for nu in 0..4 {
                        let d_alpha = if alpha == 1 { dg_dr[(nu, beta)] } else { 0.0 };
                        let d_beta = if beta == 1 { dg_dr[(nu, alpha)] } else { 0.0 };
                        let d_nu = if nu == 1 { dg_dr[(alpha, beta)] } else { 0.0 };
                        sum += 0.5 * g_inv[(mu, nu)] * (d_alpha + d_beta - d_nu);
                    }
                    gamma[mu][(alpha, beta)] = sum;
                    gamma[mu][(beta, alpha)] = sum;
                }
            }
        }
        Ok(gamma)
    }

    #[verified_engine::verified]
    fn is_regular_at(&self, point: &SVector<f64, 4>) -> bool {
        let r = point[1];
        let theta = point[2];
        let r_plus = self.outer_horizon_radius();
        self.schwarzschild_radius.is_finite()
            && self.schwarzschild_radius > 0.0
            && point.iter().all(|v| v.is_finite())
            && r > 1.02 * r_plus
            && theta > 1e-4
            && theta < PI - 1e-4
            && theta.sin() > 1e-4
    }
}
