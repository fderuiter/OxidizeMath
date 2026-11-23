//! Schwarzschild spacetime metric, orbital potentials, Flamm paraboloid, and twin clock dilation.

use std::f64::consts::PI;

use math_commons::constants::{C, G};
use nalgebra::{SMatrix, SVector};
use pure_math::pure_math::tensor::{StaticMetric, TensorError};

use crate::error::HighEnergyError;
use crate::physics::high_energy::observer::calculate_lorentz_factor;

/// Classification of worldline causal character for radial effective potentials.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeodesicKind {
    /// Massive test particle parameterized by $c\tau$ ($\kappa = 1$, $g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu = -1$).
    Timelike,
    /// Massless photon ray ($\kappa = 0$, $g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu = 0$).
    Null,
}

impl GeodesicKind {
    /// Returns $\kappa = 1.0$ for timelike particles and $\kappa = 0.0$ for null rays.
    #[verified_engine::verified]
    pub fn kappa(self) -> f64 {
        match self {
            Self::Timelike => 1.0,
            Self::Null => 0.0,
        }
    }
}

/// Static spherically symmetric Schwarzschild black hole in $(ct, r, \theta, \varphi)$ coordinates.
///
/// Line element (signature $-, +, +, +$):
/// $$ds^2 = -\left(1 - \frac{R_s}{r}\right) d(ct)^2
///        + \left(1 - \frac{R_s}{r}\right)^{-1} dr^2
///        + r^2 d\theta^2 + r^2 \sin^2\theta\,d\varphi^2$$
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchwarzschildBlackHole {
    /// Black hole mass $M > 0$ in kilograms.
    pub mass: f64,
}

impl SchwarzschildBlackHole {
    /// Creates a [`SchwarzschildBlackHole`] from physical mass $M$ in kilograms.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidMass`] if `mass` is non-positive or non-finite.
    #[verified_engine::verified]
    pub fn new(mass: f64) -> Result<Self, HighEnergyError> {
        if !mass.is_finite() || mass <= 0.0 {
            return Err(HighEnergyError::InvalidMass { mass });
        }
        Ok(Self { mass })
    }

    /// Creates a [`SchwarzschildBlackHole`] parameterized directly by its Schwarzschild radius $R_s$.
    ///
    /// Useful for geometric-unit simulations (e.g., $R_s = 2.0$ corresponding to $M_{\text{geom}} = 1$).
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidRadius`] if `r_s` is non-positive or non-finite.
    #[verified_engine::verified]
    pub fn from_schwarzschild_radius(r_s: f64) -> Result<Self, HighEnergyError> {
        if !r_s.is_finite() || r_s <= 0.0 {
            return Err(HighEnergyError::InvalidRadius {
                radius: r_s,
                limit: 0.0,
            });
        }
        let mass = (r_s * C * C) / (2.0 * G);
        Self::new(mass)
    }

    /// Calculates the Schwarzschild event horizon radius $R_s = \frac{2GM}{c^2}$.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidMass`] if `self.mass` was mutated to an invalid value.
    #[verified_engine::verified]
    pub fn schwarzschild_radius(&self) -> Result<f64, HighEnergyError> {
        if !self.mass.is_finite() || self.mass <= 0.0 {
            return Err(HighEnergyError::InvalidMass { mass: self.mass });
        }
        Ok((2.0 * G * self.mass) / C.powi(2))
    }

    /// Calculates the gravitational time dilation factor $d\tau / dt = \sqrt{1 - R_s / r}$.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidRadius`] if `r <= R_s` or `r` is non-finite.
    #[verified_engine::verified]
    pub fn gravitational_time_dilation(&self, r: f64) -> Result<f64, HighEnergyError> {
        let rs = self.schwarzschild_radius()?;
        if !r.is_finite() || r <= rs {
            return Err(HighEnergyError::InvalidRadius {
                radius: r,
                limit: rs,
            });
        }
        Ok((1.0 - rs / r).sqrt())
    }

    /// Calculates the Innermost Stable Circular Orbit radius $r_{\text{ISCO}} = 3 R_s = \frac{6GM}{c^2}$.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidMass`] if `self.mass` is invalid.
    #[verified_engine::verified]
    pub fn isco(&self) -> Result<f64, HighEnergyError> {
        Ok(3.0 * self.schwarzschild_radius()?)
    }

    /// Calculates the unstable circular photon sphere radius $r_{\text{ph}} = 1.5 R_s = \frac{3GM}{c^2}$.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidMass`] if `self.mass` is invalid.
    #[verified_engine::verified]
    pub fn photon_sphere_radius(&self) -> Result<f64, HighEnergyError> {
        eval_photon_sphere_radius(self.schwarzschild_radius()?)
    }

    /// Evaluates the radial effective potential $V_{\text{eff}}(r) = \frac{1}{2}(1 - \frac{R_s}{r})(\kappa + \frac{\ell^2}{r^2})$.
    ///
    /// Satisfies $\frac{1}{2}\dot{r}^2 + V_{\text{eff}}(r) = \frac{1}{2} e^2$ for conserved
    /// energy parameter $e = (1 - R_s/r)\dot{x}^0$ and specific angular momentum $\ell = r^2\dot{\varphi}$.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidRadius`] if `r <= R_s` or inputs are non-finite.
    #[verified_engine::verified]
    pub fn effective_potential(
        &self,
        r: f64,
        specific_angular_momentum: f64,
        kind: GeodesicKind,
    ) -> Result<f64, HighEnergyError> {
        eval_effective_potential(
            self.schwarzschild_radius()?,
            r,
            specific_angular_momentum,
            kind,
        )
    }

    /// Evaluates Flamm's paraboloid embedding height $z(r) = 2\sqrt{R_s(r - R_s)}$ for $r \ge R_s$.
    ///
    /// Embeds the equatorial spatial slice $(r, \varphi)$ isometrically into Euclidean $\mathbb{R}^3$.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidRadius`] if `r < R_s` or `r` is non-finite.
    #[verified_engine::verified]
    pub fn flamm_paraboloid_z(&self, r: f64) -> Result<f64, HighEnergyError> {
        eval_flamm_paraboloid_z(self.schwarzschild_radius()?, r)
    }

    /// Computes combined proper-to-coordinate clock rate $\frac{d\tau}{dt} = \frac{\sqrt{1 - R_s/r}}{\gamma(v)}$.
    ///
    /// Multiplies general-relativistic [`SchwarzschildBlackHole::gravitational_time_dilation`]
    /// with special-relativistic kinematic dilation $1 / \gamma(v)$ from [`calculate_lorentz_factor`].
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidRadius`] if `r <= R_s` or [`HighEnergyError::InvalidVelocity`]
    /// if `|local_velocity_m_s| >= c`.
    #[verified_engine::verified]
    pub fn combined_clock_rate(
        &self,
        r: f64,
        local_velocity_m_s: f64,
    ) -> Result<f64, HighEnergyError> {
        let grav = self.gravitational_time_dilation(r)?;
        let gamma = calculate_lorentz_factor(local_velocity_m_s)?;
        Ok(grav / gamma)
    }
}

#[verified_engine::verified]
fn eval_photon_sphere_radius(r_s: f64) -> Result<f64, HighEnergyError> {
    if !r_s.is_finite() || r_s <= 0.0 {
        return Err(HighEnergyError::InvalidRadius {
            radius: r_s,
            limit: 0.0,
        });
    }
    Ok(1.5 * r_s)
}

#[verified_engine::verified]
fn eval_effective_potential(
    r_s: f64,
    r: f64,
    specific_angular_momentum: f64,
    kind: GeodesicKind,
) -> Result<f64, HighEnergyError> {
    if !r_s.is_finite() || r_s <= 0.0 {
        return Err(HighEnergyError::InvalidRadius {
            radius: r_s,
            limit: 0.0,
        });
    }
    if !r.is_finite() || r <= r_s || !specific_angular_momentum.is_finite() {
        return Err(HighEnergyError::InvalidRadius {
            radius: r,
            limit: r_s,
        });
    }
    let f = 1.0 - r_s / r;
    let ang_term = (specific_angular_momentum * specific_angular_momentum) / (r * r);
    Ok(0.5 * f * (kind.kappa() + ang_term))
}

#[verified_engine::verified]
fn eval_flamm_paraboloid_z(r_s: f64, r: f64) -> Result<f64, HighEnergyError> {
    if !r_s.is_finite() || r_s <= 0.0 {
        return Err(HighEnergyError::InvalidRadius {
            radius: r_s,
            limit: 0.0,
        });
    }
    if !r.is_finite() || r < r_s {
        return Err(HighEnergyError::InvalidRadius {
            radius: r,
            limit: r_s,
        });
    }
    Ok(2.0 * (r_s * (r - r_s).max(0.0)).sqrt())
}

/// Computes the unstable photon sphere radius $r_{\text{ph}} = 1.5 R_s$ from $R_s > 0$.
///
/// # Errors
/// Returns [`HighEnergyError::InvalidRadius`] if `r_s <= 0.0` or `r_s` is non-finite.
#[verified_engine::verified]
pub fn photon_sphere_radius(r_s: f64) -> Result<f64, HighEnergyError> {
    eval_photon_sphere_radius(r_s)
}

/// Evaluates the Schwarzschild radial effective potential
/// $V_{\text{eff}}(r) = \frac{1}{2}(1 - \frac{R_s}{r})(\kappa + \frac{\ell^2}{r^2})$.
///
/// # Errors
/// Returns [`HighEnergyError::InvalidRadius`] if `r_s <= 0`, `r <= r_s`, or inputs are non-finite.
#[verified_engine::verified]
pub fn effective_potential(
    r_s: f64,
    r: f64,
    specific_angular_momentum: f64,
    kind: GeodesicKind,
) -> Result<f64, HighEnergyError> {
    eval_effective_potential(r_s, r, specific_angular_momentum, kind)
}

/// Evaluates Flamm's paraboloid height $z(r) = 2\sqrt{R_s(r - R_s)}$ for $r \ge R_s$.
///
/// # Errors
/// Returns [`HighEnergyError::InvalidRadius`] if `r_s <= 0`, `r < r_s`, or inputs are non-finite.
#[verified_engine::verified]
pub fn flamm_paraboloid_z(r_s: f64, r: f64) -> Result<f64, HighEnergyError> {
    eval_flamm_paraboloid_z(r_s, r)
}

/// Computes combined gravitational and kinematic clock rate $\frac{d\tau}{dt}$ for an observer at
/// radius `r` moving with local shell speed `local_velocity_m_s`.
///
/// # Errors
/// Propagates [`HighEnergyError`] if `r <= R_s` or `|local_velocity_m_s| >= c`.
#[verified_engine::verified]
pub fn combined_clock_rate(
    bh: &SchwarzschildBlackHole,
    r: f64,
    local_velocity_m_s: f64,
) -> Result<f64, HighEnergyError> {
    bh.combined_clock_rate(r, local_velocity_m_s)
}

impl StaticMetric<4> for SchwarzschildBlackHole {
    #[verified_engine::verified]
    fn metric_at(&self, point: &SVector<f64, 4>) -> Result<SMatrix<f64, 4, 4>, TensorError> {
        let rs = self
            .schwarzschild_radius()
            .map_err(|_| TensorError::SingularMetric)?;
        if !point.iter().all(|v| v.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        let r = point[1];
        let theta = point[2];
        if r <= rs {
            return Err(TensorError::SingularMetric);
        }
        let f = 1.0 - rs / r;
        let sin_t = theta.sin();
        let mut g = SMatrix::<f64, 4, 4>::zeros();
        g[(0, 0)] = -f;
        g[(1, 1)] = 1.0 / f;
        g[(2, 2)] = r * r;
        g[(3, 3)] = r * r * sin_t * sin_t;
        Ok(g)
    }

    #[verified_engine::verified]
    fn inverse_metric_at(
        &self,
        point: &SVector<f64, 4>,
    ) -> Result<SMatrix<f64, 4, 4>, TensorError> {
        let rs = self
            .schwarzschild_radius()
            .map_err(|_| TensorError::SingularMetric)?;
        if !point.iter().all(|v| v.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        let r = point[1];
        let sin_t = point[2].sin();
        if r <= rs || sin_t.abs() <= 1e-12 {
            return Err(TensorError::SingularMetric);
        }
        let f = 1.0 - rs / r;
        let inv_r2 = 1.0 / (r * r);
        let mut g_inv = SMatrix::<f64, 4, 4>::zeros();
        g_inv[(0, 0)] = -1.0 / f;
        g_inv[(1, 1)] = f;
        g_inv[(2, 2)] = inv_r2;
        g_inv[(3, 3)] = inv_r2 / (sin_t * sin_t);
        Ok(g_inv)
    }

    #[verified_engine::verified]
    fn christoffel_at(
        &self,
        point: &SVector<f64, 4>,
    ) -> Result<[SMatrix<f64, 4, 4>; 4], TensorError> {
        let rs = self
            .schwarzschild_radius()
            .map_err(|_| TensorError::SingularMetric)?;
        if !point.iter().all(|v| v.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        let r = point[1];
        let (sin_t, cos_t) = point[2].sin_cos();
        if r <= rs || sin_t.abs() <= 1e-12 {
            return Err(TensorError::SingularMetric);
        }

        let mut gamma = [SMatrix::<f64, 4, 4>::zeros(); 4];
        let denom_r = 2.0 * r * (r - rs);
        let g0_01 = rs / denom_r;
        gamma[0][(0, 1)] = g0_01;
        gamma[0][(1, 0)] = g0_01;

        gamma[1][(0, 0)] = (rs * (r - rs)) / (2.0 * r * r * r);
        gamma[1][(1, 1)] = -g0_01;
        gamma[1][(2, 2)] = -(r - rs);
        gamma[1][(3, 3)] = -(r - rs) * sin_t * sin_t;

        let inv_r = 1.0 / r;
        gamma[2][(1, 2)] = inv_r;
        gamma[2][(2, 1)] = inv_r;
        gamma[2][(3, 3)] = -sin_t * cos_t;

        let cot_t = cos_t / sin_t;
        gamma[3][(1, 3)] = inv_r;
        gamma[3][(3, 1)] = inv_r;
        gamma[3][(2, 3)] = cot_t;
        gamma[3][(3, 2)] = cot_t;
        Ok(gamma)
    }

    #[verified_engine::verified]
    fn is_regular_at(&self, point: &SVector<f64, 4>) -> bool {
        let Ok(rs) = self.schwarzschild_radius() else {
            return false;
        };
        let r = point[1];
        let theta = point[2];
        point.iter().all(|v| v.is_finite())
            && r > 1.02 * rs
            && theta > 1e-4
            && theta < PI - 1e-4
            && theta.sin() > 1e-4
    }
}
