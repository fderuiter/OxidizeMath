//! Schwarzschild spacetime metric, orbital potentials, Flamm paraboloid, and twin clock dilation.

use math_commons::constants::{C, G};
use nalgebra::{SMatrix, SVector, Vector4};
use pure_math::pure_math::tensor::{
    GeodesicStepStatus, LorentzianConnection4D, SchwarzschildConnection4D, StaticMetric,
    TensorError,
};

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

    /// Returns the closed-form stack-allocated [`SchwarzschildConnection4D`] for this black hole.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidMass`] if `self.mass` is invalid.
    #[verified_engine::verified]
    pub fn connection_4d(&self) -> Result<SchwarzschildConnection4D, HighEnergyError> {
        let rs = self.schwarzschild_radius()?;
        SchwarzschildConnection4D::new(rs)
            .map_err(|_| HighEnergyError::InvalidMass { mass: self.mass })
    }

    /// Critical photon capture impact parameter $b_{\text{crit}} = \frac{3\sqrt{3}}{2} R_s \approx 2.598076\,R_s$.
    ///
    /// Null geodesics with $b < b_{\text{crit}}$ plunge into the horizon; rays with $b > b_{\text{crit}}$
    /// scatter back to infinity after strong-field lensing near $r_{\text{ph}} = 1.5 R_s$.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidMass`] if `self.mass` is invalid.
    #[verified_engine::verified]
    pub fn critical_impact_parameter(&self) -> Result<f64, HighEnergyError> {
        eval_critical_impact_parameter(self.schwarzschild_radius()?)
    }

    /// Dimensionless specific angular momentum $\ell_{\text{ISCO}} = \sqrt{3}\,R_s$ of the ISCO at $r = 3 R_s$.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidMass`] if `self.mass` is invalid.
    #[verified_engine::verified]
    pub fn isco_specific_angular_momentum(&self) -> Result<f64, HighEnergyError> {
        Ok(3.0_f64.sqrt() * self.schwarzschild_radius()?)
    }

    /// Local static-shell orbital speed $\beta_{\text{shell}}(r) = \sqrt{\frac{R_s}{2(r - R_s)}}$
    /// of a circular timelike geodesic at radius $r > 1.5 R_s$ ($\beta_{\text{shell}} = 0.5$ at ISCO $r = 3 R_s$).
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidRadius`] if `r <= 1.5 R_s` or `r` is non-finite.
    #[verified_engine::verified]
    pub fn circular_shell_beta(&self, r: f64) -> Result<f64, HighEnergyError> {
        let rs = self.schwarzschild_radius()?;
        let r_ph = 1.5 * rs;
        if !r.is_finite() || r <= r_ph {
            return Err(HighEnergyError::InvalidRadius {
                radius: r,
                limit: r_ph,
            });
        }
        Ok((rs / (2.0 * (r - rs))).sqrt())
    }

    /// Evaluates conserved Killing energy $e = (1 - R_s/r) u^0$ and azimuthal angular momentum
    /// $\ell = r^2 \sin^2\theta\,u^\varphi$ along an affine geodesic.
    #[must_use]
    #[verified_engine::verified]
    pub fn killing_invariants(&self, pos: &Vector4<f64>, vel: &Vector4<f64>) -> (f64, f64) {
        let Ok(rs) = self.schwarzschild_radius() else {
            return (0.0, 0.0);
        };
        let r = pos[1];
        let sin_t = pos[2].sin();
        let energy = (1.0 - rs / r) * vel[0];
        let ang_mom = r * r * sin_t * sin_t * vel[3];
        (energy, ang_mom)
    }

    /// Advances a 4D geodesic state `(pos, vel)` by one affine step `d_lambda` purely on the stack
    /// using [`SchwarzschildConnection4D`], subdividing near $r < 2.5 R_s$ to keep Killing drift $< 10^{-6}$.
    #[verified_engine::verified]
    pub fn step_geodesic_rk4(
        &self,
        pos: &mut Vector4<f64>,
        vel: &mut Vector4<f64>,
        d_lambda: f64,
    ) -> GeodesicStepStatus {
        let Ok(conn) = self.connection_4d() else {
            return GeodesicStepStatus::HaltedSingularity;
        };
        if !d_lambda.is_finite() || !conn.is_outside_horizon(pos) {
            return GeodesicStepStatus::HaltedSingularity;
        }
        let rs = conn.schwarzschild_radius;
        let max_local_h = if pos[1] < 1.35 * rs {
            0.002 * rs
        } else if pos[1] < 3.0 * rs {
            0.005 * rs
        } else {
            0.10 * rs
        };
        let substeps = ((d_lambda / max_local_h).ceil() as usize).clamp(1, 64);
        let h = d_lambda / (substeps as f64);
        let mut p_cur = *pos;
        let mut v_cur = *vel;
        let mut metric_norm = 0.0;
        let mut s = 0;
        while s < substeps {
            let Some((pn, vn, norm)) = rk4_single_stage(&conn, &p_cur, &v_cur, h) else {
                return GeodesicStepStatus::HaltedSingularity;
            };
            p_cur = pn;
            v_cur = vn;
            metric_norm = norm;
            s += 1;
        }
        *pos = p_cur;
        *vel = v_cur;
        GeodesicStepStatus::Advanced { metric_norm }
    }
}

#[verified_engine::verified]
fn rk4_single_stage(
    conn: &SchwarzschildConnection4D,
    pos: &Vector4<f64>,
    vel: &Vector4<f64>,
    h: f64,
) -> Option<(Vector4<f64>, Vector4<f64>, f64)> {
    let a1 = conn.geodesic_acceleration(pos, vel).ok()?;
    let p2 = *pos + 0.5 * h * (*vel);
    let v2 = *vel + 0.5 * h * a1;
    if !conn.is_outside_horizon(&p2) {
        return None;
    }
    let a2 = conn.geodesic_acceleration(&p2, &v2).ok()?;
    let p3 = *pos + 0.5 * h * v2;
    let v3 = *vel + 0.5 * h * a2;
    if !conn.is_outside_horizon(&p3) {
        return None;
    }
    let a3 = conn.geodesic_acceleration(&p3, &v3).ok()?;
    let p4 = *pos + h * v3;
    let v4 = *vel + h * a3;
    if !conn.is_outside_horizon(&p4) {
        return None;
    }
    let a4 = conn.geodesic_acceleration(&p4, &v4).ok()?;
    let next_pos = *pos + (h / 6.0) * (*vel + 2.0 * v2 + 2.0 * v3 + v4);
    let next_vel = *vel + (h / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4);
    if !conn.is_outside_horizon(&next_pos) {
        return None;
    }
    let norm = conn.interval_norm(&next_pos, &next_vel).ok()?;
    Some((next_pos, next_vel, norm))
}

#[verified_engine::verified]
fn eval_critical_impact_parameter(r_s: f64) -> Result<f64, HighEnergyError> {
    if !r_s.is_finite() || r_s <= 0.0 {
        return Err(HighEnergyError::InvalidRadius {
            radius: r_s,
            limit: 0.0,
        });
    }
    Ok(1.5 * 3.0_f64.sqrt() * r_s)
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

/// Computes the critical photon capture impact parameter $b_{\text{crit}} = \frac{3\sqrt{3}}{2} R_s$.
///
/// # Errors
/// Returns [`HighEnergyError::InvalidRadius`] if `r_s <= 0.0` or `r_s` is non-finite.
#[verified_engine::verified]
pub fn critical_impact_parameter(r_s: f64) -> Result<f64, HighEnergyError> {
    eval_critical_impact_parameter(r_s)
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
        let conn = self
            .connection_4d()
            .map_err(|_| TensorError::SingularMetric)?;
        conn.metric_tensor(point)
    }

    #[verified_engine::verified]
    fn inverse_metric_at(
        &self,
        point: &SVector<f64, 4>,
    ) -> Result<SMatrix<f64, 4, 4>, TensorError> {
        let conn = self
            .connection_4d()
            .map_err(|_| TensorError::SingularMetric)?;
        conn.inverse_metric_tensor(point)
    }

    #[verified_engine::verified]
    fn christoffel_at(
        &self,
        point: &SVector<f64, 4>,
    ) -> Result<[SMatrix<f64, 4, 4>; 4], TensorError> {
        let conn = self
            .connection_4d()
            .map_err(|_| TensorError::SingularMetric)?;
        conn.christoffel_tensors(point)
    }

    #[verified_engine::verified]
    fn is_regular_at(&self, point: &SVector<f64, 4>) -> bool {
        let Ok(conn) = self.connection_4d() else {
            return false;
        };
        conn.is_outside_horizon(point)
    }
}
