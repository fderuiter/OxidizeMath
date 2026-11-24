//! Equatorial Kerr spacetime metric (`StaticMetric<4>`) with Lense-Thirring frame dragging.

use std::f64::consts::PI;

use math_commons::constants::{C, G};
use nalgebra::{SMatrix, SVector};
use pure_math::pure_math::tensor::{StaticMetric, TensorError};

use crate::error::HighEnergyError;
use crate::physics::high_energy::general_relativity::schwarzschild::GeodesicKind;

/// Rotating Kerr black hole restricted to the equatorial slice ($\theta = \pi/2$) in
/// Boyer-Lindquist coordinates $(ct, r, \theta, \varphi)$.
///
/// Line element near $\theta = \pi/2$ (with $\Delta = r^2 - R_s r + a^2$):
/// $$ds^2 = -\left(1 - \frac{R_s}{r}\right) d(ct)^2
///        - \frac{2 a R_s}{r} d(ct)\,d\varphi
///        + \frac{r^2}{\Delta} dr^2 + r^2 d\theta^2
///        + \left(r^2 + a^2 + \frac{a^2 R_s}{r}\right) d\varphi^2$$
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KerrEquatorialMetric {
    /// Schwarzschild mass scale $R_s = \frac{2GM}{c^2} > 0$.
    pub schwarzschild_radius: f64,
    /// Kerr spin length parameter $a = \frac{J}{Mc}$ satisfying $|a| < \frac{1}{2} R_s$.
    pub spin_a: f64,
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

    /// Evaluates exact equatorial proper-to-coordinate clock dilation $d\tau / dt$ for a circular
    /// worldline at radius `r` with coordinate angular rate `dphi_dct` ($\Omega = d\varphi / d(ct)$).
    ///
    /// Uses the ADM lapse-shift identity around [`Self::frame_dragging_omega`] and
    /// [`Self::ergosphere_radius`]:
    /// $$\left(\frac{d\tau}{dt}\right)^2 = \frac{\Delta}{g_{33}} - g_{33}\left(\Omega - \omega(r)\right)^2$$
    /// Static observers ($\Omega = 0$) fail inside $r \le r_{\text{ergo}} = R_s$, whereas frame-dragged
    /// observers ($\Omega \approx \omega(r)$) remain timelike down to $r > r_+$.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidRadius`] if $r \le r_+$ or [`HighEnergyError::InvalidVelocity`]
    /// if the specified angular rate yields a null or spacelike worldline.
    #[verified_engine::verified]
    pub fn clock_dilation(&self, r: f64, dphi_dct: f64) -> Result<f64, HighEnergyError> {
        let omega_zamo = self.frame_dragging_omega(r)?;
        if !dphi_dct.is_finite() {
            return Err(HighEnergyError::InvalidVelocity { v: f64::NAN });
        }
        let r_ergo = self.ergosphere_radius();
        let a = self.spin_a;
        let delta = r * r - r_ergo * r + a * a;
        let g33 = r * r + a * a + (a * a * r_ergo) / r;
        let zamo_lapse_sq = delta / g33;
        let rel_omega = dphi_dct - omega_zamo;
        let dtau_dt_sq = zamo_lapse_sq - g33 * rel_omega * rel_omega;
        if dtau_dt_sq <= 0.0 {
            return Err(HighEnergyError::InvalidVelocity { v: C });
        }
        Ok(dtau_dt_sq.sqrt())
    }

    /// Evaluates the Zero-Angular-Momentum Observer (ZAMO) clock dilation $d\tau / dt$ at radius `r`
    /// corotating at [`Self::frame_dragging_omega`].
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidRadius`] if $r \le r_+$.
    #[verified_engine::verified]
    pub fn zamo_clock_dilation(&self, r: f64) -> Result<f64, HighEnergyError> {
        let omega = self.frame_dragging_omega(r)?;
        self.clock_dilation(r, omega)
    }

    /// Evaluates the exact equatorial Kerr radial effective potential $V_{\text{eff}}(r; e, \ell)$
    /// satisfying $\frac{1}{2}\dot{r}^2 + V_{\text{eff}}(r; e, \ell) = \frac{1}{2} e^2$.
    ///
    /// $$V_{\text{eff}}(r; e, \ell) = \frac{\kappa \Delta}{2 r^2}
    ///   + \frac{(r - r_{\text{ergo}})\ell^2}{2 r^3}
    ///   + \frac{a R_s e \ell}{r^3}
    ///   - \frac{a^2 e^2 (r + R_s)}{2 r^3}$$
    /// Reduces identically to Schwarzschild when $a = 0$.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidRadius`] if $r \le r_+$ or inputs are non-finite.
    #[verified_engine::verified]
    pub fn effective_potential(
        &self,
        r: f64,
        energy_e: f64,
        angular_momentum_l: f64,
        kind: GeodesicKind,
    ) -> Result<f64, HighEnergyError> {
        let r_plus = self.outer_horizon_radius();
        if !r.is_finite() || r <= r_plus || !energy_e.is_finite() || !angular_momentum_l.is_finite()
        {
            return Err(HighEnergyError::InvalidRadius {
                radius: r,
                limit: r_plus,
            });
        }
        let rs = self.schwarzschild_radius;
        let r_ergo = self.ergosphere_radius();
        let a = self.spin_a;
        let delta = r * r - rs * r + a * a;
        let r2 = r * r;
        let r3 = r2 * r;
        let mass_term = (0.5 * kind.kappa() * delta) / r2;
        let centrif_term = ((r - r_ergo) * angular_momentum_l * angular_momentum_l) / (2.0 * r3);
        let spin_orbit_term = (a * rs * energy_e * angular_momentum_l) / r3;
        let spin_quad_term = -(a * a * energy_e * energy_e * (r + rs)) / (2.0 * r3);
        Ok(mass_term + centrif_term + spin_orbit_term + spin_quad_term)
    }

    /// Unstable circular equatorial photon orbit radius $r_{\text{ph},\pm}$ (Bardeen 1972)
    /// for rays corotating (`prograde == true`) or counter-rotating (`prograde == false`) with $|a|$.
    #[verified_engine::verified]
    pub fn photon_orbit_radius(&self, prograde: bool) -> f64 {
        let rs = self.schwarzschild_radius;
        let chi_abs = ((2.0 * self.spin_a.abs()) / rs).clamp(0.0, 0.999_999);
        let arg = if prograde { -chi_abs } else { chi_abs };
        rs * (1.0 + ((2.0 / 3.0) * arg.acos()).cos())
    }

    /// Critical signed impact parameter $b_{\text{crit}} = \ell / e$ at the prograde or retrograde
    /// circular photon orbit, signed relative to `self.spin_a` (`+` for corotating when $a \ge 0$).
    #[verified_engine::verified]
    pub fn critical_impact_parameter(&self, prograde: bool) -> f64 {
        let rs = self.schwarzschild_radius;
        let a_abs = self.spin_a.abs();
        let r_ph = self.photon_orbit_radius(prograde);
        let delta = (r_ph * r_ph - rs * r_ph + a_abs * a_abs).max(0.0);
        let numer = r_ph * r_ph * r_ph + a_abs * a_abs * (r_ph + rs);
        let b_mag = if prograde {
            numer / (r_ph * delta.sqrt() + a_abs * rs).max(1e-15)
        } else {
            numer / (r_ph * delta.sqrt() - a_abs * rs).max(1e-15)
        };
        let spin_sign = if self.spin_a < 0.0 { -1.0 } else { 1.0 };
        if prograde {
            spin_sign * b_mag
        } else {
            -spin_sign * b_mag
        }
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
