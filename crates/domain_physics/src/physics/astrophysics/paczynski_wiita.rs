//! # Paczyński-Wiita Relativistic Black Hole Potential & Beaming
//!
//! Couples a central [`SchwarzschildBlackHole`] pseudo-Newtonian Paczyński-Wiita (1980)
//! potential $\Phi_{\text{PW}}(r) = -\frac{G M_{\text{BH}}}{r - R_s}$ into [`GalacticDiskSystem`],
//! reproducing exact $r_{\text{ISCO}} = 3 R_s$ capture, apsidal rosetta precession validated
//! against stack-allocated [`christoffel_symbols_fixed`], and relativistic gravitational +
//! Doppler + synchrotron beaming.
//!
//! # Example
//! ```rust
//! # use domain_physics::__doctest_facade as math_explorer;
//! use math_explorer::physics::astrophysics::{
//!     CentralBlackHoleConfig, paczynski_wiita_acceleration_kpc, validate_christoffel_radial_connection,
//! };
//!
//! let bh_cfg = CentralBlackHoleConfig::enabled_preset();
//! let bh = bh_cfg.to_schwarzschild_bh().unwrap();
//! let (gamma_r00, pw_factor) = validate_christoffel_radial_connection(&bh, 6.0).unwrap();
//! assert!(gamma_r00 > 0.0 && pw_factor > gamma_r00);
//! let (ax, _ay) = paczynski_wiita_acceleration_kpc(1.5, 0.0, &bh_cfg);
//! assert!(ax < 0.0);
//! ```

// [cite:dwarf_galaxy_empirical_dependencies]
// [cite:general_relativity_geodesics]

use std::f64::consts::FRAC_PI_2;

use math_commons::constants::{C, SOLAR_MASS};
use nalgebra::SVector;
use pure_math::pure_math::tensor::christoffel_symbols_fixed;

use crate::error::HighEnergyError;
use crate::physics::high_energy::general_relativity::SchwarzschildBlackHole;
use crate::physics::high_energy::observer::calculate_lorentz_factor;
use crate::physics::high_energy::radiation::synchrotron_power;

use super::orbits::{GalacticDiskSystem, OrbitalInvariants};
use super::rotation_curves::{G_KPC_KMS2_PER_SOLAR_MASS, total_gravitational_potential};

/// Strict upper bound on $|v| / c$ (`< 0.999`) preventing singular Lorentz factors.
pub const MAX_RELATIVISTIC_BETA: f64 = 0.998_999;

/// Minimum radial clearance factor above $R_s$ (`r >= 1.001 * R_s`) preventing horizon division by zero.
pub const HORIZON_CLEARANCE_FACTOR: f64 = 1.001;

/// Configuration for a central Schwarzschild black hole embedded in a galactic potential.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CentralBlackHoleConfig {
    /// Whether the central Paczyński-Wiita black hole potential and ISCO capture are active.
    pub enabled: bool,
    /// Central black hole gravitational mass $M_{\text{BH}}$ in solar masses ($M_\odot$).
    pub mass_solar: f64,
    /// Effective Schwarzschild horizon radius $R_s$ in kpc for potential steepening & ISCO ($3 R_s$).
    pub effective_rs_kpc: f64,
    /// Dimensionless velocity-to-$c$ mapping scale for relativistic Doppler & synchrotron beaming.
    pub relativistic_beta_scale: f64,
    /// Ambient coronal magnetic energy density $U_B$ ($\text{J}\,\text{m}^{-3}$) for synchrotron emission.
    pub magnetic_energy_density_j_m3: f64,
}

impl Default for CentralBlackHoleConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mass_solar: 1.0e8,
            effective_rs_kpc: 0.12,
            relativistic_beta_scale: 0.0025,
            magnetic_energy_density_j_m3: 1.0,
        }
    }
}

impl CentralBlackHoleConfig {
    /// Returns a pre-configured active supermassive black hole setup for rosetta & ISCO exploration.
    #[must_use]
    #[verified_engine::verified]
    pub fn enabled_preset() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }

    /// Returns `true` when this configuration matches [`Self::default`] (used for compact URL state).
    #[must_use]
    #[verified_engine::verified]
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Instantiates the corresponding [`SchwarzschildBlackHole`] metric in SI kilograms.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidMass`] if `mass_solar` is non-finite or non-positive.
    #[verified_engine::verified]
    pub fn to_schwarzschild_bh(&self) -> Result<SchwarzschildBlackHole, HighEnergyError> {
        let m_solar = if self.mass_solar.is_finite() {
            self.mass_solar.max(1.0)
        } else {
            1.0e8
        };
        SchwarzschildBlackHole::new(m_solar * SOLAR_MASS)
    }

    /// Clamped Schwarzschild horizon scale $R_s > 0$ in kpc.
    #[must_use]
    #[verified_engine::verified]
    pub fn horizon_radius_kpc(&self) -> f64 {
        if self.effective_rs_kpc.is_finite() {
            self.effective_rs_kpc.max(1.0e-6)
        } else {
            0.12
        }
    }

    /// Unstable circular photon-sphere radius $r_{\text{ph}} = 1.5 R_s$ in kpc.
    #[must_use]
    #[verified_engine::verified]
    pub fn photon_sphere_radius_kpc(&self) -> f64 {
        if let Ok(bh) = self.to_schwarzschild_bh()
            && let (Ok(rs), Ok(rph)) = (bh.schwarzschild_radius(), bh.photon_sphere_radius())
        {
            return (rph / rs) * self.horizon_radius_kpc();
        }
        1.5 * self.horizon_radius_kpc()
    }

    /// Innermost Stable Circular Orbit radius $r_{\text{ISCO}} = 3 R_s$ in kpc derived via [`SchwarzschildBlackHole::isco`].
    #[must_use]
    #[verified_engine::verified]
    pub fn isco_radius_kpc(&self) -> f64 {
        if let Ok(bh) = self.to_schwarzschild_bh()
            && let (Ok(rs), Ok(risco)) = (bh.schwarzschild_radius(), bh.isco())
        {
            return (risco / rs) * self.horizon_radius_kpc();
        }
        3.0 * self.horizon_radius_kpc()
    }
}

/// Evaluates the Paczyński-Wiita gravitational potential $\Phi_{\text{PW}}(r) = -\frac{G M_{\text{BH}}}{r_{\text{eff}} - R_s}$ in $(\text{km/s})^2$.
#[must_use]
#[verified_engine::verified]
pub fn paczynski_wiita_potential_kpc(r_kpc: f64, bh: &CentralBlackHoleConfig) -> f64 {
    if !bh.enabled {
        return 0.0;
    }
    let rs = bh.horizon_radius_kpc();
    let r_abs = if r_kpc.is_finite() {
        r_kpc.abs()
    } else {
        HORIZON_CLEARANCE_FACTOR * rs
    };
    let r_eff = r_abs.max(HORIZON_CLEARANCE_FACTOR * rs);
    let mu = G_KPC_KMS2_PER_SOLAR_MASS * bh.mass_solar.max(0.0);
    -mu / (r_eff - rs)
}

/// Evaluates planar Paczyński-Wiita central acceleration $(a_x, a_y)$ in $(\text{km/s})^2/\text{kpc}$.
#[must_use]
#[verified_engine::verified]
pub fn paczynski_wiita_acceleration_kpc(
    x_kpc: f64,
    y_kpc: f64,
    bh: &CentralBlackHoleConfig,
) -> (f64, f64) {
    if !bh.enabled || !x_kpc.is_finite() || !y_kpc.is_finite() {
        return (0.0, 0.0);
    }
    let r = (x_kpc * x_kpc + y_kpc * y_kpc).sqrt();
    if r <= 1.0e-12 {
        return (0.0, 0.0);
    }
    let rs = bh.horizon_radius_kpc();
    let r_eff = r.max(HORIZON_CLEARANCE_FACTOR * rs);
    let denom = r * (r_eff - rs) * (r_eff - rs);
    let mu = G_KPC_KMS2_PER_SOLAR_MASS * bh.mass_solar.max(0.0);
    let factor = -mu / denom;
    (factor * x_kpc, factor * y_kpc)
}

/// Evaluates the Paczyński-Wiita circular orbital speed $v_{\text{PW}}(r) = \sqrt{\frac{G M_{\text{BH}} r}{(r - R_s)^2}}$ in km/s.
#[must_use]
#[verified_engine::verified]
pub fn paczynski_wiita_circular_speed_kms(r_kpc: f64, bh: &CentralBlackHoleConfig) -> f64 {
    if !bh.enabled {
        return 0.0;
    }
    let rs = bh.horizon_radius_kpc();
    let r_eff = if r_kpc.is_finite() {
        r_kpc.max(HORIZON_CLEARANCE_FACTOR * rs)
    } else {
        3.0 * rs
    };
    let mu = G_KPC_KMS2_PER_SOLAR_MASS * bh.mass_solar.max(0.0);
    ((mu * r_eff) / ((r_eff - rs) * (r_eff - rs)))
        .max(0.0)
        .sqrt()
}

/// Returns `true` when `bh.enabled` and planar radius $\sqrt{x^2 + y^2}$ crosses inside $r_{\text{ISCO}} = 3 R_s$.
#[must_use]
#[verified_engine::verified]
pub fn is_captured_at_isco(x_kpc: f64, y_kpc: f64, bh: &CentralBlackHoleConfig) -> bool {
    if !bh.enabled {
        return false;
    }
    if !x_kpc.is_finite() || !y_kpc.is_finite() {
        return true;
    }
    let r = (x_kpc * x_kpc + y_kpc * y_kpc).sqrt();
    r <= bh.isco_radius_kpc()
}

/// Cross-validates the equatorial radial connection component $\Gamma^r_{00}$ via stack-allocated
/// [`christoffel_symbols_fixed`] against the Paczyński-Wiita radial gradient $\frac{R_s}{2(r - R_s)^2}$.
///
/// Returns `(gamma_r_00_m_inv, paczynski_wiita_radial_m_inv)` at radius `r = max(r_over_rs, 1.01) * R_s`.
///
/// # Errors
/// Propagates [`HighEnergyError`] if `bh` is invalid or metric connection evaluation fails.
#[verified_engine::verified]
pub fn validate_christoffel_radial_connection(
    bh: &SchwarzschildBlackHole,
    r_over_rs: f64,
) -> Result<(f64, f64), HighEnergyError> {
    let rs = bh.schwarzschild_radius()?;
    let ratio = if r_over_rs.is_finite() {
        r_over_rs.max(1.01)
    } else {
        6.0
    };
    let r = ratio * rs;
    let point = SVector::<f64, 4>::new(0.0, r, FRAC_PI_2, 0.0);
    let gammas =
        christoffel_symbols_fixed(bh, &point).map_err(|_| HighEnergyError::InvalidRadius {
            radius: r,
            limit: rs,
        })?;
    let gamma_r00 = gammas[1][(0, 0)];
    let pw_radial = rs / (2.0 * (r - rs) * (r - rs));
    Ok((gamma_r00, pw_radial))
}

/// Combined general-relativistic and special-relativistic radiative beaming telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RelativisticBeamingSample {
    /// Clamped speed ratio $\beta = |v| / c \in [0, 0.999)$.
    pub beta: f64,
    /// Special-relativistic Lorentz factor $\gamma = (1 - \beta^2)^{-1/2} \ge 1$.
    pub lorentz_factor: f64,
    /// Schwarzschild gravitational clock rate $g_{\text{grav}} = \sqrt{1 - R_s / r} \in (0, 1)$.
    pub gravitational_time_dilation: f64,
    /// Relativistic kinematic Doppler factor $\delta = [\gamma(1 - \beta_{\text{los}})]^{-1}$.
    pub doppler_factor: f64,
    /// Single-electron synchrotron power $P_{\text{sync}}$ in Watts from [`synchrotron_power`].
    pub synchrotron_power_watts: f64,
    /// Net bolometric radiance boost weight combining gravitational redshift, Doppler beaming, and synchrotron emission.
    pub radiance_weight: f64,
}

/// Evaluates clamped gravitational time dilation, Lorentz factor, Doppler boost, and synchrotron power
/// at dimensionless radius `r_over_rs` ($r / R_s$, clamped $> 1.001$) and speed `speed_m_s` ($< 0.999 c$).
///
/// # Errors
/// Propagates [`HighEnergyError`] only if `bh` has invalid mass.
#[verified_engine::verified]
pub fn evaluate_relativistic_beaming(
    bh: &SchwarzschildBlackHole,
    r_over_rs: f64,
    speed_m_s: f64,
    los_cos: f64,
    magnetic_u_b: f64,
) -> Result<RelativisticBeamingSample, HighEnergyError> {
    let rs = bh.schwarzschild_radius()?;
    let safe_ratio = if r_over_rs.is_finite() {
        r_over_rs.max(HORIZON_CLEARANCE_FACTOR)
    } else {
        6.0
    };
    let r_m = safe_ratio * rs;
    let max_v = MAX_RELATIVISTIC_BETA * C;
    let v_mag = if speed_m_s.is_finite() {
        speed_m_s.abs().min(max_v)
    } else {
        0.0
    };
    let beta = (v_mag / C).clamp(0.0, MAX_RELATIVISTIC_BETA);
    let mu_los = if los_cos.is_finite() {
        los_cos.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let g_grav = bh.gravitational_time_dilation(r_m)?;
    let gamma = calculate_lorentz_factor(beta * C)?;
    let u_b = if magnetic_u_b.is_finite() {
        magnetic_u_b.max(0.0)
    } else {
        0.0
    };
    let sync_w = synchrotron_power(u_b, beta, gamma)?;
    let ref_sync =
        ((4.0 / 3.0) * math_commons::constants::SIGMA_T * C * u_b.max(1.0e-12)).max(1.0e-30);
    let sync_gain = (sync_w / ref_sync).clamp(0.0, 6.0);
    let doppler = (1.0 / (gamma * (1.0 - beta * mu_los)).max(1.0e-4)).clamp(0.05, 12.0);
    let radiance_weight =
        (g_grav * g_grav * doppler.powi(3) * (1.0 + 0.18 * sync_gain)).clamp(0.02, 16.0);

    Ok(RelativisticBeamingSample {
        beta,
        lorentz_factor: gamma,
        gravitational_time_dilation: g_grav,
        doppler_factor: doppler,
        synchrotron_power_watts: sync_w,
        radiance_weight,
    })
}

impl GalacticDiskSystem {
    /// Configures the central Paczyński-Wiita [`CentralBlackHoleConfig`] and reseeds baseline invariants.
    #[verified_engine::verified]
    pub fn set_central_black_hole(&mut self, central_bh: CentralBlackHoleConfig) {
        self.central_bh = central_bh;
        self.refresh_invariants_baseline();
    }

    /// Number of tracer particles currently captured inside $r_{\text{ISCO}} = 3 R_s$.
    #[must_use]
    #[verified_engine::verified]
    pub fn captured_particle_count(&self) -> usize {
        let mut count = 0;
        let mut i = 0;
        while i < self.particles().len() {
            if self.particles()[i].captured {
                count += 1;
            }
            i += 1;
        }
        count
    }

    /// Evaluates conserved orbital invariants for particle `index`.
    #[must_use]
    #[verified_engine::verified]
    pub fn particle_invariants(&self, index: usize) -> Option<OrbitalInvariants> {
        let p = self.particles.get(index)?;
        let r = (p.x_kpc * p.x_kpc + p.y_kpc * p.y_kpc).sqrt();
        let v_sq = p.vx_kms * p.vx_kms + p.vy_kms * p.vy_kms;
        let phi = total_gravitational_potential(r, &self.baryons, &self.halo)
            + paczynski_wiita_potential_kpc(r, &self.central_bh);
        Some(OrbitalInvariants {
            specific_energy_kms2: 0.5 * v_sq + phi,
            angular_momentum_kpc_kms: p.x_kpc * p.vy_kms - p.y_kpc * p.vx_kms,
            radius_kpc: r,
            speed_kms: v_sq.sqrt(),
        })
    }

    /// Sum of specific mechanical energies across all tracer particles in $(\text{km/s})^2$.
    #[must_use]
    #[verified_engine::verified]
    pub fn total_energy(&self) -> f64 {
        let mut sum = 0.0;
        let mut i = 0;
        while i < self.particles.len() {
            if let Some(inv) = self.particle_invariants(i) {
                sum += inv.specific_energy_kms2;
            }
            i += 1;
        }
        sum
    }

    /// Sum of planar specific angular momenta $L_z$ across all tracer particles in $\text{kpc}\cdot\text{km/s}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn total_angular_momentum(&self) -> f64 {
        let mut sum = 0.0;
        let mut i = 0;
        while i < self.particles.len() {
            let p = &self.particles[i];
            sum += p.x_kpc * p.vy_kms - p.y_kpc * p.vx_kms;
            i += 1;
        }
        sum
    }

    /// Dimensionless relative drift $|\Delta E| / \max(|E_0|, 10^{-12})$ from the initial ensemble energy.
    #[must_use]
    #[verified_engine::verified]
    pub fn relative_energy_drift(&self) -> f64 {
        let current = self.total_energy();
        (current - self.initial_energy).abs() / self.initial_energy.abs().max(1.0e-12)
    }

    /// Dimensionless relative drift $|\Delta L_z| / \max(|L_{z,0}|, 10^{-12})$ from the initial ensemble angular momentum.
    #[must_use]
    #[verified_engine::verified]
    pub fn relative_angular_momentum_drift(&self) -> f64 {
        let current = self.total_angular_momentum();
        (current - self.initial_lz).abs() / self.initial_lz.abs().max(1.0e-12)
    }
}
