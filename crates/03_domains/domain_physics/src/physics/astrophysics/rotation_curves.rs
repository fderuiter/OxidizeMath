//! # Dwarf Galaxy Baryonic & Dark-Matter Rotation Curves
//!
//! Translates the empirical scaling laws of Tillaboev, Tadjibaev & Otojanova (2025)
//! (`tillaboev2025`) into radially decomposed circular rotation curves
//! $v_{\text{circ}}(r) = \sqrt{v_{\text{disk}}^2 + v_{\text{gas}}^2 + v_{\text{halo}}^2}$
//! across pure baryonic (Keplerian falloff), Navarro-Frenk-White (`NfwCusp`), and
//! Burkert (`BurkertCore`) dark-matter halos.
//!
//! # Example
//! ```rust
//! # use domain_physics::__doctest_facade as math_explorer;
//! use math_explorer::physics::astrophysics::{
//!     HaloProfile, Mpc, TypeCode10, baryons_from_distance, decompose_rotation_curve,
//! };
//!
//! let baryons = baryons_from_distance(&TypeCode10, Mpc(12.0), 0.6, 0.8);
//! let halo = HaloProfile::BurkertCore {
//!     virial_mass_solar: 8.0e9,
//!     core_radius_kpc: 2.0,
//!     concentration: 12.0,
//! };
//! let sample = decompose_rotation_curve(4.0, &baryons, &halo);
//! assert!(sample.v_circ_kms > sample.v_disk_kms);
//! ```

// [cite:dwarf_galaxy_empirical_dependencies]

use super::galaxies::{
    GalaxyModel, GeneralIrregular, Magnitude, Mpc, Redshift, SolarMassLog, TypeCode10,
    TypeCode95To99,
};

/// Meters per kiloparsec ($1\,\text{kpc} = 3.085677581491367 \times 10^{19}\,\text{m}$).
pub const METERS_PER_KPC: f64 = 3.085_677_581_491_367e19;

/// Gravitational constant in galactic units ($\text{kpc}\,(\text{km/s})^2\,M_\odot^{-1}$).
pub const G_KPC_KMS2_PER_SOLAR_MASS: f64 =
    (math_commons::constants::G * math_commons::constants::SOLAR_MASS) / (METERS_PER_KPC * 1.0e6);

/// Minimum softened galactocentric radius in kpc preventing central division by zero.
pub const MIN_GALACTOCENTRIC_RADIUS_KPC: f64 = 1.0e-3;

/// Morphological classification of irregular dwarf galaxies from Tillaboev et al. (2025).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum DwarfMorphology {
    /// Complete sample of 413 irregular dwarf galaxies (`GeneralIrregular`).
    GeneralIrregular,
    /// Morphological Type Code 10 irregular dwarf galaxies (`TypeCode10`).
    #[default]
    TypeCode10,
    /// Morphological Type Code 9.5–9.9 irregular dwarf galaxies (`TypeCode95To99`).
    TypeCode95To99,
}

impl DwarfMorphology {
    /// Human-readable morphological classification label.
    #[must_use]
    #[verified_engine::verified]
    pub fn label(&self) -> &'static str {
        match self {
            Self::GeneralIrregular => "All Irregulars (General Group)",
            Self::TypeCode10 => "Type Code 10 Irregular",
            Self::TypeCode95To99 => "Type Code 9.5–9.9 Irregular",
        }
    }

    /// Evaluates $\log_{10}(M / M_\odot)$ from distance $D$ (Mpc) via `GalaxyModel`.
    #[must_use]
    #[verified_engine::verified]
    pub fn log_mass_from_distance(&self, distance: Mpc) -> SolarMassLog {
        match self {
            Self::GeneralIrregular => GeneralIrregular.log_mass_from_distance(distance),
            Self::TypeCode10 => TypeCode10.log_mass_from_distance(distance),
            Self::TypeCode95To99 => TypeCode95To99.log_mass_from_distance(distance),
        }
    }

    /// Evaluates B-band apparent magnitude from distance $D$ (Mpc), returning `None` when unestablished.
    #[must_use]
    #[verified_engine::verified]
    pub fn apparent_magnitude_from_distance(&self, distance: Mpc) -> Option<Magnitude> {
        match self {
            Self::GeneralIrregular => GeneralIrregular.apparent_magnitude_from_distance(distance),
            Self::TypeCode10 => TypeCode10.apparent_magnitude_from_distance(distance),
            Self::TypeCode95To99 => TypeCode95To99.apparent_magnitude_from_distance(distance),
        }
    }

    /// Evaluates $\log_{10}(M / M_\odot)$ from V-band absolute magnitude $M_V$ via `GalaxyModel`.
    #[must_use]
    #[verified_engine::verified]
    pub fn log_mass_from_absolute_magnitude(&self, abs_mag_v: Magnitude) -> SolarMassLog {
        match self {
            Self::GeneralIrregular => GeneralIrregular.log_mass_from_absolute_magnitude(abs_mag_v),
            Self::TypeCode10 => TypeCode10.log_mass_from_absolute_magnitude(abs_mag_v),
            Self::TypeCode95To99 => TypeCode95To99.log_mass_from_absolute_magnitude(abs_mag_v),
        }
    }

    /// Evaluates redshift $z$ from $\log_{10}(M / M_\odot)$ via `GalaxyModel`.
    #[must_use]
    #[verified_engine::verified]
    pub fn redshift_from_log_mass(&self, log_mass: SolarMassLog) -> Redshift {
        match self {
            Self::GeneralIrregular => GeneralIrregular.redshift_from_log_mass(log_mass),
            Self::TypeCode10 => TypeCode10.redshift_from_log_mass(log_mass),
            Self::TypeCode95To99 => TypeCode95To99.redshift_from_log_mass(log_mass),
        }
    }

    /// Builds [`BaryonicDiskConfig`] from distance $D$ (Mpc) for this morphology.
    #[must_use]
    #[verified_engine::verified]
    pub fn baryonic_config_from_distance(
        &self,
        distance: Mpc,
        gas_fraction: f64,
        disk_scale_kpc: f64,
    ) -> BaryonicDiskConfig {
        match self {
            Self::GeneralIrregular => {
                baryons_from_distance(&GeneralIrregular, distance, gas_fraction, disk_scale_kpc)
            }
            Self::TypeCode10 => {
                baryons_from_distance(&TypeCode10, distance, gas_fraction, disk_scale_kpc)
            }
            Self::TypeCode95To99 => {
                baryons_from_distance(&TypeCode95To99, distance, gas_fraction, disk_scale_kpc)
            }
        }
    }

    /// Builds [`BaryonicDiskConfig`] from V-band absolute magnitude $M_V$ for this morphology.
    #[must_use]
    #[verified_engine::verified]
    pub fn baryonic_config_from_magnitude(
        &self,
        abs_mag_v: Magnitude,
        gas_fraction: f64,
        disk_scale_kpc: f64,
    ) -> BaryonicDiskConfig {
        match self {
            Self::GeneralIrregular => baryons_from_absolute_magnitude(
                &GeneralIrregular,
                abs_mag_v,
                gas_fraction,
                disk_scale_kpc,
            ),
            Self::TypeCode10 => baryons_from_absolute_magnitude(
                &TypeCode10,
                abs_mag_v,
                gas_fraction,
                disk_scale_kpc,
            ),
            Self::TypeCode95To99 => baryons_from_absolute_magnitude(
                &TypeCode95To99,
                abs_mag_v,
                gas_fraction,
                disk_scale_kpc,
            ),
        }
    }
}

/// Summary of empirical Tillaboev (2025) observable predictions for a dwarf galaxy.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TillaboevScalingSummary {
    /// Logarithmic baryonic mass $\log_{10}(M / M_\odot)$.
    pub log_mass_solar: f64,
    /// Total baryonic mass $M_{\text{bary}} = 10^{\log_{10}(M / M_\odot)}$ in $M_\odot$.
    pub total_baryonic_mass_solar: f64,
    /// Predicted B-band apparent magnitude ($m_B$), or `None` when unestablished in the paper.
    pub apparent_magnitude_b: Option<f64>,
    /// Predicted redshift $z$ from $\log_{10}(M / M_\odot)$.
    pub redshift: f64,
}

/// Baryonic stellar-disk and neutral-HI gas parameters derived from a [`GalaxyModel`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BaryonicDiskConfig {
    /// Stellar disk mass $M_{\text{disk}}$ in solar masses ($M_\odot$).
    pub disk_mass_solar: f64,
    /// Neutral HI gas disk mass $M_{\text{gas}}$ in solar masses ($M_\odot$).
    pub gas_mass_solar: f64,
    /// Stellar Kuzmin disk radial scale length $R_d$ in kpc.
    pub disk_scale_kpc: f64,
    /// Extended HI gas disk radial scale length $R_g$ in kpc.
    pub gas_scale_kpc: f64,
    /// Empirical scaling values evaluated from Tillaboev (2025).
    pub scaling: TillaboevScalingSummary,
}

/// Dark-matter halo density profile governing outer galactic rotation curves.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum HaloProfile {
    /// Pure baryonic potential with outer Keplerian $v(r) \propto r^{-1/2}$ falloff.
    #[default]
    BaryonicOnly,
    /// Navarro-Frenk-White (1997) cold dark matter halo with inner $\rho \propto r^{-1}$ cusp.
    NfwCusp {
        /// Virial halo mass $M_{200}$ in solar masses ($M_\odot$).
        virial_mass_solar: f64,
        /// Characteristic scale radius $r_s$ in kpc.
        scale_radius_kpc: f64,
        /// Virial concentration $c = r_{200} / r_s$.
        concentration: f64,
    },
    /// Burkert (1995) cored dark-matter halo tailored to irregular dwarf galaxies.
    BurkertCore {
        /// Virial halo mass $M_{200}$ in solar masses ($M_\odot$).
        virial_mass_solar: f64,
        /// Constant-density core radius $r_0$ in kpc.
        core_radius_kpc: f64,
        /// Virial concentration $c = r_{200} / r_0$.
        concentration: f64,
    },
}

impl HaloProfile {
    /// Human-readable label for UI display and telemetry.
    #[must_use]
    #[verified_engine::verified]
    pub fn label(&self) -> &'static str {
        match self {
            Self::BaryonicOnly => "Baryonic Only (Keplerian)",
            Self::NfwCusp { .. } => "NFW Cusp",
            Self::BurkertCore { .. } => "Burkert Core",
        }
    }
}

/// Quadrature circular velocity decomposition at a galactocentric radius $r$.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct VelocityDecomposition {
    /// Effective galactocentric radius $r \ge 10^{-3}\,\text{kpc}$.
    pub radius_kpc: f64,
    /// Stellar disk circular speed $v_{\text{disk}}(r)$ in km/s.
    pub v_disk_kms: f64,
    /// Extended HI gas circular speed $v_{\text{gas}}(r)$ in km/s.
    pub v_gas_kms: f64,
    /// Dark-matter halo circular speed $v_{\text{halo}}(r)$ in km/s.
    pub v_halo_kms: f64,
    /// Total circular speed $v_{\text{circ}}(r) = \sqrt{v_{\text{disk}}^2 + v_{\text{gas}}^2 + v_{\text{halo}}^2}$ in km/s.
    pub v_circ_kms: f64,
}

/// Clamps galactocentric radius $r$ to at least [`MIN_GALACTOCENTRIC_RADIUS_KPC`] ($10^{-3}\,\text{kpc}$).
#[must_use]
#[verified_engine::verified]
pub fn clamp_radius_kpc(r_kpc: f64) -> f64 {
    if r_kpc.is_finite() && r_kpc > MIN_GALACTOCENTRIC_RADIUS_KPC {
        r_kpc
    } else {
        MIN_GALACTOCENTRIC_RADIUS_KPC
    }
}

#[verified_engine::verified]
fn build_baryons_from_log_mass(
    log_mass: SolarMassLog,
    apparent_mag_b: Option<Magnitude>,
    redshift: Redshift,
    gas_fraction: f64,
    disk_scale_kpc: f64,
) -> BaryonicDiskConfig {
    let clamped_log = log_mass.as_f64().clamp(-6.0, 15.0);
    let total_mass = 10.0_f64.powf(clamped_log);
    let f_gas = if gas_fraction.is_finite() {
        gas_fraction.clamp(0.0, 1.0)
    } else {
        0.5
    };
    let r_d = if disk_scale_kpc.is_finite() {
        disk_scale_kpc.max(0.05)
    } else {
        0.8
    };
    BaryonicDiskConfig {
        disk_mass_solar: (1.0 - f_gas) * total_mass,
        gas_mass_solar: f_gas * total_mass,
        disk_scale_kpc: r_d,
        gas_scale_kpc: 2.0 * r_d,
        scaling: TillaboevScalingSummary {
            log_mass_solar: log_mass.as_f64(),
            total_baryonic_mass_solar: total_mass,
            apparent_magnitude_b: apparent_mag_b.map(|m| m.as_f64()),
            redshift: redshift.as_f64(),
        },
    }
}

/// Derives stellar disk and neutral HI gas components from galaxy distance (`Mpc`) using a [`GalaxyModel`].
#[must_use]
#[verified_engine::verified]
pub fn baryons_from_distance<M: GalaxyModel>(
    model: &M,
    distance: Mpc,
    gas_fraction: f64,
    disk_scale_kpc: f64,
) -> BaryonicDiskConfig {
    let safe_d = Mpc(if distance.as_f64().is_finite() {
        distance.as_f64().max(0.01)
    } else {
        10.0
    });
    let log_mass = model.log_mass_from_distance(safe_d);
    let app_mag = model.apparent_magnitude_from_distance(safe_d);
    let redshift = model.redshift_from_log_mass(log_mass);
    build_baryons_from_log_mass(log_mass, app_mag, redshift, gas_fraction, disk_scale_kpc)
}

/// Derives stellar disk and neutral HI gas components from V-band absolute magnitude (`Magnitude`) using a [`GalaxyModel`].
#[must_use]
#[verified_engine::verified]
pub fn baryons_from_absolute_magnitude<M: GalaxyModel>(
    model: &M,
    abs_magnitude_v: Magnitude,
    gas_fraction: f64,
    disk_scale_kpc: f64,
) -> BaryonicDiskConfig {
    let safe_mag = Magnitude(if abs_magnitude_v.as_f64().is_finite() {
        abs_magnitude_v.as_f64().clamp(-25.0, 0.0)
    } else {
        -15.0
    });
    let log_mass = model.log_mass_from_absolute_magnitude(safe_mag);
    let redshift = model.redshift_from_log_mass(log_mass);
    build_baryons_from_log_mass(log_mass, None, redshift, gas_fraction, disk_scale_kpc)
}

#[verified_engine::verified]
fn kuzmin_v_sq(r_kpc: f64, mass_solar: f64, scale_kpc: f64) -> f64 {
    let m = mass_solar.max(0.0);
    let a = scale_kpc.max(MIN_GALACTOCENTRIC_RADIUS_KPC);
    let denom = (r_kpc * r_kpc + a * a).powf(1.5);
    (G_KPC_KMS2_PER_SOLAR_MASS * m * r_kpc * r_kpc / denom).max(0.0)
}

#[verified_engine::verified]
fn kuzmin_potential(r_kpc: f64, mass_solar: f64, scale_kpc: f64) -> f64 {
    let m = mass_solar.max(0.0);
    let a = scale_kpc.max(MIN_GALACTOCENTRIC_RADIUS_KPC);
    -G_KPC_KMS2_PER_SOLAR_MASS * m / (r_kpc * r_kpc + a * a).sqrt()
}

#[verified_engine::verified]
fn nfw_shape(x: f64) -> f64 {
    (1.0 + x).ln() - x / (1.0 + x)
}

#[verified_engine::verified]
fn burkert_shape(x: f64) -> f64 {
    0.5 * (1.0 + x * x).ln() + (1.0 + x).ln() - x.atan()
}

#[verified_engine::verified]
fn halo_v_sq(r_eff_kpc: f64, halo: &HaloProfile) -> f64 {
    match *halo {
        HaloProfile::BaryonicOnly => 0.0,
        HaloProfile::NfwCusp {
            virial_mass_solar,
            scale_radius_kpc,
            concentration,
        } => {
            let m200 = virial_mass_solar.max(0.0);
            let rs = scale_radius_kpc.max(MIN_GALACTOCENTRIC_RADIUS_KPC);
            let c = concentration.max(0.5);
            let m_scale = m200 / nfw_shape(c).max(1.0e-12);
            (G_KPC_KMS2_PER_SOLAR_MASS * m_scale * nfw_shape(r_eff_kpc / rs) / r_eff_kpc).max(0.0)
        }
        HaloProfile::BurkertCore {
            virial_mass_solar,
            core_radius_kpc,
            concentration,
        } => {
            let m200 = virial_mass_solar.max(0.0);
            let r0 = core_radius_kpc.max(MIN_GALACTOCENTRIC_RADIUS_KPC);
            let c = concentration.max(0.5);
            let m_scale = m200 / burkert_shape(c).max(1.0e-12);
            (G_KPC_KMS2_PER_SOLAR_MASS * m_scale * burkert_shape(r_eff_kpc / r0) / r_eff_kpc)
                .max(0.0)
        }
    }
}

#[verified_engine::verified]
fn halo_potential_at_eff_radius(r_eff_kpc: f64, halo: &HaloProfile) -> f64 {
    match *halo {
        HaloProfile::BaryonicOnly => 0.0,
        HaloProfile::NfwCusp {
            virial_mass_solar,
            scale_radius_kpc,
            concentration,
        } => {
            let m200 = virial_mass_solar.max(0.0);
            let rs = scale_radius_kpc.max(MIN_GALACTOCENTRIC_RADIUS_KPC);
            let c = concentration.max(0.5);
            let m_scale = m200 / nfw_shape(c).max(1.0e-12);
            -G_KPC_KMS2_PER_SOLAR_MASS * m_scale * (1.0 + r_eff_kpc / rs).ln() / r_eff_kpc
        }
        HaloProfile::BurkertCore {
            virial_mass_solar,
            core_radius_kpc,
            concentration,
        } => {
            let m200 = virial_mass_solar.max(0.0);
            let r0 = core_radius_kpc.max(MIN_GALACTOCENTRIC_RADIUS_KPC);
            let c = concentration.max(0.5);
            let m_scale = m200 / burkert_shape(c).max(1.0e-12);
            let x = r_eff_kpc / r0;
            let h = burkert_shape(x) / x + std::f64::consts::FRAC_PI_2 - x.atan() + (1.0 + x).ln()
                - 0.5 * (1.0 + x * x).ln();
            -G_KPC_KMS2_PER_SOLAR_MASS * m_scale * h / r0
        }
    }
}

/// Decomposes circular velocity $v_{\text{circ}}(r) = \sqrt{v_{\text{disk}}^2 + v_{\text{gas}}^2 + v_{\text{halo}}^2}$ at radius $r$ (kpc).
#[must_use]
#[verified_engine::verified]
pub fn decompose_rotation_curve(
    r_kpc: f64,
    baryons: &BaryonicDiskConfig,
    halo: &HaloProfile,
) -> VelocityDecomposition {
    let r_eff = clamp_radius_kpc(r_kpc);
    let v_disk_sq = kuzmin_v_sq(r_eff, baryons.disk_mass_solar, baryons.disk_scale_kpc);
    let v_gas_sq = kuzmin_v_sq(r_eff, baryons.gas_mass_solar, baryons.gas_scale_kpc);
    let v_halo_sq = halo_v_sq(r_eff, halo);
    let v_tot_sq = v_disk_sq + v_gas_sq + v_halo_sq;
    VelocityDecomposition {
        radius_kpc: r_eff,
        v_disk_kms: v_disk_sq.sqrt(),
        v_gas_kms: v_gas_sq.sqrt(),
        v_halo_kms: v_halo_sq.sqrt(),
        v_circ_kms: v_tot_sq.sqrt(),
    }
}

/// Evaluates the exact midplane gravitational potential $\Phi(r)$ in $(\text{km/s})^2$ matched to clamped radii.
#[must_use]
#[verified_engine::verified]
pub fn total_gravitational_potential(
    r_kpc: f64,
    baryons: &BaryonicDiskConfig,
    halo: &HaloProfile,
) -> f64 {
    let r_nonneg = if r_kpc.is_finite() {
        r_kpc.abs()
    } else {
        MIN_GALACTOCENTRIC_RADIUS_KPC
    };
    if r_nonneg >= MIN_GALACTOCENTRIC_RADIUS_KPC {
        kuzmin_potential(r_nonneg, baryons.disk_mass_solar, baryons.disk_scale_kpc)
            + kuzmin_potential(r_nonneg, baryons.gas_mass_solar, baryons.gas_scale_kpc)
            + halo_potential_at_eff_radius(r_nonneg, halo)
    } else {
        let r_min = MIN_GALACTOCENTRIC_RADIUS_KPC;
        let phi_min = kuzmin_potential(r_min, baryons.disk_mass_solar, baryons.disk_scale_kpc)
            + kuzmin_potential(r_min, baryons.gas_mass_solar, baryons.gas_scale_kpc)
            + halo_potential_at_eff_radius(r_min, halo);
        let v_min = decompose_rotation_curve(r_min, baryons, halo).v_circ_kms;
        let omega_sq = (v_min * v_min) / (r_min * r_min);
        phi_min - 0.5 * omega_sq * (r_min * r_min - r_nonneg * r_nonneg)
    }
}

/// Fills a caller-allocated slice with evenly spaced [`VelocityDecomposition`] samples over `[r_min_kpc, r_max_kpc]`.
#[verified_engine::verified]
pub fn sample_rotation_curve_into(
    r_min_kpc: f64,
    r_max_kpc: f64,
    baryons: &BaryonicDiskConfig,
    halo: &HaloProfile,
    out: &mut [VelocityDecomposition],
) {
    let n = out.len();
    if n == 0 {
        return;
    }
    let start = clamp_radius_kpc(r_min_kpc);
    let end = clamp_radius_kpc(r_max_kpc).max(start);
    let denom = if n > 1 { (n - 1) as f64 } else { 1.0 };
    let mut i = 0;
    while i < n {
        let frac = if n > 1 { i as f64 / denom } else { 0.0 };
        let r = start + frac * (end - start);
        out[i] = decompose_rotation_curve(r, baryons, halo);
        i += 1;
    }
}
