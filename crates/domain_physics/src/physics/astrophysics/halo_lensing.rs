//! # Dark-Matter Halo & Baryonic Disk Gravitational Lensing
//!
//! Closed-form surface mass convergence $\kappa(R)$, mean convergence $\bar{\kappa}(R)$,
//! tangential shear $\gamma(R)$, $2 \times 2$ magnification Jacobians, and critical/caustic
//! radii comparing Navarro-Frenk-White (`NfwCusp`) against Burkert (`BurkertCore`) halos.

// [cite:dwarf_galaxy_empirical_dependencies]
// [cite:general_relativity_geodesics]

use std::f64::consts::{FRAC_PI_2, PI};

use nalgebra::{Matrix2, Vector2};

use super::lensing::{
    JACOBIAN_EIGEN_FLOOR, ThinLensSample, axisymmetric_lens_jacobian, clamp_away_from_zero,
};
use super::rotation_curves::{
    BaryonicDiskConfig, HaloProfile, MIN_GALACTOCENTRIC_RADIUS_KPC, clamp_radius_kpc,
};

#[verified_engine::verified]
fn nfw_lens_functions(x: f64) -> (f64, f64) {
    let xs = x.max(1.0e-6);
    let dx = xs - 1.0;
    if dx.abs() < 1.0e-4 {
        let f = (1.0 / 3.0) - 0.4 * dx + (3.0 / 7.0) * dx * dx;
        let g = (1.0 - 2.0_f64.ln()) + (1.0 / 3.0) * dx - (1.0 / 30.0) * dx * dx;
        return (f, g);
    }
    if xs < 1.0 {
        let root_term = ((1.0 - xs) / (1.0 + xs)).sqrt();
        let inv_s = 1.0 / (1.0 - xs * xs).sqrt();
        let arch = root_term.atanh();
        let f = (1.0 - 2.0 * inv_s * arch) / (xs * xs - 1.0);
        let g = (0.5 * xs).ln() + 2.0 * inv_s * arch;
        (f, g)
    } else {
        let root_term = ((xs - 1.0) / (xs + 1.0)).sqrt();
        let inv_s = 1.0 / (xs * xs - 1.0).sqrt();
        let ang = root_term.atan();
        let f = (1.0 - 2.0 * inv_s * ang) / (xs * xs - 1.0);
        let g = (0.5 * xs).ln() + 2.0 * inv_s * ang;
        (f, g)
    }
}

#[verified_engine::verified]
fn burkert_lens_functions(x: f64) -> (f64, f64) {
    let xs = x.max(0.0);
    if xs < 1.0e-3 {
        let kappa_unit = FRAC_PI_2;
        return (kappa_unit, kappa_unit);
    }
    let s_plus = (1.0 + xs * xs).sqrt();
    let i1 = FRAC_PI_2 / s_plus;
    let asinh_inv = (1.0 / xs).asinh();
    let i2 = asinh_inv / s_plus;
    let dx = xs - 1.0;
    let (i3, h3) = if dx.abs() < 1.0e-4 {
        let i3_ser = 1.0 - (2.0 / 3.0) * dx + (8.0 / 15.0) * dx * dx;
        let h3_ser = -2.0 * dx + (1.0 / 3.0) * dx * dx;
        (i3_ser, h3_ser)
    } else if xs < 1.0 {
        let s_minus = (1.0 - xs * xs).sqrt();
        let acosh_inv = (1.0 / xs).acosh();
        (acosh_inv / s_minus, s_minus * acosh_inv)
    } else {
        let s_minus = (xs * xs - 1.0).sqrt();
        let acos_inv = (1.0 / xs).acos();
        (acos_inv / s_minus, -s_minus * acos_inv)
    };
    let kappa_over_kb = (i1 + i2 - i3).max(0.0);
    let g_burk =
        (FRAC_PI_2 * (s_plus - 1.0) + s_plus * asinh_inv + 2.0 * (0.5 * xs).ln() + h3).max(0.0);
    let mean_kappa_over_kb = (2.0 * g_burk) / (xs * xs);
    (kappa_over_kb, mean_kappa_over_kb)
}

#[verified_engine::verified]
fn kuzmin_convergence_and_mean(
    r_kpc: f64,
    mass_solar: f64,
    scale_kpc: f64,
    sigma_crit: f64,
) -> (f64, f64) {
    let m = mass_solar.max(0.0);
    if m <= 0.0 {
        return (0.0, 0.0);
    }
    let a = scale_kpc.max(MIN_GALACTOCENTRIC_RADIUS_KPC);
    let s = (r_kpc * r_kpc + a * a).sqrt();
    let sigma = (a * m) / (2.0 * PI * s * s * s);
    let m_cyl = m * (1.0 - a / s);
    let mean_sigma = m_cyl / (PI * r_kpc * r_kpc);
    (sigma / sigma_crit, mean_sigma / sigma_crit)
}

/// Evaluates convergence $\kappa$, mean convergence $\bar{\kappa}$, shear $\gamma$, deflection,
/// and inverse-magnification eigenvalues for a [`HaloProfile`] (and optional [`BaryonicDiskConfig`]).
#[must_use]
#[verified_engine::verified]
pub fn halo_thin_lens_sample(
    r_kpc: f64,
    baryons: Option<&BaryonicDiskConfig>,
    halo: &HaloProfile,
    sigma_crit_solar_per_kpc2: f64,
) -> ThinLensSample {
    let r = clamp_radius_kpc(r_kpc);
    let sigma_crit = if sigma_crit_solar_per_kpc2.is_finite() {
        sigma_crit_solar_per_kpc2.max(1.0e3)
    } else {
        1.0e8
    };
    let (mut kappa, mut mean_kappa) = match *halo {
        HaloProfile::BaryonicOnly => (0.0, 0.0),
        HaloProfile::NfwCusp {
            virial_mass_solar,
            scale_radius_kpc,
            concentration,
        } => {
            let rs = scale_radius_kpc.max(MIN_GALACTOCENTRIC_RADIUS_KPC);
            let c = concentration.max(0.5);
            let nfw_c = ((1.0 + c).ln() - c / (1.0 + c)).max(1.0e-12);
            let m_scale = virial_mass_solar.max(0.0) / nfw_c;
            let kappa_s = m_scale / (4.0 * PI * rs * rs * sigma_crit);
            let x = r / rs;
            let (f, g) = nfw_lens_functions(x);
            (2.0 * kappa_s * f, (4.0 * kappa_s * g) / (x * x))
        }
        HaloProfile::BurkertCore {
            virial_mass_solar,
            core_radius_kpc,
            concentration,
        } => {
            let r0 = core_radius_kpc.max(MIN_GALACTOCENTRIC_RADIUS_KPC);
            let c = concentration.max(0.5);
            let burk_c = (0.5 * (1.0 + c * c).ln() + (1.0 + c).ln() - c.atan()).max(1.0e-12);
            let m_scale = virial_mass_solar.max(0.0) / burk_c;
            let kappa_b = m_scale / (2.0 * PI * r0 * r0 * sigma_crit);
            let (ku, mku) = burkert_lens_functions(r / r0);
            (kappa_b * ku, kappa_b * mku)
        }
    };
    if let Some(b) = baryons {
        let (kd, mkd) =
            kuzmin_convergence_and_mean(r, b.disk_mass_solar, b.disk_scale_kpc, sigma_crit);
        let (kg, mkg) =
            kuzmin_convergence_and_mean(r, b.gas_mass_solar, b.gas_scale_kpc, sigma_crit);
        kappa += kd + kg;
        mean_kappa += mkd + mkg;
    }
    let shear = (mean_kappa - kappa).max(0.0);
    let lambda_t = 1.0 - mean_kappa;
    let lambda_r = 1.0 - 2.0 * kappa + mean_kappa;
    let denom = clamp_away_from_zero(lambda_t, JACOBIAN_EIGEN_FLOOR)
        * clamp_away_from_zero(lambda_r, JACOBIAN_EIGEN_FLOOR);
    ThinLensSample {
        radius: r,
        deflection: r * mean_kappa,
        convergence: kappa,
        mean_convergence: mean_kappa,
        shear,
        lambda_tangential: lambda_t,
        lambda_radial: lambda_r,
        magnification: (1.0 / denom).clamp(-64.0, 64.0),
    }
}

/// Evaluates vector deflection $\boldsymbol{\alpha}(\mathbf{x})$ and $2 \times 2$ Jacobian $J_{\text{lens}}$ for a halo lens.
#[must_use]
#[verified_engine::verified]
pub fn halo_lens_deflection_and_jacobian(
    offset_kpc: Vector2<f64>,
    baryons: Option<&BaryonicDiskConfig>,
    halo: &HaloProfile,
    sigma_crit_solar_per_kpc2: f64,
) -> (Vector2<f64>, Matrix2<f64>, ThinLensSample) {
    let ox = if offset_kpc.x.is_finite() {
        offset_kpc.x
    } else {
        0.0
    };
    let oy = if offset_kpc.y.is_finite() {
        offset_kpc.y
    } else {
        0.0
    };
    let r = (ox * ox + oy * oy).sqrt();
    let sample = halo_thin_lens_sample(r, baryons, halo, sigma_crit_solar_per_kpc2);
    let dir = if r > 1.0e-9 {
        Vector2::new(ox / r, oy / r)
    } else {
        Vector2::zeros()
    };
    let deflection_vec = dir * sample.deflection;
    let jac = axisymmetric_lens_jacobian(
        Vector2::new(ox, oy),
        sample.lambda_radial,
        sample.lambda_tangential,
    );
    (deflection_vec, jac, sample)
}

/// Locates the outer tangential critical radius $R_{\text{crit}}$ where $\lambda_t(R) = 1 - \bar{\kappa}(R) = 0$.
#[must_use]
#[verified_engine::verified]
pub fn tangential_critical_radius_kpc(
    baryons: Option<&BaryonicDiskConfig>,
    halo: &HaloProfile,
    sigma_crit_solar_per_kpc2: f64,
    r_max_kpc: f64,
) -> Option<f64> {
    let mut lo = MIN_GALACTOCENTRIC_RADIUS_KPC;
    let mut hi = clamp_radius_kpc(r_max_kpc).max(0.1);
    let f_lo =
        halo_thin_lens_sample(lo, baryons, halo, sigma_crit_solar_per_kpc2).lambda_tangential;
    if f_lo > 0.0 {
        return None;
    }
    let mut f_hi =
        halo_thin_lens_sample(hi, baryons, halo, sigma_crit_solar_per_kpc2).lambda_tangential;
    let mut expand = 0;
    while f_hi <= 0.0 && expand < 20 {
        lo = hi;
        hi *= 2.0;
        f_hi =
            halo_thin_lens_sample(hi, baryons, halo, sigma_crit_solar_per_kpc2).lambda_tangential;
        expand += 1;
    }
    if f_hi <= 0.0 {
        return None;
    }
    let mut iter = 0;
    while iter < 40 {
        let mid = 0.5 * (lo + hi);
        let f_mid =
            halo_thin_lens_sample(mid, baryons, halo, sigma_crit_solar_per_kpc2).lambda_tangential;
        if f_mid <= 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
        iter += 1;
    }
    Some(0.5 * (lo + hi))
}

/// Fills `out` with evenly spaced [`ThinLensSample`] evaluations over `[r_min_kpc, r_max_kpc]` with zero heap allocations.
#[verified_engine::verified]
pub fn sample_halo_lensing_into(
    r_min_kpc: f64,
    r_max_kpc: f64,
    baryons: Option<&BaryonicDiskConfig>,
    halo: &HaloProfile,
    sigma_crit_solar_per_kpc2: f64,
    out: &mut [ThinLensSample],
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
        out[i] = halo_thin_lens_sample(r, baryons, halo, sigma_crit_solar_per_kpc2);
        i += 1;
    }
}
