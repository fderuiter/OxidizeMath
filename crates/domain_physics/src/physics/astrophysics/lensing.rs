//! # Closed-Form Thin-Lens Deflections & $2 \times 2$ Shear Jacobians
//!
//! Computes zero-allocation thin-lens deflection angles, convergence $\kappa$, tangential shear
//! $\gamma$, $2 \times 2$ magnification Jacobians $J_{\text{lens}} = \partial\boldsymbol{\theta}/\partial\boldsymbol{\beta}$,
//! dual Einstein-ring images with $1.5 R_s$ photon-sphere shadow masking for
//! [`SchwarzschildBlackHole`], and caustic radii comparing [`super::rotation_curves::HaloProfile::NfwCusp`] against
//! [`super::rotation_curves::HaloProfile::BurkertCore`].
//!
//! # Example
//! ```rust
//! # use domain_physics::__doctest_facade as math_explorer;
//! use math_explorer::physics::astrophysics::{HaloProfile, halo_thin_lens_sample};
//! use math_explorer::scalars::Kiloparsecs;
//!
//! let nfw = HaloProfile::NfwCusp {
//!     virial_mass_solar: 2.0e11,
//!     scale_radius_kpc: Kiloparsecs::new(4.0),
//!     concentration: 12.0,
//! };
//! let sample = halo_thin_lens_sample(Kiloparsecs::new(1.0), None, &nfw, 1.0e8);
//! assert!(sample.convergence > 0.0 && sample.shear > 0.0);
//! ```

// [cite:dwarf_galaxy_empirical_dependencies]
// [cite:general_relativity_geodesics]

use nalgebra::{Matrix2, Vector2};

use crate::error::HighEnergyError;
use crate::physics::high_energy::general_relativity::SchwarzschildBlackHole;

/// Minimum relative angular separation $|\theta - \theta_E| / \theta_E$ near the Einstein ring.
pub const EINSTEIN_SEPARATION_REL_FLOOR: f64 = 1.0e-3;

/// Minimum absolute inverse-magnification eigenvalue preventing singular Jacobian inversion.
pub const JACOBIAN_EIGEN_FLOOR: f64 = 1.0e-3;

/// Axisymmetric thin-lens optical scalars evaluated at projected radius $R$ or angle $\theta$.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct ThinLensSample {
    /// Clamped radial coordinate ($R$ in kpc or $\theta$ in viewport/angular units).
    pub radius: f64,
    /// Radial deflection magnitude $\alpha(R) = R \bar{\kappa}(R)$.
    pub deflection: f64,
    /// Dimensionless surface mass convergence $\kappa(R) = \Sigma(R) / \Sigma_{\text{crit}}$.
    pub convergence: f64,
    /// Mean convergence inside radius $R$, $\bar{\kappa}(R) = \bar{\Sigma}(<R) / \Sigma_{\text{crit}}$.
    pub mean_convergence: f64,
    /// Tangential shear $\gamma(R) = \bar{\kappa}(R) - \kappa(R)$.
    pub shear: f64,
    /// Inverse tangential magnification eigenvalue $\lambda_t = 1 - \bar{\kappa}(R)$.
    pub lambda_tangential: f64,
    /// Inverse radial magnification eigenvalue $\lambda_r = 1 - \alpha'(R) = 1 - 2\kappa(R) + \bar{\kappa}(R)$.
    pub lambda_radial: f64,
    /// Signed scalar magnification $\mu = (\lambda_t \lambda_r)^{-1}$ with caustic clamping.
    pub magnification: f64,
}

/// Single lensed image state produced by a Schwarzschild point-mass thin lens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchwarzschildLensImage {
    /// Image-plane offset $\boldsymbol{\theta}_{\pm}$ relative to the black hole center.
    pub offset: Vector2<f64>,
    /// Image-plane $2 \times 2$ Jacobian $J_{\text{lens}} = \partial\boldsymbol{\theta}/\partial\boldsymbol{\beta}$.
    pub jacobian: Matrix2<f64>,
    /// Unsigned magnification factor $|\mu_{\pm}|$ clamped near the Einstein ring.
    pub magnification_abs: f64,
}

/// Dual-image thin-lens solution and $1.5 R_s$ shadow horizon metadata for [`SchwarzschildBlackHole`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchwarzschildDualImages {
    /// Effective Einstein ring radius $\theta_E > 0$.
    pub einstein_radius: f64,
    /// Unstable photon-sphere shadow radius $r_{\text{ph}} = 1.5 R_s$ in the same coordinate units.
    pub shadow_radius: f64,
    /// Positive-parity outer image $\boldsymbol{\theta}_+$ (`None` if captured inside $1.5 R_s$).
    pub primary: Option<SchwarzschildLensImage>,
    /// Negative-parity inner Einstein-ring counterpart $\boldsymbol{\theta}_-$ (`None` if inside $1.5 R_s$).
    pub secondary: Option<SchwarzschildLensImage>,
}

/// Clamps angular radius `theta` away from `theta_einstein` so $|\theta - \theta_E| \ge 10^{-3}\theta_E$.
#[must_use]
#[verified_engine::verified]
pub fn clamp_einstein_separation(theta: f64, theta_einstein: f64) -> f64 {
    let te = if theta_einstein.is_finite() {
        theta_einstein.max(1.0e-9)
    } else {
        1.0e-3
    };
    let t = if theta.is_finite() {
        theta.max(1.0e-9)
    } else {
        te * (1.0 + EINSTEIN_SEPARATION_REL_FLOOR)
    };
    let floor = EINSTEIN_SEPARATION_REL_FLOOR * te;
    let delta = t - te;
    if delta.abs() < floor {
        if delta < 0.0 {
            (te - floor).max(1.0e-9)
        } else {
            te + floor
        }
    } else {
        t
    }
}

#[verified_engine::verified]
pub(super) fn clamp_away_from_zero(val: f64, floor: f64) -> f64 {
    if !val.is_finite() {
        return floor;
    }
    if val.abs() < floor {
        if val < 0.0 { -floor } else { floor }
    } else {
        val
    }
}

/// Constructs the $2 \times 2$ magnification Jacobian $J_{\text{lens}} = \mu_r \hat{\mathbf{u}}_r\hat{\mathbf{u}}_r^T + \mu_t \hat{\mathbf{u}}_t\hat{\mathbf{u}}_t^T$.
#[must_use]
#[verified_engine::verified]
pub fn axisymmetric_lens_jacobian(
    offset: Vector2<f64>,
    lambda_radial: f64,
    lambda_tangential: f64,
) -> Matrix2<f64> {
    let mu_r = (1.0 / clamp_away_from_zero(lambda_radial, JACOBIAN_EIGEN_FLOOR)).clamp(-64.0, 64.0);
    let mu_t =
        (1.0 / clamp_away_from_zero(lambda_tangential, JACOBIAN_EIGEN_FLOOR)).clamp(-64.0, 64.0);
    let r_sq = offset.x * offset.x + offset.y * offset.y;
    if r_sq <= 1.0e-18 || !r_sq.is_finite() {
        return Matrix2::new(mu_r, 0.0, 0.0, mu_t);
    }
    let inv_r2 = 1.0 / r_sq;
    let ux2 = offset.x * offset.x * inv_r2;
    let uy2 = offset.y * offset.y * inv_r2;
    let uxy = offset.x * offset.y * inv_r2;
    Matrix2::new(
        mu_r * ux2 + mu_t * uy2,
        (mu_r - mu_t) * uxy,
        (mu_r - mu_t) * uxy,
        mu_r * uy2 + mu_t * ux2,
    )
}

/// Computes the weak-field Schwarzschild bending angle $\hat{\alpha}(b) = \frac{2 R_s}{b}$ outside $1.5 R_s$.
///
/// # Errors
/// Returns [`HighEnergyError`] if the black hole mass is invalid or `impact_parameter_m` is non-finite.
#[verified_engine::verified]
pub fn schwarzschild_deflection_angle(
    bh: &SchwarzschildBlackHole,
    impact_parameter_m: f64,
) -> Result<f64, HighEnergyError> {
    let rs = bh.schwarzschild_radius()?;
    let r_ph = bh.photon_sphere_radius()?;
    if !impact_parameter_m.is_finite() || impact_parameter_m <= 0.0 {
        return Err(HighEnergyError::InvalidRadius {
            radius: impact_parameter_m,
            limit: r_ph,
        });
    }
    let b_safe = impact_parameter_m.max(r_ph);
    Ok((2.0 * rs) / b_safe)
}

#[verified_engine::verified]
fn build_schwarzschild_image(
    dir: Vector2<f64>,
    theta_signed: f64,
    theta_e: f64,
    shadow_radius: f64,
) -> Option<SchwarzschildLensImage> {
    let theta_mag = theta_signed.abs();
    if theta_mag < shadow_radius {
        return None;
    }
    let theta_safe = clamp_einstein_separation(theta_mag, theta_e);
    let ratio_sq = (theta_e * theta_e) / (theta_safe * theta_safe);
    let lambda_t = 1.0 - ratio_sq;
    let lambda_r = 1.0 + ratio_sq;
    let sign = if theta_signed < 0.0 { -1.0 } else { 1.0 };
    let offset = dir * (sign * theta_safe);
    let jacobian = axisymmetric_lens_jacobian(offset, lambda_r, lambda_t);
    let mag = (1.0
        / (clamp_away_from_zero(lambda_t, JACOBIAN_EIGEN_FLOOR)
            * clamp_away_from_zero(lambda_r, JACOBIAN_EIGEN_FLOOR)))
    .abs()
    .clamp(0.0, 64.0);
    Some(SchwarzschildLensImage {
        offset,
        jacobian,
        magnification_abs: mag,
    })
}

/// Evaluates dual Einstein-ring thin-lens images and $2 \times 2$ shear Jacobians for a
/// [`SchwarzschildBlackHole`], masking any ray inside `photon_sphere_radius` ($1.5 R_s$).
///
/// # Errors
/// Propagates [`HighEnergyError`] if `bh` has an invalid mass.
#[verified_engine::verified]
pub fn schwarzschild_thin_lens_images(
    bh: &SchwarzschildBlackHole,
    source_offset: Vector2<f64>,
    einstein_radius: f64,
    viewport_rs: f64,
) -> Result<SchwarzschildDualImages, HighEnergyError> {
    let rs_phys = bh.schwarzschild_radius()?;
    let r_ph_phys = bh.photon_sphere_radius()?;
    let shadow_ratio = r_ph_phys / rs_phys;
    let rs_view = if viewport_rs.is_finite() {
        viewport_rs.max(0.0)
    } else {
        0.0
    };
    let shadow_radius = shadow_ratio * rs_view;
    let theta_e = if einstein_radius.is_finite() {
        einstein_radius.max(1.0e-5)
    } else {
        0.25
    };
    let bx = if source_offset.x.is_finite() {
        source_offset.x
    } else {
        0.0
    };
    let by = if source_offset.y.is_finite() {
        source_offset.y
    } else {
        0.0
    };
    let beta = (bx * bx + by * by).sqrt();
    let dir = if beta > 1.0e-9 {
        Vector2::new(bx / beta, by / beta)
    } else {
        Vector2::new(1.0, 0.0)
    };
    let disc = (beta * beta + 4.0 * theta_e * theta_e).sqrt();
    let theta_plus = 0.5 * (beta + disc);
    let theta_minus = 0.5 * (beta - disc);

    Ok(SchwarzschildDualImages {
        einstein_radius: theta_e,
        shadow_radius,
        primary: build_schwarzschild_image(dir, theta_plus, theta_e, shadow_radius),
        secondary: build_schwarzschild_image(dir, theta_minus, theta_e, shadow_radius),
    })
}

pub use super::halo_lensing::{
    halo_lens_deflection_and_jacobian, halo_thin_lens_sample, sample_halo_lensing_into,
    tangential_critical_radius_kpc,
};
