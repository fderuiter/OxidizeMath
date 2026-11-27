//! 20-point Genz (2004) Gauss-Legendre bivariate normal and Gaussian copula solver.

use super::core::{Correlation, Probability};
use super::transforms::{fast_norm_cdf, fast_norm_inv};
use crate::error::CopulaError;
use std::f64::consts::{PI, TAU};

/// Correlation split threshold separating angular Plackett quadrature from Genz tail quadrature.
pub const RHO_SPLIT: f64 = 0.925;

/// Threshold on `|a - b| / sqrt(1 - rho^2)` separating boundary-layer subtraction from smooth GL.
const ASYMPTOTIC_LAMBDA_CUTOFF: f64 = 3.5;

/// Positive abscissas and weights of the 20-point Gauss-Legendre rule on `[-1, 1]`.
const GL20_HALF_NODES_WEIGHTS: [(f64, f64); 10] = [
    (0.076_526_521_133_497_34, 0.152_753_387_130_725_85),
    (0.227_785_851_141_645_08, 0.149_172_986_472_603_75),
    (0.373_706_088_715_419_56, 0.142_096_109_318_382_05),
    (0.510_867_001_950_827_1, 0.131_688_638_449_176_63),
    (0.636_053_680_726_515, 0.118_194_531_961_518_42),
    (0.746_331_906_460_150_8, 0.101_930_119_817_240_44),
    (0.839_116_971_822_218_8, 0.083_276_741_576_704_75),
    (0.912_234_428_251_326, 0.062_672_048_334_109_06),
    (0.963_971_927_277_913_8, 0.040_601_429_800_386_94),
    (0.993_128_599_185_094_9, 0.017_614_007_139_152_12),
];

fn integrate_plackett_moderate(z1: f64, z2: f64, rho: f64) -> f64 {
    let hs = 0.5 * (z1 * z1 + z2 * z2);
    let hk = z1 * z2;
    let half_theta = 0.5 * rho.asin();
    let mut acc = 0.0;

    for &(node, weight) in &GL20_HALF_NODES_WEIGHTS {
        for sign in [-1.0, 1.0] {
            let theta = half_theta * (1.0 + sign * node);
            let sn = theta.sin();
            let cos_sq = (1.0 - sn) * (1.0 + sn);
            let exponent = (sn * hk - hs) / cos_sq;
            if exponent > -700.0 {
                acc += weight * exponent.exp();
            }
        }
    }
    fast_norm_cdf(z1) * fast_norm_cdf(z2) + (acc * half_theta) / TAU
}

fn high_corr_tail_asymptotic(a: f64, b: f64, as_sq: f64, width: f64) -> f64 {
    let hk = a * b;
    let diff_sq = (a - b) * (a - b);
    let beta = (a - b).abs();
    let lambda = beta / width;
    let c1 = (4.0 - hk) * 0.125;
    let c2 = c1 * (12.0 - hk) * 0.0625;
    let asr = -0.5 * (diff_sq / as_sq + hk);
    let exp_asr = if asr > -700.0 { asr.exp() } else { 0.0 };
    let scaled_mill = TAU.sqrt() * (0.5 * lambda * lambda).exp() * fast_norm_cdf(-lambda);
    let j0 = width - beta * scaled_mill;
    let j2 = (width * as_sq - diff_sq * j0) / 3.0;
    let j4 = (width * as_sq * as_sq - diff_sq * j2) * 0.2;
    let asymp = exp_asr * (j0 + c1 * j2 + c2 * j4);

    let half_w = 0.5 * width;
    let mut rem = 0.0;
    for &(node, weight) in &GL20_HALF_NODES_WEIGHTS {
        for sign in [-1.0, 1.0] {
            let x = half_w * (1.0 + sign * node);
            let x_sq = x * x;
            let arg0 = -0.5 * (diff_sq / x_sq + hk);
            if arg0 > -700.0 {
                let sp = ((1.0 - x) * (1.0 + x)).sqrt();
                let denom = 1.0 + sp;
                let exact_kernel = (-hk * x_sq / (2.0 * denom * denom)).exp() / sp;
                let poly_kernel = 1.0 + c1 * x_sq + c2 * x_sq * x_sq;
                rem += weight * arg0.exp() * (exact_kernel - poly_kernel);
            }
        }
    }
    (asymp + half_w * rem) / TAU
}

fn high_corr_tail_smooth(a: f64, b: f64, width: f64) -> f64 {
    let hk = a * b;
    let diff_sq = (a - b) * (a - b);
    let half_w = 0.5 * width;
    let mut acc = 0.0;

    for &(node, weight) in &GL20_HALF_NODES_WEIGHTS {
        for sign in [-1.0, 1.0] {
            let x = half_w * (1.0 + sign * node);
            let sp = ((1.0 - x) * (1.0 + x)).sqrt();
            let exponent = -diff_sq / (2.0 * x * x) - hk / (1.0 + sp);
            if exponent > -700.0 {
                acc += weight * exponent.exp() / sp;
            }
        }
    }
    (half_w * acc) / (2.0 * PI)
}

fn integrate_genz_high_corr(z1: f64, z2: f64, rho: f64) -> f64 {
    let abs_rho = rho.abs();
    let a = z1;
    let b = if rho > 0.0 { z2 } else { -z2 };
    let as_sq = (1.0 - abs_rho) * (1.0 + abs_rho);
    let width = as_sq.sqrt();
    let lambda = (a - b).abs() / width;

    let tail = if lambda <= ASYMPTOTIC_LAMBDA_CUTOFF {
        high_corr_tail_asymptotic(a, b, as_sq, width)
    } else {
        high_corr_tail_smooth(a, b, width)
    };

    if rho > 0.0 {
        fast_norm_cdf(z1.min(z2)) - tail
    } else {
        let base = if z1 > 0.0 && -z2 > 0.0 {
            fast_norm_cdf(z2) - fast_norm_cdf(-z1)
        } else {
            fast_norm_cdf(z1) - fast_norm_cdf(-z2)
        };
        base.max(0.0) + tail
    }
}

/// Evaluates the bivariate standard normal CDF `Phi_2(z1, z2; rho)` using Genz's 20-point
/// Gauss-Legendre algorithm split at `|rho| = 0.925`.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::copula::bivariate_normal_cdf;
///
/// let p = bivariate_normal_cdf(0.0, 0.0, 0.5).unwrap();
/// assert!((p - (1.0 / 3.0)).abs() < 1e-15);
/// ```
#[verified_engine::verified]
pub fn bivariate_normal_cdf(z1: f64, z2: f64, rho: f64) -> Result<f64, CopulaError> {
    if !rho.is_finite() || !(-1.0..=1.0).contains(&rho) {
        return Err(CopulaError::InvalidCorrelation { value: rho });
    }
    if z1.is_nan() || z2.is_nan() {
        return Err(CopulaError::InvalidProbability { value: f64::NAN });
    }
    if z1 == f64::NEG_INFINITY || z2 == f64::NEG_INFINITY {
        return Ok(0.0);
    }
    if z1 == f64::INFINITY {
        return Ok(fast_norm_cdf(z2));
    }
    if z2 == f64::INFINITY {
        return Ok(fast_norm_cdf(z1));
    }

    let u1 = fast_norm_cdf(z1);
    let u2 = fast_norm_cdf(z2);
    let frechet_lo = (u1 + u2 - 1.0).max(0.0);
    let frechet_hi = u1.min(u2);

    if rho == 0.0 {
        return Ok((u1 * u2).clamp(frechet_lo, frechet_hi));
    }
    if rho >= 1.0 {
        return Ok(frechet_hi);
    }
    if rho <= -1.0 {
        return Ok(frechet_lo);
    }

    let raw = if rho.abs() < RHO_SPLIT {
        integrate_plackett_moderate(z1, z2, rho)
    } else {
        integrate_genz_high_corr(z1, z2, rho)
    };
    Ok(raw.clamp(frechet_lo, frechet_hi))
}

/// Evaluates the bivariate Gaussian copula `C(u1, u2; rho)` with exact boundary short-circuiting
/// and Fréchet-Hoeffding bound guarantees.
#[verified_engine::verified]
pub fn bivariate_copula_cdf(
    u1: Probability,
    u2: Probability,
    rho: Correlation,
) -> Result<Probability, CopulaError> {
    let p1 = u1.value();
    let p2 = u2.value();
    let r = rho.value();

    if p1 == 0.0 || p2 == 0.0 {
        return Probability::new(0.0);
    }
    if p1 == 1.0 {
        return Ok(u2);
    }
    if p2 == 1.0 {
        return Ok(u1);
    }

    let frechet_lo = (p1 + p2 - 1.0).max(0.0);
    let frechet_hi = p1.min(p2);
    if r >= 1.0 {
        return Probability::new(frechet_hi);
    }
    if r <= -1.0 {
        return Probability::new(frechet_lo);
    }
    if r == 0.0 {
        return Probability::new((p1 * p2).clamp(frechet_lo, frechet_hi));
    }

    let z1 = fast_norm_inv(p1);
    let z2 = fast_norm_inv(p2);
    let raw = if r.abs() < RHO_SPLIT {
        integrate_plackett_moderate(z1, z2, r)
    } else {
        integrate_genz_high_corr(z1, z2, r)
    };
    Probability::new(raw.clamp(frechet_lo, frechet_hi))
}
