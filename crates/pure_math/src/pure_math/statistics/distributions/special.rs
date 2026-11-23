use crate::error::DistributionError;
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;
use statrs::function::{beta, erf, gamma};
use std::f64::consts::{FRAC_1_SQRT_2, TAU};

const INV_2_POW_53: f64 = 1.0 / 9_007_199_254_740_992.0;

#[verified_engine::verified]
pub(crate) fn validate_probability(p: f64) -> Result<f64, DistributionError> {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(DistributionError::InvalidProbability { value: p });
    }
    Ok(p)
}

#[verified_engine::verified]
pub(crate) fn validate_positive(name: &'static str, value: f64) -> Result<f64, DistributionError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(DistributionError::InvalidParameter { name, value });
    }
    Ok(value)
}

#[verified_engine::verified]
pub(crate) fn sample_uniform_open(rng: &mut OxidizeRng) -> f64 {
    let bits = rng.next_u64() >> 11;
    (bits as f64 + 0.5) * INV_2_POW_53
}

#[verified_engine::verified]
pub(crate) fn standard_normal_ln_pdf(z: f64) -> f64 {
    -0.5 * (TAU.ln() + z * z)
}

#[verified_engine::verified]
pub(crate) fn standard_normal_pdf(z: f64) -> f64 {
    standard_normal_ln_pdf(z).exp()
}

#[verified_engine::verified]
pub(crate) fn standard_normal_cdf(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    0.5 * erf::erfc(-z * FRAC_1_SQRT_2)
}

/// Acklam rational approximation followed by one Halley refinement step (~1e-15 relative accuracy).
#[verified_engine::verified]
pub(crate) fn standard_normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let p_low = 0.02425;
    let x0 = if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - p_low {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (-p).ln_1p()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    let err = standard_normal_cdf(x0) - p;
    let pdf = standard_normal_pdf(x0);
    if pdf > 0.0 {
        let u = err / pdf;
        x0 - u / (1.0 + 0.5 * x0 * u)
    } else {
        x0
    }
}

#[verified_engine::verified]
pub(crate) fn sample_standard_normal(rng: &mut OxidizeRng) -> f64 {
    let u1 = sample_uniform_open(rng);
    let u2 = sample_uniform_open(rng);
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}

/// Marsaglia-Tsang squeeze rejection for Gamma(shape, scale = 1.0) with bounded iterations.
#[verified_engine::verified]
pub(crate) fn sample_gamma_unit_scale(shape: f64, rng: &mut OxidizeRng) -> f64 {
    if shape < 1.0 {
        let boost = sample_uniform_open(rng).powf(1.0 / shape);
        return sample_gamma_ge_one(shape + 1.0, rng) * boost;
    }
    sample_gamma_ge_one(shape, rng)
}

fn sample_gamma_ge_one(shape: f64, rng: &mut OxidizeRng) -> f64 {
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    for _ in 0..512 {
        let z = sample_standard_normal(rng);
        let one_plus_cz = 1.0 + c * z;
        if one_plus_cz <= 0.0 {
            continue;
        }
        let v = one_plus_cz * one_plus_cz * one_plus_cz;
        let u = sample_uniform_open(rng);
        if u < 1.0 - 0.0331 * (z * z) * (z * z) {
            return d * v;
        }
        if u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
    shape
}

/// Exact Knuth / Hormann PTRS Poisson variate sampler with zero heap allocations.
#[verified_engine::verified]
pub(crate) fn sample_poisson(lambda: f64, rng: &mut OxidizeRng) -> u64 {
    if lambda <= 0.0 {
        return 0;
    }
    if lambda < 30.0 {
        let limit = (-lambda).exp();
        let mut k = 0_u64;
        let mut prod = 1.0;
        for _ in 0..2048 {
            prod *= sample_uniform_open(rng);
            if prod <= limit {
                return k;
            }
            k += 1;
        }
        return k;
    }
    sample_poisson_ptrs(lambda, rng)
}

fn sample_poisson_ptrs(lambda: f64, rng: &mut OxidizeRng) -> u64 {
    let slam = lambda.sqrt();
    let b = 0.931 + 2.53 * slam;
    let a = -0.059 + 0.02483 * b;
    let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    let vr = 0.9277 - 3.6224 / (b - 2.0);
    let log_lam = lambda.ln();
    for _ in 0..1024 {
        let u = sample_uniform_open(rng) - 0.5;
        let v = sample_uniform_open(rng);
        let us = 0.5 - u.abs();
        let k_float = ((2.0 * a / us + b) * u + lambda + 0.43).floor();
        if k_float < 0.0 {
            continue;
        }
        let k = k_float as u64;
        if us >= 0.07 && v <= vr {
            return k;
        }
        if us < 0.013 && v > us {
            continue;
        }
        let lhs = (v * inv_alpha / (a / (us * us) + b)).ln();
        let rhs = -lambda + (k as f64) * log_lam - gamma::ln_gamma(k as f64 + 1.0);
        if lhs <= rhs {
            return k;
        }
    }
    lambda.round() as u64
}

/// Safeguarded Newton-bisection solver on `(lo_init, hi_init)` for continuous quantiles.
pub(crate) fn invert_monotone_cdf<FCdf, FPdf>(
    target_p: f64,
    mut lo: f64,
    mut hi: f64,
    mut guess: f64,
    cdf_fn: FCdf,
    pdf_fn: FPdf,
) -> f64
where
    FCdf: Fn(f64) -> f64,
    FPdf: Fn(f64) -> f64,
{
    guess = guess.clamp(lo, hi);
    for _ in 0..64 {
        let f_val = cdf_fn(guess) - target_p;
        if f_val.abs() <= 1e-13 || (hi - lo) <= 1e-13 * (1.0 + guess.abs()) {
            return guess;
        }
        if f_val < 0.0 {
            lo = guess;
        } else {
            hi = guess;
        }
        let slope = pdf_fn(guess);
        let candidate = if slope > 1e-30 && slope.is_finite() {
            guess - f_val / slope
        } else {
            0.5 * (lo + hi)
        };
        guess = if candidate > lo && candidate < hi {
            candidate
        } else {
            0.5 * (lo + hi)
        };
    }
    0.5 * (lo + hi)
}

#[verified_engine::verified]
pub(crate) fn gamma_unit_quantile(shape: f64, p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let z = standard_normal_quantile(p);
    let wh_term = 1.0 - 1.0 / (9.0 * shape) + z / (3.0 * shape.sqrt());
    let mut guess = if wh_term > 0.05 {
        shape * wh_term * wh_term * wh_term
    } else {
        (p * gamma::gamma(shape + 1.0)).powf(1.0 / shape).max(1e-12)
    };
    let mut lo = 0.0;
    let mut hi = (guess * 4.0 + shape + 20.0).max(1.0);
    while gamma::gamma_lr(shape, hi) < p && hi < 1e12 {
        lo = hi;
        hi *= 2.0;
    }
    guess = guess.clamp(lo, hi);
    let ln_g = gamma::ln_gamma(shape);
    invert_monotone_cdf(
        p,
        lo,
        hi,
        guess,
        |x| {
            if x <= 0.0 {
                0.0
            } else {
                gamma::gamma_lr(shape, x)
            }
        },
        |x| {
            if x <= 0.0 {
                0.0
            } else {
                ((shape - 1.0) * x.ln() - x - ln_g).exp()
            }
        },
    )
}

#[verified_engine::verified]
pub(crate) fn beta_quantile(alpha: f64, beta_param: f64, p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    let mean = alpha / (alpha + beta_param);
    let guess = mean.clamp(1e-8, 1.0 - 1e-8);
    let ln_b = beta::ln_beta(alpha, beta_param);
    invert_monotone_cdf(
        p,
        0.0,
        1.0,
        guess,
        |x| {
            if x <= 0.0 {
                0.0
            } else if x >= 1.0 {
                1.0
            } else {
                beta::beta_reg(alpha, beta_param, x)
            }
        },
        |x| {
            if x <= 0.0 || x >= 1.0 {
                0.0
            } else {
                ((alpha - 1.0) * x.ln() + (beta_param - 1.0) * (1.0 - x).ln() - ln_b).exp()
            }
        },
    )
}

#[verified_engine::verified]
pub(crate) fn sample_mean_and_var(data: &[f64]) -> Result<(f64, f64, f64), DistributionError> {
    if data.len() < 2 {
        return Err(DistributionError::InsufficientData {
            required: 2,
            actual: data.len(),
        });
    }
    let n = data.len() as f64;
    let mut sum = 0.0;
    for &x in data {
        if !x.is_finite() {
            return Err(DistributionError::FittingFailed {
                reason: "sample contains non-finite values",
            });
        }
        sum += x;
    }
    let mean = sum / n;
    let mut sq = 0.0;
    for &x in data {
        let diff = x - mean;
        sq += diff * diff;
    }
    Ok((mean, sq / n, sq / (n - 1.0)))
}

// [cite:clinical_trials_statistics]
