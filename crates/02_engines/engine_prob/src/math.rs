//! Pure `#![no_std]` Gaussian distribution functions, sorting, quantiles, and Cholesky helpers.

use crate::error::ProbError;
use core::f64::consts::{FRAC_1_SQRT_2, TAU};
use oxidize_core::rng::OxidizeRng;

/// Two-sided 95% standard normal critical value $\Phi^{-1}(0.975)$.
pub const Z_95_TWO_SIDED: f64 = 1.959_963_984_540_054;

const INV_SQRT_TAU: f64 = 0.398_942_280_401_432_7;

/// Standard normal probability density $\phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$.
#[inline]
#[must_use]
pub fn standard_normal_pdf(x: f64) -> f64 {
    INV_SQRT_TAU * libm::exp(-0.5 * x * x)
}

/// Standard normal cumulative distribution $\Phi(x) = \frac{1}{2}\operatorname{erfc}(-x / \sqrt{2})$.
#[inline]
#[must_use]
pub fn standard_normal_cdf(x: f64) -> f64 {
    (0.5 * libm::erfc(-x * FRAC_1_SQRT_2)).clamp(0.0, 1.0)
}

fn acklam_initial_guess(p: f64) -> f64 {
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
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = libm::sqrt(-2.0 * libm::log(p));
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = libm::sqrt(-2.0 * libm::log(1.0 - p));
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// High-precision standard normal quantile $\Phi^{-1}(p)$ via Acklam rational + Halley refinement.
#[must_use]
pub fn standard_normal_inv_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let x0 = acklam_initial_guess(p);
    let err = standard_normal_cdf(x0) - p;
    let pdf = standard_normal_pdf(x0);
    if pdf <= 1e-300 {
        return x0;
    }
    let u = err / pdf;
    x0 - u / (1.0 + 0.5 * x0 * u)
}

/// Generates an independent standard normal pair $(Z_1, Z_2)$ via `#![no_std]` Box-Muller.
#[inline]
pub fn box_muller_pair(rng: &mut OxidizeRng) -> (f64, f64) {
    let u1 = rng.next_f64_open01().max(1e-300);
    let u2 = rng.next_f64_open01();
    let r = libm::sqrt(-2.0 * libm::log(u1));
    let theta = TAU * u2;
    (r * libm::cos(theta), r * libm::sin(theta))
}

/// Sorts `slice` in ascending IEEE-754 total order (`f64::total_cmp`) with zero heap allocations.
#[inline]
pub fn sort_total_f64(slice: &mut [f64]) {
    slice.sort_unstable_by(f64::total_cmp);
}

/// Evaluates linear-interpolated quantile `q` in `[0, 1]` on a non-decreasing `sorted` slice.
#[inline]
#[must_use]
pub fn sorted_quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * ((sorted.len() - 1) as f64);
    let lo = libm::floor(pos) as usize;
    let hi = (libm::ceil(pos) as usize).min(sorted.len() - 1);
    let frac = pos - (lo as f64);
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Computes `(mean, sample_variance)` via Welford's one-pass recurrence, validating finiteness.
pub fn mean_and_sample_variance(values: &[f64]) -> Result<(f64, f64), ProbError> {
    if values.is_empty() {
        return Err(ProbError::EmptySample);
    }
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for (idx, &x) in values.iter().enumerate() {
        if !x.is_finite() {
            return Err(ProbError::NonFiniteInput {
                context: "sample values",
            });
        }
        let k = (idx + 1) as f64;
        let delta = x - mean;
        mean += delta / k;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }
    let var = if values.len() > 1 {
        m2 / ((values.len() - 1) as f64)
    } else {
        0.0
    };
    Ok((mean, var))
}

/// Maps 2D row-major coordinates `(row, col)` with row stride `width` to a flat 1D slice index.
#[inline]
#[must_use]
pub const fn flatten_row_major(row: usize, col: usize, width: usize) -> usize {
    row * width + col
}

/// Computes the lower-triangular Cholesky factor $L$ ($A = L L^\top$) of a symmetric positive-definite
/// `dim x dim` row-major matrix into `lower_out` with zero heap allocations.
pub fn cholesky_lower_row_major(
    matrix: &[f64],
    dim: usize,
    lower_out: &mut [f64],
) -> Result<(), ProbError> {
    if dim == 0 || matrix.len() < dim * dim || lower_out.len() < dim * dim {
        return Err(ProbError::DegenerateMatrix {
            reason: "invalid matrix or output buffer length for Cholesky factorization",
        });
    }
    lower_out[..dim * dim].fill(0.0);
    for i in 0..dim {
        for j in 0..=i {
            let a_ij = matrix[flatten_row_major(i, j, dim)];
            if !a_ij.is_finite() {
                return Err(ProbError::NonFiniteInput {
                    context: "Cholesky input matrix",
                });
            }
            let mut sum = a_ij;
            for k in 0..j {
                sum -= lower_out[flatten_row_major(i, k, dim)]
                    * lower_out[flatten_row_major(j, k, dim)];
            }
            if i == j {
                if sum <= 1e-14 {
                    return Err(ProbError::DegenerateMatrix {
                        reason: "matrix is not symmetric positive-definite",
                    });
                }
                lower_out[flatten_row_major(i, i, dim)] = libm::sqrt(sum);
            } else {
                lower_out[flatten_row_major(i, j, dim)] =
                    sum / lower_out[flatten_row_major(j, j, dim)];
            }
        }
    }
    Ok(())
}
