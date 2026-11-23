//! Stack-queued zero-allocation adaptive 7-point Gauss / 15-point Kronrod (`G7-K15`) quadrature.
// [cite:engine_diffeq]

use super::{IntegrationResult, Integrator};
use crate::error::DiffeqError;
use crate::math::pow_pos_f64;
use crate::workspace::SolverWorkspace;

/// Default stack capacity (in subintervals) of [`GaussKronrod15`].
pub const GK15_STACK_SUBINTERVALS: usize = 64;

/// Non-negative 15-point Kronrod abscissae on `[-1, 1]` (QUADPACK `DQK15`).
const XGK: [f64; 8] = [
    0.991_455_371_120_812_6,
    0.949_107_912_342_758_5,
    0.864_864_423_359_769_1,
    0.741_531_185_599_394_4,
    0.586_087_235_467_691_1,
    0.405_845_151_377_397_2,
    0.207_784_955_007_898_5,
    0.0,
];

/// 15-point Kronrod weights corresponding to [`XGK`] (QUADPACK `DQK15`).
const WGK: [f64; 8] = [
    0.022_935_322_010_529_224,
    0.063_092_092_629_978_56,
    0.104_790_010_322_250_18,
    0.140_653_259_715_525_92,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_42,
    0.204_432_940_075_298_9,
    0.209_482_141_084_727_82,
];

/// Embedded 7-point Gauss-Legendre weights at `XGK[1]`, `XGK[3]`, `XGK[5]`, and `XGK[7]`.
const WG: [f64; 4] = [
    0.129_484_966_168_869_7,
    0.279_705_391_489_276_64,
    0.381_830_050_505_118_9,
    0.417_959_183_673_469_4,
];

#[derive(Clone, Copy, Default)]
struct GkSubinterval {
    a: f64,
    b: f64,
    val: f64,
    err: f64,
}

/// Evaluates a single non-adaptive 15-point Gauss-Kronrod panel over `[a, b]` with QUADPACK error rescaling.
#[must_use]
#[verified_engine::verified]
pub fn gk15_panel<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64) -> IntegrationResult {
    let center = 0.5 * (a + b);
    let half_len = 0.5 * (b - a);
    let abs_half = half_len.abs();

    let fc = f(center);
    let mut res_g = WG[3] * fc;
    let mut res_k = WGK[7] * fc;
    let mut res_abs = WGK[7] * fc.abs();
    let mut fv1 = [0.0_f64; 7];
    let mut fv2 = [0.0_f64; 7];

    for j in 0..7 {
        let absc = half_len * XGK[j];
        let f1 = f(center - absc);
        let f2 = f(center + absc);
        fv1[j] = f1;
        fv2[j] = f2;
        let fsum = f1 + f2;
        res_k += WGK[j] * fsum;
        res_abs += WGK[j] * (f1.abs() + f2.abs());
        if j % 2 == 1 {
            res_g += WG[j / 2] * fsum;
        }
    }

    let mean_k = 0.5 * res_k;
    let mut res_asc = WGK[7] * (fc - mean_k).abs();
    for j in 0..7 {
        res_asc += WGK[j] * ((fv1[j] - mean_k).abs() + (fv2[j] - mean_k).abs());
    }

    let integral = res_k * half_len;
    res_abs *= abs_half;
    res_asc *= abs_half;
    let raw_err = ((res_k - res_g) * half_len).abs();
    let rescaled = rescale_quadpack_error(raw_err, res_abs, res_asc);
    IntegrationResult {
        value: integral,
        error: rescaled,
    }
}

#[verified_engine::verified]
fn rescale_quadpack_error(mut err: f64, res_abs: f64, res_asc: f64) -> f64 {
    if res_asc != 0.0 && err != 0.0 {
        let ratio = 200.0 * err / res_asc;
        let quadpack = if ratio < 1.0 {
            res_asc * pow_pos_f64(ratio, 1.5)
        } else {
            res_asc
        };
        if quadpack > err {
            err = quadpack;
        }
    }
    let min_mach = 50.0 * f64::EPSILON * res_abs;
    if res_abs > f64::MIN_POSITIVE / (50.0 * f64::EPSILON) && err < min_mach {
        err = min_mach;
    }
    err
}

/// Adaptive 7-point Gauss / 15-point Kronrod (`G7-K15`) quadrature engine using a fixed-capacity stack queue.
#[derive(Debug, Clone, Copy)]
pub struct GaussKronrod15 {
    /// Maximum number of subintervals maintained in the stack buffer (`1..=GK15_STACK_SUBINTERVALS`).
    pub max_subintervals: usize,
}

impl Default for GaussKronrod15 {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new()
    }
}

impl GaussKronrod15 {
    /// Creates an adaptive `GaussKronrod15` integrator backed by [`GK15_STACK_SUBINTERVALS`] stack slots.
    #[must_use]
    #[verified_engine::verified]
    pub const fn new() -> Self {
        Self {
            max_subintervals: GK15_STACK_SUBINTERVALS,
        }
    }

    /// Creates a `GaussKronrod15` integrator capped at `max_subintervals` (`<= GK15_STACK_SUBINTERVALS`).
    #[must_use]
    #[verified_engine::verified]
    pub const fn with_max_subintervals(max_subintervals: usize) -> Self {
        let cap = if max_subintervals == 0 {
            1
        } else if max_subintervals > GK15_STACK_SUBINTERVALS {
            GK15_STACK_SUBINTERVALS
        } else {
            max_subintervals
        };
        Self {
            max_subintervals: cap,
        }
    }

    /// Integrates `f` over `[a, b]` using caller-owned `workspace` storage (`4 * max_subintervals` `f64`s),
    /// enabling custom subinterval capacities beyond [`GK15_STACK_SUBINTERVALS`] with zero heap allocation.
    #[verified_engine::verified]
    pub fn integrate_with_workspace<F: Fn(f64) -> f64>(
        &self,
        f: F,
        a: f64,
        b: f64,
        eps: f64,
        workspace: &mut SolverWorkspace<'_>,
    ) -> Result<IntegrationResult, DiffeqError> {
        if !a.is_finite() || !b.is_finite() {
            return Err(DiffeqError::InvalidBounds { lower: a, upper: b });
        }
        if !eps.is_finite() || eps <= 0.0 {
            return Err(DiffeqError::InvalidTolerance { tol: eps });
        }
        let max_subs = workspace.capacity() / 4;
        if max_subs == 0 {
            return Err(DiffeqError::InsufficientWorkspace {
                required: 4,
                available: workspace.capacity(),
            });
        }
        if (b - a).abs() == 0.0 {
            return Ok(IntegrationResult {
                value: 0.0,
                error: 0.0,
            });
        }
        let raw = workspace.try_slice_mut(4 * max_subs)?;
        let p0 = gk15_panel(&f, a, b);
        raw[0] = a;
        raw[1] = b;
        raw[2] = p0.value;
        raw[3] = p0.error;
        let mut count = 1_usize;
        let mut total_val = p0.value;
        let mut total_err = p0.error;

        while count < max_subs && total_err > eps.max(1e-14 * total_val.abs()) {
            let mut worst = 0_usize;
            let mut max_e = raw[3];
            for idx in 1..count {
                let e = raw[4 * idx + 3];
                if e > max_e {
                    max_e = e;
                    worst = idx;
                }
            }
            let sub_a = raw[4 * worst];
            let sub_b = raw[4 * worst + 1];
            let mid = 0.5 * (sub_a + sub_b);
            if mid == sub_a || mid == sub_b {
                break;
            }
            let left = gk15_panel(&f, sub_a, mid);
            let right = gk15_panel(&f, mid, sub_b);
            total_val += (left.value + right.value) - raw[4 * worst + 2];
            total_err += (left.error + right.error) - raw[4 * worst + 3];
            raw[4 * worst] = sub_a;
            raw[4 * worst + 1] = mid;
            raw[4 * worst + 2] = left.value;
            raw[4 * worst + 3] = left.error;
            raw[4 * count] = mid;
            raw[4 * count + 1] = sub_b;
            raw[4 * count + 2] = right.value;
            raw[4 * count + 3] = right.error;
            count += 1;
        }
        Ok(IntegrationResult {
            value: total_val,
            error: total_err.max(0.0),
        })
    }
}

impl Integrator for GaussKronrod15 {
    #[verified_engine::verified]
    fn integrate<F>(&self, f: F, a: f64, b: f64, eps: f64) -> IntegrationResult
    where
        F: Fn(f64) -> f64,
    {
        if !a.is_finite() || !b.is_finite() || (b - a).abs() == 0.0 {
            return IntegrationResult {
                value: 0.0,
                error: 0.0,
            };
        }
        let tol = if eps.is_finite() && eps > 0.0 {
            eps
        } else {
            1e-10
        };
        let cap = self.max_subintervals.clamp(1, GK15_STACK_SUBINTERVALS);
        let mut queue = [GkSubinterval::default(); GK15_STACK_SUBINTERVALS];
        let p0 = gk15_panel(&f, a, b);
        queue[0] = GkSubinterval {
            a,
            b,
            val: p0.value,
            err: p0.error,
        };
        let mut count = 1_usize;
        let mut total_val = p0.value;
        let mut total_err = p0.error;

        while count < cap && total_err > tol.max(1e-14 * total_val.abs()) {
            let mut worst = 0_usize;
            let mut max_e = queue[0].err;
            for (idx, item) in queue.iter().enumerate().take(count).skip(1) {
                if item.err > max_e {
                    max_e = item.err;
                    worst = idx;
                }
            }
            let cur = queue[worst];
            let mid = 0.5 * (cur.a + cur.b);
            if mid == cur.a || mid == cur.b {
                break;
            }
            let left = gk15_panel(&f, cur.a, mid);
            let right = gk15_panel(&f, mid, cur.b);
            total_val += (left.value + right.value) - cur.val;
            total_err += (left.error + right.error) - cur.err;
            queue[worst] = GkSubinterval {
                a: cur.a,
                b: mid,
                val: left.value,
                err: left.error,
            };
            queue[count] = GkSubinterval {
                a: mid,
                b: cur.b,
                val: right.value,
                err: right.error,
            };
            count += 1;
        }
        IntegrationResult {
            value: total_val,
            error: total_err.max(0.0),
        }
    }
}
