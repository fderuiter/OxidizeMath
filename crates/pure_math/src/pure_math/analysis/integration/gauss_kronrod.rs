//! Stack-bounded adaptive 7-point Gauss / 15-point Kronrod quadrature (`GaussKronrod15`).
//!
//! Subdivides worst-error subintervals inside a fixed 64-slot stack pool (`[GkSubinterval; 64]`)
//! so [`Integrator::integrate`] performs adaptive refinement with zero heap allocations.

use super::{IntegrationResult, Integrator};

/// Maximum number of stack-allocated subintervals retained by [`GaussKronrod15`].
pub const GK15_MAX_PANELS: usize = 64;

const XGK: [f64; 8] = [
    0.991_455_371_120_812_6,
    0.949_107_912_342_758_5,
    0.864_864_423_359_769_1,
    0.741_531_185_599_394_5,
    0.586_087_235_467_691_1,
    0.405_845_151_377_397_2,
    0.207_784_955_007_898_5,
    0.0,
];

const WGK: [f64; 8] = [
    0.022_935_322_010_529_224,
    0.063_092_092_629_978_56,
    0.104_790_010_322_250_18,
    0.140_653_259_715_525_92,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_42,
    0.204_432_940_075_298_89,
    0.209_482_141_084_727_82,
];

const WG: [f64; 4] = [
    0.129_484_966_168_869_7,
    0.279_705_391_489_276_64,
    0.381_830_050_505_118_9,
    0.417_959_183_673_469_4,
];

/// Metadata for a single adaptive Gauss-Kronrod subinterval `[left, right]`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GkSubinterval {
    /// Left endpoint of the subinterval.
    pub left: f64,
    /// Right endpoint of the subinterval.
    pub right: f64,
    /// 15-point Kronrod integral estimate on `[left, right]`.
    pub value: f64,
    /// Local QUADPACK-rescaled error bound on `[left, right]`.
    pub error: f64,
}

/// Stack-bounded adaptive 15-point Gauss-Kronrod quadrature integrator (`G7-K15`).
///
/// Maintains up to [`GK15_MAX_PANELS`] subintervals on the stack and bisects the subinterval
/// with the largest local error until the global tolerance `eps` or panel budget is reached.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::integration::{GaussKronrod15, Integrator};
///
/// let res = GaussKronrod15::new().integrate(|x| x.sin(), 0.0, std::f64::consts::PI, 1e-12);
/// assert!((res.value - 2.0).abs() <= res.error);
/// assert!(res.error < 1e-10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GaussKronrod15 {
    /// Maximum stack subintervals used during adaptive refinement (`1..=64`).
    pub max_panels: usize,
}

impl GaussKronrod15 {
    /// Creates an adaptive `GaussKronrod15` integrator using the full 64-panel stack budget.
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self {
            max_panels: GK15_MAX_PANELS,
        }
    }

    /// Creates an adaptive `GaussKronrod15` integrator capped at `max_panels` (`1..=64`).
    #[verified_engine::verified]
    pub fn with_max_panels(max_panels: usize) -> Self {
        Self {
            max_panels: max_panels.clamp(1, GK15_MAX_PANELS),
        }
    }

    /// Integrates `f` over `[min, max]` and copies the final left-to-right sorted adaptive
    /// subintervals into `out_panels`, returning `(result, panel_count)`.
    #[verified_engine::verified]
    pub fn integrate_panels<F>(
        &self,
        f: F,
        min: f64,
        max: f64,
        eps: f64,
        out_panels: &mut [GkSubinterval],
    ) -> (IntegrationResult, usize)
    where
        F: Fn(f64) -> f64,
    {
        if !min.is_finite() || !max.is_finite() || min == max {
            return (
                IntegrationResult {
                    value: 0.0,
                    error: 0.0,
                },
                0,
            );
        }

        let cap = self.max_panels.clamp(1, GK15_MAX_PANELS);
        let mut pool = [GkSubinterval::default(); GK15_MAX_PANELS];
        let seed_count = cap.min(4);
        let span = max - min;
        let mut total_val = 0.0;
        let mut total_err = 0.0;

        for (i, slot) in pool.iter_mut().take(seed_count).enumerate() {
            let a = min + (i as f64 / seed_count as f64) * span;
            let b = if i + 1 == seed_count {
                max
            } else {
                min + ((i + 1) as f64 / seed_count as f64) * span
            };
            *slot = eval_gk15_panel(&f, a, b);
            total_val += slot.value;
            total_err += slot.error;
        }

        let mut count = seed_count;
        let target_tol = eps.abs().max(1e-15);
        while count < cap {
            let eff_tol = target_tol.max(50.0 * f64::EPSILON * total_val.abs());
            if total_err <= eff_tol {
                break;
            }
            let worst = find_worst_panel(&pool, count);
            let panel = pool[worst];
            let mid = 0.5 * (panel.left + panel.right);
            let lo = panel.left.min(panel.right);
            let hi = panel.left.max(panel.right);
            if mid <= lo || mid >= hi {
                break;
            }
            let left_sub = eval_gk15_panel(&f, panel.left, mid);
            let right_sub = eval_gk15_panel(&f, mid, panel.right);
            total_val = total_val - panel.value + left_sub.value + right_sub.value;
            total_err = (total_err - panel.error + left_sub.error + right_sub.error).max(0.0);
            pool[worst] = left_sub;
            pool[count] = right_sub;
            count += 1;
        }

        let mut final_val = 0.0;
        let mut final_err = 0.0;
        for item in pool.iter().take(count) {
            final_val += item.value;
            final_err += item.error;
        }
        let written = copy_sorted_panels(&pool, count, out_panels);
        (
            IntegrationResult {
                value: final_val,
                error: final_err,
            },
            written,
        )
    }
}

impl Default for GaussKronrod15 {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new()
    }
}

impl Integrator for GaussKronrod15 {
    #[verified_engine::verified]
    fn integrate<F>(&self, f: F, min: f64, max: f64, eps: f64) -> IntegrationResult
    where
        F: Fn(f64) -> f64,
    {
        let mut empty: [GkSubinterval; 0] = [];
        self.integrate_panels(f, min, max, eps, &mut empty).0
    }
}

#[verified_engine::verified]
fn find_worst_panel(pool: &[GkSubinterval; GK15_MAX_PANELS], count: usize) -> usize {
    let mut best_idx = 0;
    let mut best_err = pool[0].error;
    for (idx, item) in pool.iter().take(count).enumerate().skip(1) {
        if item.error > best_err {
            best_err = item.error;
            best_idx = idx;
        }
    }
    best_idx
}

#[verified_engine::verified]
fn copy_sorted_panels(
    pool: &[GkSubinterval; GK15_MAX_PANELS],
    count: usize,
    out: &mut [GkSubinterval],
) -> usize {
    let written = count.min(out.len());
    out[..written].copy_from_slice(&pool[..written]);
    for i in 1..written {
        let cur = out[i];
        let cur_x = cur.left.min(cur.right);
        let mut j = i;
        while j > 0 && out[j - 1].left.min(out[j - 1].right) > cur_x {
            out[j] = out[j - 1];
            j -= 1;
        }
        out[j] = cur;
    }
    written
}

#[verified_engine::verified]
fn eval_gk15_panel<F>(f: &F, a: f64, b: f64) -> GkSubinterval
where
    F: Fn(f64) -> f64,
{
    let center = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let abs_half = half.abs();
    let f_center = f(center);

    let mut res_gauss = WG[3] * f_center;
    let mut res_kronrod = WGK[7] * f_center;
    let mut res_abs = WGK[7] * f_center.abs();
    let mut fv1 = [0.0; 7];
    let mut fv2 = [0.0; 7];

    for j in 0..7 {
        let absc = half * XGK[j];
        let f1 = f(center - absc);
        let f2 = f(center + absc);
        fv1[j] = f1;
        fv2[j] = f2;
        let fsum = f1 + f2;
        res_kronrod += WGK[j] * fsum;
        res_abs += WGK[j] * (f1.abs() + f2.abs());
        if j % 2 == 1 {
            res_gauss += WG[j / 2] * fsum;
        }
    }

    let mean = 0.5 * res_kronrod;
    let mut res_asc = WGK[7] * (f_center - mean).abs();
    for j in 0..7 {
        res_asc += WGK[j] * ((fv1[j] - mean).abs() + (fv2[j] - mean).abs());
    }

    let integral = res_kronrod * half;
    res_abs *= abs_half;
    res_asc *= abs_half;
    let err_raw = ((res_kronrod - res_gauss) * half).abs();

    let mut err = if res_asc != 0.0 && err_raw != 0.0 {
        let scaled = res_asc * (200.0 * err_raw / res_asc).powf(1.5).min(1.0);
        scaled.max(err_raw)
    } else {
        err_raw
    };
    let roundoff = 50.0 * f64::EPSILON * res_abs;
    if res_abs > f64::MIN_POSITIVE / (50.0 * f64::EPSILON) {
        err = err.max(roundoff);
    }

    GkSubinterval {
        left: a,
        right: b,
        value: integral,
        error: err,
    }
}
