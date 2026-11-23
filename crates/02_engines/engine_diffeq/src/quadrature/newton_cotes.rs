//! Composite `Trapezoidal` and `Simpson` quadrature rules with Richardson error bounds and discrete grid kernels.
// [cite:engine_diffeq]

use super::{IntegrationResult, Integrator};

/// Maximum midpoint-doubling refinement levels when `eps > 0.0`.
const MAX_RICHARDSON_LEVELS: usize = 12;

/// Composite Trapezoidal quadrature rule with midpoint-reusing Richardson error estimation.
///
/// When `eps <= 0.0`, evaluates `steps` panels alongside a midpoint-doubled `2 * steps` estimate
/// to report a conservative $\mathcal{O}(h^2)$ Richardson error bound $|I_{2n} - I_n| \cdot \frac{4}{3}$.
/// When `eps > 0.0`, iteratively doubles `n` until the Richardson error bound meets `eps`
/// (up to `MAX_RICHARDSON_LEVELS` refinements).
#[derive(Debug, Clone, Copy)]
pub struct Trapezoidal {
    /// Base number of subintervals (`n`).
    pub steps: usize,
}

impl Trapezoidal {
    /// Creates a new `Trapezoidal` integrator with `steps` initial subintervals.
    #[must_use]
    #[verified_engine::verified]
    pub const fn new(steps: usize) -> Self {
        Self { steps }
    }
}

impl Default for Trapezoidal {
    #[verified_engine::verified]
    fn default() -> Self {
        Self { steps: 1000 }
    }
}

#[verified_engine::verified]
fn composite_trapezoidal_eval<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / (n as f64);
    let mut sum = 0.5 * (f(a) + f(b));
    for i in 1..n {
        sum += f(a + (i as f64) * h);
    }
    sum * h
}

#[verified_engine::verified]
fn midpoint_refine_trapezoidal<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    n: usize,
    i_n: f64,
) -> f64 {
    let h = (b - a) / (n as f64);
    let mut mid_sum = 0.0_f64;
    for i in 0..n {
        mid_sum += f(a + ((i as f64) + 0.5) * h);
    }
    0.5 * i_n + 0.5 * h * mid_sum
}

impl Integrator for Trapezoidal {
    #[verified_engine::verified]
    fn integrate<F>(&self, f: F, a: f64, b: f64, eps: f64) -> IntegrationResult
    where
        F: Fn(f64) -> f64,
    {
        if self.steps == 0 || !a.is_finite() || !b.is_finite() || (b - a).abs() == 0.0 {
            return IntegrationResult {
                value: 0.0,
                error: 0.0,
            };
        }
        let mut n = self.steps;
        let i_base = composite_trapezoidal_eval(&f, a, b, n);
        let i_double = midpoint_refine_trapezoidal(&f, a, b, n, i_base);
        let err_base = (i_double - i_base).abs() * (4.0 / 3.0);

        if !eps.is_finite() || eps <= 0.0 {
            return IntegrationResult {
                value: i_base,
                error: err_base,
            };
        }

        let mut prev = i_base;
        let mut curr = i_double;
        let mut err = (curr - prev).abs() * (4.0 / 3.0);
        for _ in 0..MAX_RICHARDSON_LEVELS {
            if err <= eps || n >= (1 << 20) {
                break;
            }
            n *= 2;
            prev = curr;
            curr = midpoint_refine_trapezoidal(&f, a, b, n, prev);
            err = (curr - prev).abs() * (4.0 / 3.0);
        }
        IntegrationResult {
            value: curr,
            error: err,
        }
    }
}

/// Composite Simpson's 1/3 rule ($\mathcal{O}(h^4)$) with Richardson error estimation and `eps > 0` refinement.
#[derive(Debug, Clone, Copy)]
pub struct Simpson {
    /// Base number of subintervals (rounded up to an even count `>= 2`).
    pub steps: usize,
}

impl Simpson {
    /// Creates a new `Simpson` integrator with `steps` initial subintervals.
    #[must_use]
    #[verified_engine::verified]
    pub const fn new(steps: usize) -> Self {
        Self { steps }
    }
}

impl Default for Simpson {
    #[verified_engine::verified]
    fn default() -> Self {
        Self { steps: 200 }
    }
}

impl Integrator for Simpson {
    #[verified_engine::verified]
    fn integrate<F>(&self, f: F, a: f64, b: f64, eps: f64) -> IntegrationResult
    where
        F: Fn(f64) -> f64,
    {
        if self.steps == 0 || !a.is_finite() || !b.is_finite() || (b - a).abs() == 0.0 {
            return IntegrationResult {
                value: 0.0,
                error: 0.0,
            };
        }
        // S_{2m} = (4 * T_{2m} - T_m) / 3
        let mut half_n = self.steps.div_ceil(2).max(1);
        let t_init = composite_trapezoidal_eval(&f, a, b, half_n);
        let mut t_curr = midpoint_refine_trapezoidal(&f, a, b, half_n, t_init);
        let s_base = (4.0 * t_curr - t_init) / 3.0;

        half_n *= 2;
        let t_next = midpoint_refine_trapezoidal(&f, a, b, half_n, t_curr);
        let s_fine = (4.0 * t_next - t_curr) / 3.0;
        let err_base = (s_fine - s_base).abs() * (16.0 / 15.0);

        if !eps.is_finite() || eps <= 0.0 {
            return IntegrationResult {
                value: s_base,
                error: err_base,
            };
        }

        let mut s_curr = s_fine;
        t_curr = t_next;
        let mut err = err_base / 16.0;
        for _ in 0..MAX_RICHARDSON_LEVELS {
            if err <= eps || half_n >= (1 << 19) {
                break;
            }
            half_n *= 2;
            let t_doubled = midpoint_refine_trapezoidal(&f, a, b, half_n, t_curr);
            let s_doubled = (4.0 * t_doubled - t_curr) / 3.0;
            err = (s_doubled - s_curr).abs() * (16.0 / 15.0);
            t_curr = t_doubled;
            s_curr = s_doubled;
        }
        IntegrationResult {
            value: s_curr,
            error: err,
        }
    }
}

/// Integrates discrete non-uniform samples `(xs[i], ys[i])` via the composite trapezoidal rule.
#[must_use]
#[verified_engine::verified]
pub fn trapezoidal_nonuniform(xs: &[f64], ys: &[f64]) -> f64 {
    let n = core::cmp::min(xs.len(), ys.len());
    if n < 2 {
        return 0.0;
    }
    let mut area = 0.0_f64;
    let mut idx = 0;
    while idx + 1 < n {
        let dx = xs[idx + 1] - xs[idx];
        area += 0.5 * (ys[idx] + ys[idx + 1]) * dx;
        idx += 1;
    }
    area
}

/// Integrates uniformly spaced samples `values` with grid spacing `dx > 0` via the composite trapezoidal rule.
#[must_use]
#[verified_engine::verified]
pub fn trapezoidal_uniform_grid(values: &[f64], dx: f64) -> f64 {
    trapezoidal_uniform_mapped(values, dx, |v| v)
}

/// Integrates uniformly spaced samples `values` mapped through `sanitize` with spacing `dx > 0`.
#[must_use]
#[verified_engine::verified]
pub fn trapezoidal_uniform_mapped<F: Fn(f64) -> f64>(values: &[f64], dx: f64, sanitize: F) -> f64 {
    let n = values.len();
    if n < 2 || !dx.is_finite() || dx <= 0.0 {
        return 0.0;
    }
    let mut sum = 0.5 * (sanitize(values[0]) + sanitize(values[n - 1]));
    let mut idx = 1;
    while idx + 1 < n {
        sum += sanitize(values[idx]);
        idx += 1;
    }
    sum * dx
}
