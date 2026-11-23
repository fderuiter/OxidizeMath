//! Concentric-ring kernel shells $K(r)$, Gaussian growth curve $G(u; \mu, \sigma)$,
//! and optional zero-allocation [`math_commons::ast::CompiledExpr`] formula evaluators.

use super::{MAX_KERNEL_RADIUS, MAX_KERNEL_RINGS, MIN_KERNEL_RADIUS};
use math_commons::ast::CompiledExpr;
use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex;
use verified_engine::Theory;

/// Maximum number of free scalar parameters stored inline for custom Lenia AST rules.
pub const MAX_LENIA_RULE_PARAMS: usize = 8;

/// Evaluates Lenia's smooth $C^\infty$ exponential bump core on $r \in (0, 1)$:
///
/// $$K_C(r) = \exp\!\left(4 - \frac{1}{r(1 - r)}\right)$$
///
/// Returns `0.0` outside $(0, 1)$, peaking at $K_C(0.5) = 1.0$.
#[verified_engine::verified]
pub fn kernel_core_bump(r: f64) -> f64 {
    if !(1e-6..(1.0 - 1e-6)).contains(&r) {
        return 0.0;
    }
    let denom = r * (1.0 - r);
    (4.0 - 1.0 / denom).exp()
}

/// Evaluates Lenia's Gaussian growth mapping $G(u; \mu, \sigma) \in [-1, 1]$:
///
/// $$G(u; \mu, \sigma) = 2 \exp\!\left(-\frac{(u - \mu)^2}{2\sigma^2}\right) - 1$$
#[verified_engine::verified]
pub fn growth_gaussian(u: f64, mu: f64, sigma: f64) -> f64 {
    let safe_sigma = if sigma.is_finite() {
        sigma.abs().max(1e-6)
    } else {
        0.015
    };
    let z = (u - mu) / safe_sigma;
    2.0 * (-0.5 * z * z).exp() - 1.0
}

/// Writes a periodic wrap-around spatial kernel of interaction `radius` into `out` (`width * height`)
/// normalized to unit $L^1$ torus mass ($\sum_{x, y} K(x, y) = 1$), returning the pre-normalized sum.
#[verified_engine::verified]
pub fn write_normalized_torus_kernel<F>(
    width: usize,
    height: usize,
    radius: f64,
    out: &mut [Complex<f64>],
    radial_fn: F,
) -> f64
where
    F: Fn(f64) -> f64,
{
    let total = width * height;
    if total == 0 || out.len() < total {
        return 0.0;
    }
    let safe_r = if radius.is_finite() {
        radius.abs().max(1e-3)
    } else {
        13.0
    };
    let raw_sum = populate_torus_radial(width, height, 1.0 / safe_r, out, &radial_fn);
    if raw_sum.is_finite() && raw_sum > 1e-15 {
        let inv_sum = 1.0 / raw_sum;
        for cell in out.iter_mut().take(total) {
            cell.re *= inv_sum;
        }
        raw_sum
    } else {
        for cell in out.iter_mut().take(total) {
            *cell = Complex::new(0.0, 0.0);
        }
        out[0] = Complex::new(1.0, 0.0);
        0.0
    }
}

#[verified_engine::verified]
fn populate_torus_radial<F>(
    width: usize,
    height: usize,
    inv_r: f64,
    out: &mut [Complex<f64>],
    radial_fn: &F,
) -> f64
where
    F: Fn(f64) -> f64,
{
    let half_w = width / 2;
    let half_h = height / 2;
    let mut sum = 0.0;
    for y in 0..height {
        let dy = if y <= half_h {
            y as f64
        } else {
            (y as f64) - (height as f64)
        };
        for x in 0..width {
            let dx = if x <= half_w {
                x as f64
            } else {
                (x as f64) - (width as f64)
            };
            let r_norm = dx.hypot(dy) * inv_r;
            let raw = radial_fn(r_norm);
            let val = if raw.is_finite() && raw > 0.0 {
                raw.min(1e6)
            } else {
                0.0
            };
            out[flatten_2d_index(x, y, width)] = Complex::new(val, 0.0);
            sum += val;
        }
    }
    sum
}

#[verified_engine::verified]
fn copy_inline_params(src: &[f64]) -> ([f64; MAX_LENIA_RULE_PARAMS], usize) {
    let count = src.len().min(MAX_LENIA_RULE_PARAMS);
    let mut buf = [0.0; MAX_LENIA_RULE_PARAMS];
    for (dst, &val) in buf.iter_mut().zip(src.iter().take(count)) {
        *dst = if val.is_finite() { val } else { 0.0 };
    }
    (buf, count)
}

#[verified_engine::verified]
fn sample_unit_interval<F>(samples: usize, eval: F) -> Vec<[f64; 2]>
where
    F: Fn(f64) -> f64,
{
    let count = samples.max(2);
    let denom = (count - 1) as f64;
    let mut points = Vec::with_capacity(count);
    for idx in 0..count {
        let x = (idx as f64) / denom;
        points.push([x, eval(x)]);
    }
    points
}

/// Parameters governing the continuous Gaussian growth mapping $G(u; \mu, \sigma)$.
#[derive(Debug, Clone, Copy, PartialEq, Theory)]
#[theory(
    description = "Gaussian growth response curve mapping neighborhood potential u to local growth rate in [-1, 1].",
    citation = "Chan, B. W.-C. (2019). Lenia: Biology of Artificial Life. Complex Systems, 28(3), 251-286."
)]
pub struct GrowthParams {
    /// Optimal neighborhood potential center ($\mu$).
    #[theory(min = 0.01, max = 0.60, step = 0.005)]
    pub mu: f64,
    /// Potential niche width ($\sigma$).
    #[theory(min = 0.002, max = 0.15, step = 0.001)]
    pub sigma: f64,
}

impl GrowthParams {
    /// Creates clamped finite Gaussian growth parameters.
    #[verified_engine::verified]
    pub fn new(mu: f64, sigma: f64) -> Self {
        let clean_mu = if mu.is_finite() {
            mu.clamp(0.001, 1.0)
        } else {
            0.15
        };
        let clean_sigma = if sigma.is_finite() {
            sigma.abs().clamp(1e-4, 0.5)
        } else {
            0.015
        };
        Self {
            mu: clean_mu,
            sigma: clean_sigma,
        }
    }

    /// Evaluates $G(u; \mu, \sigma)$ at potential `u`.
    #[verified_engine::verified]
    pub fn evaluate(&self, u: f64) -> f64 {
        growth_gaussian(u, self.mu, self.sigma)
    }

    /// Samples `[u, G(u)]` across $u \in [0, 1]$ for plot inspection and telemetry export.
    #[verified_engine::verified]
    pub fn sample_curve(&self, samples: usize) -> Vec<[f64; 2]> {
        sample_unit_interval(samples, |u| self.evaluate(u))
    }
}

impl Default for GrowthParams {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            mu: 0.15,
            sigma: 0.015,
        }
    }
}

/// Custom 1D AST bytecode growth evaluator clamping $G(u)$ safely into $[-1, 1]$.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledGrowthRule {
    compiled: CompiledExpr,
    params: [f64; MAX_LENIA_RULE_PARAMS],
    param_count: usize,
}

impl CompiledGrowthRule {
    /// Wraps a compiled 1D expression (`axis 0 = u`) and up to [`MAX_LENIA_RULE_PARAMS`] parameters.
    #[verified_engine::verified]
    pub fn new(compiled: CompiledExpr, params: &[f64]) -> Self {
        let (buf, param_count) = copy_inline_params(params);
        Self {
            compiled,
            params: buf,
            param_count,
        }
    }

    /// Underlying postfix bytecode program.
    #[must_use]
    #[verified_engine::verified]
    pub fn compiled(&self) -> &CompiledExpr {
        &self.compiled
    }

    /// Active inline parameter slice corresponding to `compiled.param_names()`.
    #[must_use]
    #[verified_engine::verified]
    pub fn params(&self) -> &[f64] {
        &self.params[..self.param_count]
    }

    /// Updates inline parameter values in-place without reallocating bytecode.
    #[verified_engine::verified]
    pub fn set_params(&mut self, params: &[f64]) {
        let (buf, count) = copy_inline_params(params);
        self.params = buf;
        self.param_count = count;
    }

    /// Evaluates $G(u)$ with `self.params()`, clamping finite outputs to `[-1, 1]` and `NaN` to `-1.0`.
    #[verified_engine::verified]
    pub fn evaluate(&self, u: f64) -> f64 {
        self.evaluate_with_params(u, self.params())
    }

    /// Evaluates $G(u)$ with an explicit parameter slice, performing zero heap allocations.
    #[verified_engine::verified]
    pub fn evaluate_with_params(&self, u: f64, params: &[f64]) -> f64 {
        let safe_u = if u.is_finite() { u } else { 0.0 };
        let raw = self.compiled.eval_1d(safe_u, params);
        if raw.is_nan() {
            -1.0
        } else {
            raw.clamp(-1.0, 1.0)
        }
    }

    /// Samples `[u, G(u)]` across $u \in [0, 1]$ for inspector plotting and dataset export.
    #[verified_engine::verified]
    pub fn sample_curve(&self, samples: usize) -> Vec<[f64; 2]> {
        sample_unit_interval(samples, |u| self.evaluate(u))
    }
}

/// Custom 1D AST bytecode radial kernel evaluator $K(r)$ on $r \in [0, 1]$ normalized to unit $L^1$ torus mass.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledKernelRule {
    compiled: CompiledExpr,
    params: [f64; MAX_LENIA_RULE_PARAMS],
    param_count: usize,
}

impl CompiledKernelRule {
    /// Wraps a compiled 1D radial kernel expression (`axis 0 = r`) and inline parameters.
    #[verified_engine::verified]
    pub fn new(compiled: CompiledExpr, params: &[f64]) -> Self {
        let (buf, param_count) = copy_inline_params(params);
        Self {
            compiled,
            params: buf,
            param_count,
        }
    }

    /// Underlying postfix bytecode program.
    #[must_use]
    #[verified_engine::verified]
    pub fn compiled(&self) -> &CompiledExpr {
        &self.compiled
    }

    /// Active inline parameter slice corresponding to `compiled.param_names()`.
    #[must_use]
    #[verified_engine::verified]
    pub fn params(&self) -> &[f64] {
        &self.params[..self.param_count]
    }

    /// Updates inline parameter values in-place without reallocating bytecode.
    #[verified_engine::verified]
    pub fn set_params(&mut self, params: &[f64]) {
        let (buf, count) = copy_inline_params(params);
        self.params = buf;
        self.param_count = count;
    }

    /// Evaluates the unnormalized radial kernel weight on $r \in [0, 1]$ (clamped non-negative).
    #[verified_engine::verified]
    pub fn shell_unnormalized(&self, r_norm: f64) -> f64 {
        self.shell_unnormalized_with_params(r_norm, self.params())
    }

    /// Evaluates the unnormalized radial kernel weight on $r \in [0, 1]$ using an explicit parameter slice.
    #[verified_engine::verified]
    pub fn shell_unnormalized_with_params(&self, r_norm: f64, params: &[f64]) -> f64 {
        if !(0.0..=1.0).contains(&r_norm) {
            return 0.0;
        }
        let raw = self.compiled.eval_1d(r_norm, params);
        if raw.is_finite() && raw > 0.0 {
            raw.min(1e6)
        } else {
            0.0
        }
    }

    /// Samples `[r_norm, K(r_norm)]` across $r_{\text{norm}} \in [0, 1]$.
    #[verified_engine::verified]
    pub fn sample_radial_profile(&self, samples: usize) -> Vec<[f64; 2]> {
        sample_unit_interval(samples, |r| self.shell_unnormalized(r))
    }

    /// Writes the periodic wrap-around spatial kernel normalized to unit $L^1$ torus mass.
    #[verified_engine::verified]
    pub fn write_wrapped_spatial_kernel(
        &self,
        width: usize,
        height: usize,
        radius: f64,
        out: &mut [Complex<f64>],
    ) -> f64 {
        self.write_wrapped_spatial_kernel_with_params(width, height, radius, self.params(), out)
    }

    /// Writes the periodic wrap-around spatial kernel using an explicit parameter slice with zero heap allocations.
    #[verified_engine::verified]
    pub fn write_wrapped_spatial_kernel_with_params(
        &self,
        width: usize,
        height: usize,
        radius: f64,
        params: &[f64],
        out: &mut [Complex<f64>],
    ) -> f64 {
        write_normalized_torus_kernel(width, height, radius, out, |r| {
            self.shell_unnormalized_with_params(r, params)
        })
    }
}

/// Concentric-ring kernel shell specification $K_S(r; R, \beta)$ with $R \in [4, 32]$ and $1..=4$ rings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KernelShells {
    radius: f64,
    weights: [f64; MAX_KERNEL_RINGS],
    ring_count: usize,
}

impl KernelShells {
    /// Constructs a concentric-ring kernel shell with radius clamped to `4.0..=32.0`
    /// and `1..=4` non-negative ring weights $\beta_k \in [0, 1]$.
    #[verified_engine::verified]
    pub fn new(radius: f64, ring_weights: &[f64]) -> Self {
        let clamped_r = if radius.is_finite() {
            radius.clamp(MIN_KERNEL_RADIUS, MAX_KERNEL_RADIUS)
        } else {
            13.0
        };
        let count = ring_weights.len().clamp(1, MAX_KERNEL_RINGS);
        let mut weights = [0.0; MAX_KERNEL_RINGS];
        let mut weight_sum = 0.0;
        for (idx, slot) in weights.iter_mut().enumerate().take(count) {
            let raw = ring_weights.get(idx).copied().unwrap_or(1.0);
            let clean = if raw.is_finite() {
                raw.clamp(0.0, 1.0)
            } else {
                0.0
            };
            *slot = clean;
            weight_sum += clean;
        }
        if weight_sum <= 1e-12 {
            weights[0] = 1.0;
        }
        Self {
            radius: clamped_r,
            weights,
            ring_count: count,
        }
    }

    /// Returns the interaction radius $R$ in grid cells (`4.0..=32.0`).
    #[verified_engine::verified]
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Returns the number of active concentric rings (`1..=4`).
    #[verified_engine::verified]
    pub fn ring_count(&self) -> usize {
        self.ring_count
    }

    /// Returns the active concentric ring weights $\beta_0, \dots, \beta_{B-1}$.
    #[verified_engine::verified]
    pub fn ring_weights(&self) -> &[f64] {
        &self.weights[..self.ring_count]
    }

    /// Evaluates the unnormalized radial shell value $K_S(r) \in [0, 1]$ for normalized distance $r = d / R \in [0, 1]$.
    #[verified_engine::verified]
    pub fn shell_unnormalized(&self, r_norm: f64) -> f64 {
        if !(0.0..1.0).contains(&r_norm) {
            return 0.0;
        }
        let scaled = r_norm * (self.ring_count as f64);
        let ring_idx = (scaled.floor() as usize).min(self.ring_count - 1);
        let local_r = scaled - (ring_idx as f64);
        self.weights[ring_idx] * kernel_core_bump(local_r)
    }

    /// Samples `[r_norm, K_S(r_norm)]` across $r_{\text{norm}} \in [0, 1]$ for radial kernel inspection.
    #[verified_engine::verified]
    pub fn sample_radial_profile(&self, samples: usize) -> Vec<[f64; 2]> {
        sample_unit_interval(samples, |r| self.shell_unnormalized(r))
    }

    /// Writes the periodic wrap-around spatial kernel into `out` (`width * height`)
    /// normalized to unit $L^1$ sum ($\sum_{x, y} K(x, y) = 1$), returning the raw pre-normalized sum.
    #[verified_engine::verified]
    pub fn write_wrapped_spatial_kernel(
        &self,
        width: usize,
        height: usize,
        out: &mut [Complex<f64>],
    ) -> f64 {
        write_normalized_torus_kernel(width, height, self.radius, out, |r| {
            self.shell_unnormalized(r)
        })
    }
}

impl Default for KernelShells {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new(13.0, &[1.0])
    }
}
