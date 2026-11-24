//! Concentric-ring kernel shells $K(r)$, Gaussian growth curve $G(u; \mu, \sigma)$,
//! and optional zero-allocation [`math_commons::ast::CompiledExpr`] formula evaluators.

use super::{MAX_KERNEL_RADIUS, MAX_KERNEL_RINGS, MIN_KERNEL_RADIUS};
use math_commons::ast::CompiledExpr;
use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex;
use verified_engine::Theory;

/// Maximum number of free scalar parameters stored inline for custom Lenia AST rules.
pub const MAX_LENIA_RULE_PARAMS: usize = 8;

/// Evaluates Lenia's smooth $C^\infty$ bump $K_C(r) = \exp(4 - (r(1 - r))^{-1})$ (`@chan2019lenia`, `spec:biology_v1`).
/// Numerical stability: guards $r \notin (10^{-6}, 1 - 10^{-6})$ to prevent underflow/division overflow. Domain of validity: finite $r \in \mathbb{R}$ (supported on $(0, 1)$).
#[verified_engine::verified]
pub fn kernel_core_bump(r: f64) -> f64 {
    if !(1e-6..(1.0 - 1e-6)).contains(&r) {
        return 0.0;
    }
    let denom = r * (1.0 - r);
    (4.0 - 1.0 / denom).exp()
}

/// Evaluates Lenia's Gaussian growth mapping $G(u; \mu, \sigma) = 2\exp(-(u - \mu)^2 / (2\sigma^2)) - 1 \in [-1, 1]$ (`@chan2019lenia`, `spec:biology_v1`).
/// Numerical stability: clamps $|\sigma| \ge 10^{-6}$ (fallback `0.015`) to prevent overflow in $(u - \mu)/\sigma$. Domain of validity: finite $u, \mu \in \mathbb{R}$ and $\sigma > 0$.
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

/// Writes a periodic torus kernel normalized to unit $L^1$ mass $\sum_{x,y} K(x, y) = 1$ (`@chan2019lenia`, `spec:biology_v1`).
/// Numerical stability: falls back to a unit impulse at $(0, 0)$ if raw kernel mass $\le 10^{-15}$. Domain of validity: `out.len() >= width * height > 0` and finite `radius > 0`.
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

/// Parameters $(\mu, \sigma)$ governing Lenia's Gaussian growth curve $G(u; \mu, \sigma) = 2\exp(-(u-\mu)^2/(2\sigma^2)) - 1$ (`@chan2019lenia`, `spec:biology_v1`).
/// Numerical stability: bounded away from $\sigma = 0$ via $[\text{1e-4}, 0.5]$ clamping. Domain of validity: $\mu \in [0.001, 1.0]$ and $\sigma \in [10^{-4}, 0.5]$.
#[derive(Debug, Clone, Copy, PartialEq, Theory)]
#[theory(
    description = "Gaussian growth response curve mapping neighborhood potential u to local growth rate in [-1, 1].",
    citation = "chan2019lenia"
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
    /// Creates clamped parameters for $G(u; \mu, \sigma)$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: sanitizes `NaN`/`Inf` to `(0.15, 0.015)`. Domain of validity: clamped to $\mu \in [0.001, 1.0], \sigma \in [10^{-4}, 0.5]$.
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

    /// Evaluates $G(u; \mu, \sigma) \in [-1, 1]$ at potential `u` (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: bounded exponential in $[-1, 1]$. Domain of validity: finite potential $u \in [0, 1]$.
    #[verified_engine::verified]
    pub fn evaluate(&self, u: f64) -> f64 {
        growth_gaussian(u, self.mu, self.sigma)
    }

    /// Samples $[u_k, G(u_k)]$ uniformly over $u_k = k/(N-1) \in [0, 1]$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: clamps $N \ge 2$ to avoid zero division. Domain of validity: `samples >= 2`.
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

/// Custom 1D AST bytecode growth evaluator clamping $G(u; \theta) \in [-1, 1]$ (`@chan2019lenia`, `spec:biology_v1`).
/// Numerical stability: maps `NaN` bytecode outputs to `-1.0` and clamps extremes to $[-1, 1]$. Domain of validity: 1D `CompiledExpr` over $u \in [0, 1]$ with $\le 8$ inline parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledGrowthRule {
    compiled: CompiledExpr,
    params: [f64; MAX_LENIA_RULE_PARAMS],
    param_count: usize,
}

impl CompiledGrowthRule {
    /// Wraps a compiled 1D growth expression $G(u; \theta)$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: non-finite parameters sanitize to `0.0`. Domain of validity: up to `MAX_LENIA_RULE_PARAMS` parameters.
    #[verified_engine::verified]
    pub fn new(compiled: CompiledExpr, params: &[f64]) -> Self {
        let (buf, param_count) = copy_inline_params(params);
        Self {
            compiled,
            params: buf,
            param_count,
        }
    }

    /// Borrows the postfix bytecode for $G(u; \theta)$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: exact reference borrow. Domain of validity: valid for lifetime of `self`.
    #[must_use]
    #[verified_engine::verified]
    pub fn compiled(&self) -> &CompiledExpr {
        &self.compiled
    }

    /// Returns active parameter vector $\theta \in \mathbb{R}^p$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: exact slice view. Domain of validity: $0 \le p \le 8$.
    #[must_use]
    #[verified_engine::verified]
    pub fn params(&self) -> &[f64] {
        &self.params[..self.param_count]
    }

    /// Updates inline parameters $\theta$ for $G(u; \theta)$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: replaces non-finite inputs with `0.0`. Domain of validity: truncates inputs beyond `MAX_LENIA_RULE_PARAMS`.
    #[verified_engine::verified]
    pub fn set_params(&mut self, params: &[f64]) {
        let (buf, count) = copy_inline_params(params);
        self.params = buf;
        self.param_count = count;
    }

    /// Evaluates clamped $G(u; \theta) \in [-1, 1]$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: `NaN` maps to `-1.0` and infinities clamp to $\pm 1$. Domain of validity: finite $u \in [0, 1]$.
    #[verified_engine::verified]
    pub fn evaluate(&self, u: f64) -> f64 {
        self.evaluate_with_params(u, self.params())
    }

    /// Evaluates clamped $G(u; \theta) \in [-1, 1]$ with explicit $\theta$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: zero-allocation stack evaluation clamping `NaN` to `-1.0`. Domain of validity: finite $u \in [0, 1]$ and matching `params`.
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

    /// Samples $[u_k, G(u_k)]$ over $u_k = k/(N-1) \in [0, 1]$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: clamps $N \ge 2$ to prevent zero division. Domain of validity: `samples >= 2`.
    #[verified_engine::verified]
    pub fn sample_curve(&self, samples: usize) -> Vec<[f64; 2]> {
        sample_unit_interval(samples, |u| self.evaluate(u))
    }
}

/// Custom 1D AST bytecode radial kernel $K(r; \theta)$ normalized to $\sum K = 1$ (`@chan2019lenia`, `spec:biology_v1`).
/// Numerical stability: clamps raw radial weights to $[0, 10^6]$ before $L^1$ normalization. Domain of validity: normalized radius $r \in [0, 1]$ with $\le 8$ inline parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledKernelRule {
    compiled: CompiledExpr,
    params: [f64; MAX_LENIA_RULE_PARAMS],
    param_count: usize,
}

impl CompiledKernelRule {
    /// Wraps a compiled 1D radial kernel $K(r; \theta)$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: non-finite parameters sanitize to `0.0`. Domain of validity: up to `MAX_LENIA_RULE_PARAMS` parameters.
    #[verified_engine::verified]
    pub fn new(compiled: CompiledExpr, params: &[f64]) -> Self {
        let (buf, param_count) = copy_inline_params(params);
        Self {
            compiled,
            params: buf,
            param_count,
        }
    }

    /// Borrows the postfix bytecode for $K(r; \theta)$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: exact reference borrow. Domain of validity: valid for lifetime of `self`.
    #[must_use]
    #[verified_engine::verified]
    pub fn compiled(&self) -> &CompiledExpr {
        &self.compiled
    }

    /// Returns active parameter slice $\theta \in \mathbb{R}^p$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: exact slice view. Domain of validity: $0 \le p \le 8$.
    #[must_use]
    #[verified_engine::verified]
    pub fn params(&self) -> &[f64] {
        &self.params[..self.param_count]
    }

    /// Updates inline parameters $\theta$ for $K(r; \theta)$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: replaces non-finite inputs with `0.0`. Domain of validity: truncates inputs beyond `MAX_LENIA_RULE_PARAMS`.
    #[verified_engine::verified]
    pub fn set_params(&mut self, params: &[f64]) {
        let (buf, count) = copy_inline_params(params);
        self.params = buf;
        self.param_count = count;
    }

    /// Evaluates unnormalized $K(r_{\text{norm}}; \theta) \ge 0$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: clamps finite positive values to $\le 10^6$. Domain of validity: $r_{\text{norm}} \in [0, 1]$ (`0.0` outside).
    #[verified_engine::verified]
    pub fn shell_unnormalized(&self, r_norm: f64) -> f64 {
        self.shell_unnormalized_with_params(r_norm, self.params())
    }

    /// Evaluates unnormalized $K(r_{\text{norm}}; \theta) \ge 0$ with explicit $\theta$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: rejects `NaN`/negative weights and caps at $10^6$. Domain of validity: $r_{\text{norm}} \in [0, 1]$.
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

    /// Samples $[r_k, K(r_k)]$ over $r_k = k/(N-1) \in [0, 1]$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: clamps $N \ge 2$ to prevent zero division. Domain of validity: `samples >= 2`.
    #[verified_engine::verified]
    pub fn sample_radial_profile(&self, samples: usize) -> Vec<[f64; 2]> {
        sample_unit_interval(samples, |r| self.shell_unnormalized(r))
    }

    /// Writes unit-$L^1$ torus kernel $\sum_{x,y} K(x, y) = 1$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: impulse fallback when mass $\le 10^{-15}$. Domain of validity: `out.len() >= width * height > 0` and `radius > 0`.
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

    /// Writes unit-$L^1$ torus kernel $\sum_{x,y} K(x, y; \theta) = 1$ with explicit $\theta$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: zero-allocation normalization with impulse fallback. Domain of validity: `out.len() >= width * height > 0` and `radius > 0`.
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

/// Concentric-ring kernel shell $K_S(r) = \beta_{\lfloor B r \rfloor} K_C(B r \bmod 1)$ (`@chan2019lenia`, `spec:biology_v1`).
/// Numerical stability: clamps weights to $[0, 1]$ and enforces $\sum \beta_k > 10^{-12}$. Domain of validity: radius $R \in [4, 32]$ and ring count $B \in \{1, 2, 3, 4\}$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KernelShells {
    radius: f64,
    weights: [f64; MAX_KERNEL_RINGS],
    ring_count: usize,
}

impl KernelShells {
    /// Constructs concentric-ring shell $K_S(r; R, \beta)$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: defaults degenerate zero-sum weights to $\beta_0 = 1$. Domain of validity: clamps $R \in [4, 32]$ and $1 \le B \le 4$ weights in $[0, 1]$.
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

    /// Returns interaction radius $R \in [4, 32]$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: exact scalar copy. Domain of validity: clamped grid-cell radius $[4.0, 32.0]$.
    #[verified_engine::verified]
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Returns active concentric ring count $B \in \{1, 2, 3, 4\}$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: exact integer copy. Domain of validity: $1 \le B \le 4$.
    #[verified_engine::verified]
    pub fn ring_count(&self) -> usize {
        self.ring_count
    }

    /// Returns ring weights $[\beta_0, \dots, \beta_{B-1}] \in [0, 1]^B$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: exact slice view. Domain of validity: $1 \le B \le 4$.
    #[verified_engine::verified]
    pub fn ring_weights(&self) -> &[f64] {
        &self.weights[..self.ring_count]
    }

    /// Evaluates unnormalized shell $K_S(r) = \beta_k K_C(B r - k) \in [0, 1]$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: returns `0.0` at ring boundaries via `kernel_core_bump`. Domain of validity: normalized distance $r_{\text{norm}} \in [0, 1)$.
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

    /// Samples $[r_k, K_S(r_k)]$ over $r_k = k/(N-1) \in [0, 1]$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: clamps $N \ge 2$ to prevent zero division. Domain of validity: `samples >= 2`.
    #[verified_engine::verified]
    pub fn sample_radial_profile(&self, samples: usize) -> Vec<[f64; 2]> {
        sample_unit_interval(samples, |r| self.shell_unnormalized(r))
    }

    /// Writes unit-$L^1$ torus kernel $\sum_{x,y} K(x, y) = 1$ (`@chan2019lenia`, `spec:biology_v1`). Numerical stability: impulse fallback if discrete shell sum $\le 10^{-15}$. Domain of validity: `out.len() >= width * height > 0`.
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
