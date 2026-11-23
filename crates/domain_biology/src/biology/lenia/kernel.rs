//! Concentric-ring kernel shells $K(r)$ and Gaussian growth curve $G(u; \mu, \sigma)$.

use super::{MAX_KERNEL_RADIUS, MAX_KERNEL_RINGS, MIN_KERNEL_RADIUS};
use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex;
use verified_engine::Theory;

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
        let count = samples.max(2);
        let denom = (count - 1) as f64;
        let mut points = Vec::with_capacity(count);
        for idx in 0..count {
            let u = (idx as f64) / denom;
            points.push([u, self.evaluate(u)]);
        }
        points
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
        let count = samples.max(2);
        let denom = (count - 1) as f64;
        let mut profile = Vec::with_capacity(count);
        for idx in 0..count {
            let r = (idx as f64) / denom;
            profile.push([r, self.shell_unnormalized(r)]);
        }
        profile
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
        let total = width * height;
        if total == 0 || out.len() < total {
            return 0.0;
        }
        let raw_sum = self.populate_unnormalized_torus(width, height, out);
        if raw_sum > 1e-15 {
            let inv_sum = 1.0 / raw_sum;
            for cell in out.iter_mut().take(total) {
                cell.re *= inv_sum;
            }
        } else {
            for cell in out.iter_mut().take(total) {
                *cell = Complex::new(0.0, 0.0);
            }
            out[0] = Complex::new(1.0, 0.0);
        }
        raw_sum
    }

    #[verified_engine::verified]
    fn populate_unnormalized_torus(
        &self,
        width: usize,
        height: usize,
        out: &mut [Complex<f64>],
    ) -> f64 {
        let half_w = width / 2;
        let half_h = height / 2;
        let inv_r = 1.0 / self.radius;
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
                let val = self.shell_unnormalized(r_norm);
                out[flatten_2d_index(x, y, width)] = Complex::new(val, 0.0);
                sum += val;
            }
        }
        sum
    }
}

impl Default for KernelShells {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new(13.0, &[1.0])
    }
}
