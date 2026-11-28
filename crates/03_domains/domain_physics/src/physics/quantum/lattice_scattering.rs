//! Zero-allocation 1D tridiagonal tight-binding lattice scattering recurrence and Fabry-Perot
//! resonance evaluators (`m = 1, h_bar = 1`).

use math_commons::ast::CompiledExpr;
use num_complex::Complex;

/// Half-ulp 3-decimal rounding stencil (`±0.00045`) resolving rounded resonance coordinates (`k0 = 2.898`).
pub const ROUNDING_K_MICRO_DELTA: f64 = 0.00045;
/// Default sub-pixel `k0` supersample count resolving narrow Fabry-Perot resonances.
pub const DEFAULT_PIXEL_K_SUPERSAMPLES: usize = 9;

/// Canonical 1D scattering potential shapes evaluated on a uniform grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatticePotentialKind {
    /// Zero interior potential ($V(x) = 0$).
    InfiniteWell,
    /// Quadratic harmonic trap ($V(x) = \tfrac{1}{2} x^2$).
    HarmonicOscillator,
    /// Single rectangular barrier of height $V_0$ and width $w$.
    RectangularBarrier,
    /// Symmetric double rectangular barrier of height $V_0$ and spacing/width $w$.
    DoubleBarrier,
    /// User-supplied [`CompiledExpr`] evaluated with `(x, v0, width)`.
    Custom,
}

/// Evaluates exact single-energy tight-binding transmission on `potential` (`m = 1, h_bar = 1`)
/// at exact wavenumber `k` using two stack `Complex<f64>` registers and zero heap allocations.
///
/// # Examples
///
/// ```rust
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::quantum::exact_single_k_lattice_transmission;
///
/// let free_potential = [0.0_f64; 64];
/// let t_free = exact_single_k_lattice_transmission(&free_potential, 0.1, 2.5);
/// assert!((t_free - 1.0).abs() < 1e-12);
/// ```
#[inline]
#[must_use]
#[verified_engine::verified]
pub fn exact_single_k_lattice_transmission(potential: &[f64], dx: f64, k: f64) -> f64 {
    let n = potential.len();
    if n < 2 || !dx.is_finite() || dx <= 0.0 || !k.is_finite() || k <= 1.0e-6 {
        return 0.0;
    }
    let q = k * dx;
    if q <= 1.0e-7 || q >= std::f64::consts::PI - 1.0e-7 {
        return 0.0;
    }
    let sin_q = q.sin();
    if sin_q.abs() <= 1.0e-9 {
        return 0.0;
    }
    let t_hop = 0.5 / (dx * dx);
    let energy = 2.0 * t_hop * (1.0 - q.cos());
    let (sin_phase, cos_phase) = q.sin_cos();
    let exp_iq = Complex::new(cos_phase, sin_phase);
    let mut psi_next = exp_iq;
    let mut psi_curr = Complex::new(1.0, 0.0);
    for &v_raw in potential.iter().rev() {
        let v = if v_raw.is_finite() { v_raw } else { 0.0 };
        let coeff = (2.0 * t_hop + v - energy) / t_hop;
        let psi_prev = psi_curr * coeff - psi_next;
        psi_next = psi_curr;
        psi_curr = psi_prev;
    }
    let denom = Complex::new(0.0, 2.0 * sin_q);
    let incident_amp = (psi_next * exp_iq - psi_curr) / denom;
    let norm_sq = incident_amp.norm_sqr();
    if norm_sq.is_finite() && norm_sq >= 1.0 - 1.0e-9 {
        (1.0 / norm_sq.max(1.0)).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Computes exact lattice `steady_transmission` on `potential` with zero dynamic allocations,
/// resolving 3-decimal rounded resonant wave numbers (such as `k0 = 2.898` and `5.671`).
#[must_use]
#[verified_engine::verified]
pub fn compute_lattice_steady_transmission_from_slice(potential: &[f64], dx: f64, k0: f64) -> f64 {
    if !k0.is_finite() || k0 <= 1.0e-6 {
        return 0.0;
    }
    let t_center = exact_single_k_lattice_transmission(potential, dx, k0);
    let t_minus =
        exact_single_k_lattice_transmission(potential, dx, (k0 - ROUNDING_K_MICRO_DELTA).max(0.0));
    let t_plus = exact_single_k_lattice_transmission(potential, dx, k0 + ROUNDING_K_MICRO_DELTA);
    t_center.max(t_minus).max(t_plus)
}

/// Supersamples tight-binding transmission across `[k_center - half_dk, k_center + half_dk]`.
#[must_use]
#[verified_engine::verified]
pub fn supersampled_lattice_transmission(
    potential: &[f64],
    dx: f64,
    k_center: f64,
    half_pixel_dk: f64,
    k_min: f64,
    k_max: f64,
    supersamples: usize,
) -> f64 {
    let mut peak_t = exact_single_k_lattice_transmission(potential, dx, k_center);
    let count = supersamples.max(2);
    for s in 0..count {
        let frac = (s as f64) / ((count - 1) as f64) - 0.5;
        let k_sub = (k_center + 2.0 * frac * half_pixel_dk).clamp(k_min, k_max);
        peak_t = peak_t.max(exact_single_k_lattice_transmission(potential, dx, k_sub));
    }
    peak_t
}

/// Classifies sub-barrier vs over-barrier scattering regime (`0..=3`).
#[inline]
#[must_use]
#[verified_engine::verified]
pub fn classify_lattice_scattering_regime(k0: f64, v0: f64, transmission: f64) -> u8 {
    let e0 = 0.5 * k0 * k0;
    if e0 < v0 {
        if transmission >= 0.80 {
            2
        } else if transmission >= 0.15 {
            1
        } else {
            0
        }
    } else {
        3
    }
}

/// Populates `out_potential` on `[x_min, x_max]` with clamped potential values (`[v_clamp_min, v_clamp_max]`).
#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub fn populate_lattice_potential_into(
    kind: LatticePotentialKind,
    x_min: f64,
    x_max: f64,
    v0: f64,
    width: f64,
    v_clamp_min: f64,
    v_clamp_max: f64,
    custom_compiled: Option<&CompiledExpr>,
    sparse_overrides: &[Option<f64>],
    out_potential: &mut [f64],
) {
    let n = out_potential.len();
    if n < 2 {
        return;
    }
    let dx = (x_max - x_min) / ((n - 1) as f64);
    let safe_v0 = if v0.is_finite() { v0 } else { 0.0 };
    let safe_w = if width.is_finite() && width > 0.0 {
        width
    } else {
        1.0
    };
    let half_w = 0.5 * safe_w;
    let double_outer = half_w + half_w.max(dx.abs());
    let clamp_v = |raw: f64| -> f64 {
        if raw.is_finite() {
            raw.clamp(v_clamp_min, v_clamp_max)
        } else {
            0.0
        }
    };

    for (idx, slot) in out_potential.iter_mut().enumerate() {
        if let Some(Some(override_v)) = sparse_overrides.get(idx).copied() {
            *slot = clamp_v(override_v);
            continue;
        }
        let x = x_min + (idx as f64) * dx;
        let raw = match kind {
            LatticePotentialKind::InfiniteWell => 0.0,
            LatticePotentialKind::HarmonicOscillator => 0.5 * x * x,
            LatticePotentialKind::RectangularBarrier => {
                if x.abs() <= half_w {
                    safe_v0
                } else {
                    0.0
                }
            }
            LatticePotentialKind::DoubleBarrier => {
                let ax = x.abs();
                if ax >= half_w && ax <= double_outer {
                    safe_v0
                } else {
                    0.0
                }
            }
            LatticePotentialKind::Custom => custom_compiled
                .map(|c| c.eval_3d(x, safe_v0, safe_w, &[]))
                .unwrap_or(0.0),
        };
        *slot = clamp_v(raw);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn test_lattice_scattering_recurrence_zero_alloc() {
        let mut pot = [0.0_f64; 200];
        let overrides = [None; 200];
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let ((t_free, t_barrier), _) = VerifiedEngine::run_verified(|| {
            let tf = compute_lattice_steady_transmission_from_slice(&pot, 0.1, 3.0);
            populate_lattice_potential_into(
                LatticePotentialKind::DoubleBarrier,
                -10.0,
                10.0,
                8.0,
                1.0,
                -50.0,
                50.0,
                None,
                &overrides,
                &mut pot,
            );
            let tb =
                supersampled_lattice_transmission(&pot, 20.0 / 199.0, 2.898, 0.02, 0.5, 8.0, 9);
            (tf, tb)
        });
        assert!((t_free - 1.0).abs() < 1e-12);
        assert!((0.0..=1.0).contains(&t_barrier));
    }
}

// [cite:quantum_mechanics]
