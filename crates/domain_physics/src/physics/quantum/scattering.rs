//! # Stationary 1D Transfer-Matrix Scattering & Wavepacket Transmission
//!
//! Zero-allocation unimodular $2 \times 2$ state transfer-matrix propagation across
//! piecewise-constant, double-barrier, sculpted, and bytecode-compiled `V(x)` potentials,
//! paired with 12-point Gauss-Hermite wavepacket transmission expectations.
//!
//! For incident wave number $k > 0$ with energy $E = \frac{\hbar^2 k^2}{2m}$, each constant
//! potential segment $(V_j, \Delta x_j)$ maps $[\psi, \partial_x \psi]^\top$ via
//! $M_j \in \mathrm{SL}(2, \mathbb{R})$, yielding stationary transmission:
//!
//! $$ T(k) = \frac{4}{(M_{00} + M_{11})^2 + \left(k M_{01} - \frac{M_{10}}{k}\right)^2} \in [0, 1] $$
//!
//! ## Example
//!
//! ```rust
//! # use domain_physics::__doctest_facade as math_explorer;
//! use math_explorer::physics::quantum::symmetric_double_barrier_transmission;
//!
//! // Sub-barrier Fabry-Pérot resonant peak on DoubleBarrier (E0 ≈ 10.76 < V0 = 20.0)
//! let t_res = symmetric_double_barrier_transmission(4.6385, 20.0, 1.0, 1.0, 1.0);
//! assert!(t_res > 0.99);
//! ```

// [cite:quantum_mechanics]

use math_commons::ast::CompiledExpr;
use std::f64::consts::FRAC_2_SQRT_PI;

const INV_SQRT_PI: f64 = 0.5 * FRAC_2_SQRT_PI;
const TURNING_POINT_EPS: f64 = 1.0e-14;
const MAX_EVANESCENT_ARG: f64 = 300.0;
const MAX_MATRIX_ENTRY: f64 = 1.0e140;
const POTENTIAL_CLAMP_MIN: f64 = -500.0;
const POTENTIAL_CLAMP_MAX: f64 = 500.0;

/// 12-point Gauss-Hermite nodes $u_j$ and weights $w_j$ for $\int_{-\infty}^{\infty} e^{-u^2} f(u)\,du$.
const GAUSS_HERMITE_12: [(f64, f64); 12] = [
    (-3.889_724_897_869_782, 2.658_551_684_356_302e-7),
    (-3.020_637_025_120_889_6, 8.573_687_043_587_859e-5),
    (-2.279_507_080_501_06, 3.905_390_584_629_061_6e-3),
    (-1.597_682_635_152_604_8, 5.160_798_561_588_393e-2),
    (-0.947_788_391_240_163_8, 2.604_923_102_641_611_4e-1),
    (-0.314_240_376_254_359_13, 5.701_352_362_624_795e-1),
    (0.314_240_376_254_359_13, 5.701_352_362_624_795e-1),
    (0.947_788_391_240_163_8, 2.604_923_102_641_611_4e-1),
    (1.597_682_635_152_604_8, 5.160_798_561_588_393e-2),
    (2.279_507_080_501_06, 3.905_390_584_629_061_6e-3),
    (3.020_637_025_120_889_6, 8.573_687_043_587_859e-5),
    (3.889_724_897_869_782, 2.658_551_684_356_302e-7),
];

/// Real unimodular $2 \times 2$ state transfer matrix $M \in \mathrm{SL}(2, \mathbb{R})$
/// propagating $[\psi(x), \psi'(x)]^\top$ across 1D potential segments.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TransferMatrix2x2 {
    /// Upper-left entry $\partial \psi(x + \Delta x) / \partial \psi(x)$.
    pub m00: f64,
    /// Upper-right entry $\partial \psi(x + \Delta x) / \partial \psi'(x)$.
    pub m01: f64,
    /// Lower-left entry $\partial \psi'(x + \Delta x) / \partial \psi(x)$.
    pub m10: f64,
    /// Lower-right entry $\partial \psi'(x + \Delta x) / \partial \psi'(x)$.
    pub m11: f64,
}

impl Default for TransferMatrix2x2 {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::identity()
    }
}

impl TransferMatrix2x2 {
    /// Returns the $2 \times 2$ identity transfer matrix.
    #[must_use]
    #[verified_engine::verified]
    pub const fn identity() -> Self {
        Self {
            m00: 1.0,
            m01: 0.0,
            m10: 0.0,
            m11: 1.0,
        }
    }

    /// Builds the unimodular propagator across a constant potential slice $(V, \Delta x)$.
    ///
    /// Returns `None` when evanescent decay exceeds numerical range (`\kappa \Delta x > 300`),
    /// signaling zero transmission.
    #[must_use]
    #[verified_engine::verified]
    pub fn constant_slice(
        energy: f64,
        potential: f64,
        dx: f64,
        mass: f64,
        h_bar: f64,
    ) -> Option<Self> {
        if !energy.is_finite()
            || !potential.is_finite()
            || !dx.is_finite()
            || !mass.is_finite()
            || !h_bar.is_finite()
            || mass <= 0.0
            || h_bar <= 0.0
        {
            return None;
        }
        let width = dx.max(0.0);
        if width == 0.0 {
            return Some(Self::identity());
        }
        let q_sq = (2.0 * mass / (h_bar * h_bar)) * (energy - potential);
        if q_sq > TURNING_POINT_EPS {
            let q = q_sq.sqrt();
            let (sin_qx, cos_qx) = (q * width).sin_cos();
            Some(Self {
                m00: cos_qx,
                m01: sin_qx / q,
                m10: -q * sin_qx,
                m11: cos_qx,
            })
        } else if q_sq >= -TURNING_POINT_EPS {
            Some(Self {
                m00: 1.0,
                m01: width,
                m10: 0.0,
                m11: 1.0,
            })
        } else {
            let kappa = (-q_sq).sqrt();
            let arg = kappa * width;
            if arg > MAX_EVANESCENT_ARG {
                return None;
            }
            let cosh_kx = arg.cosh();
            let sinh_kx = arg.sinh();
            Some(Self {
                m00: cosh_kx,
                m01: sinh_kx / kappa,
                m10: kappa * sinh_kx,
                m11: cosh_kx,
            })
        }
    }

    /// Left-multiplies `self` by `next` (`M_total = next * self`), guarding against overflow.
    #[must_use]
    #[verified_engine::verified]
    pub fn compose(self, next: Self) -> Option<Self> {
        let out = Self {
            m00: next.m00 * self.m00 + next.m01 * self.m10,
            m01: next.m00 * self.m01 + next.m01 * self.m11,
            m10: next.m10 * self.m00 + next.m11 * self.m10,
            m11: next.m10 * self.m01 + next.m11 * self.m11,
        };
        let max_abs = out
            .m00
            .abs()
            .max(out.m01.abs())
            .max(out.m10.abs())
            .max(out.m11.abs());
        if max_abs.is_finite() && max_abs <= MAX_MATRIX_ENTRY {
            Some(out)
        } else {
            None
        }
    }

    /// Evaluates stationary transmission $T(k) \in [0, 1]$ for asymptotic free wave number $k > 0$.
    #[must_use]
    #[verified_engine::verified]
    pub fn transmission_for_wavenumber(self, k: f64) -> f64 {
        if !k.is_finite() || k <= 0.0 {
            return 0.0;
        }
        let trace_term = self.m00 + self.m11;
        let mismatch_term = k * self.m01 - self.m10 / k;
        let denom = trace_term * trace_term + mismatch_term * mismatch_term;
        if !denom.is_finite() || denom <= 0.0 {
            0.0
        } else {
            (4.0 / denom).clamp(0.0, 1.0)
        }
    }
}

#[verified_engine::verified]
fn clamp_potential(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else {
        v.clamp(POTENTIAL_CLAMP_MIN, POTENTIAL_CLAMP_MAX)
    }
}

#[verified_engine::verified]
fn lookup_sparse_override(overrides: &[(u16, f64)], index: usize) -> Option<f64> {
    for &(idx, val) in overrides {
        if usize::from(idx) == index {
            return Some(clamp_potential(val));
        }
    }
    None
}

/// Stationary plane-wave transmission $T(k)$ across a single rectangular barrier or well of height `v0` and `width`.
#[must_use]
#[verified_engine::verified]
pub fn rectangular_barrier_transmission(k: f64, v0: f64, width: f64, mass: f64, h_bar: f64) -> f64 {
    if !k.is_finite()
        || k <= 0.0
        || !mass.is_finite()
        || !h_bar.is_finite()
        || mass <= 0.0
        || h_bar <= 0.0
    {
        return 0.0;
    }
    let energy = (h_bar * h_bar * k * k) / (2.0 * mass);
    let Some(mat) =
        TransferMatrix2x2::constant_slice(energy, clamp_potential(v0), width.max(0.0), mass, h_bar)
    else {
        return 0.0;
    };
    mat.transmission_for_wavenumber(k)
}

/// Stationary transmission $T(k)$ across a symmetric double barrier enclosing a central zero-potential well.
#[must_use]
#[verified_engine::verified]
pub fn double_barrier_transmission(
    k: f64,
    v0: f64,
    well_width: f64,
    barrier_thickness: f64,
    mass: f64,
    h_bar: f64,
) -> f64 {
    if !k.is_finite()
        || k <= 0.0
        || !mass.is_finite()
        || !h_bar.is_finite()
        || mass <= 0.0
        || h_bar <= 0.0
    {
        return 0.0;
    }
    let energy = (h_bar * h_bar * k * k) / (2.0 * mass);
    let clamped_v0 = clamp_potential(v0);
    let Some(wall) = TransferMatrix2x2::constant_slice(
        energy,
        clamped_v0,
        barrier_thickness.max(0.0),
        mass,
        h_bar,
    ) else {
        return 0.0;
    };
    let Some(well) =
        TransferMatrix2x2::constant_slice(energy, 0.0, well_width.max(0.0), mass, h_bar)
    else {
        return 0.0;
    };
    wall.compose(well)
        .and_then(|mw| mw.compose(wall))
        .map_or(0.0, |total| total.transmission_for_wavenumber(k))
}

/// Stationary transmission $T(k)$ matching `WavePotentialPreset::DoubleBarrier`
/// (`well_width = width` on `[-width/2, width/2]`, outer barriers of thickness `width / 2`).
#[must_use]
#[verified_engine::verified]
pub fn symmetric_double_barrier_transmission(
    k: f64,
    v0: f64,
    width: f64,
    mass: f64,
    h_bar: f64,
) -> f64 {
    let safe_w = if width.is_finite() {
        width.max(0.0)
    } else {
        0.0
    };
    double_barrier_transmission(k, v0, safe_w, 0.5 * safe_w, mass, h_bar)
}

/// Stationary transmission $T(k)$ across uniform-width staircase potential samples `potential_slices`.
#[must_use]
#[verified_engine::verified]
pub fn piecewise_potential_transmission(
    k: f64,
    potential_slices: &[f64],
    dx: f64,
    mass: f64,
    h_bar: f64,
) -> f64 {
    if !k.is_finite() || k <= 0.0 || !dx.is_finite() || dx < 0.0 || mass <= 0.0 || h_bar <= 0.0 {
        return 0.0;
    }
    if potential_slices.is_empty() || dx == 0.0 {
        return 1.0;
    }
    let energy = (h_bar * h_bar * k * k) / (2.0 * mass);
    let mut total = TransferMatrix2x2::identity();
    for &v_raw in potential_slices {
        let Some(step) =
            TransferMatrix2x2::constant_slice(energy, clamp_potential(v_raw), dx, mass, h_bar)
        else {
            return 0.0;
        };
        let Some(next) = total.compose(step) else {
            return 0.0;
        };
        total = next;
    }
    total.transmission_for_wavenumber(k)
}

/// Zero-allocation transfer-matrix transmission across a bytecode-compiled [`CompiledExpr`] barrier `V(x)`.
///
/// `expr` is evaluated via [`CompiledExpr::eval_1d`] with `params` (e.g. `&[v0, width]`) across
/// `grid = (x_min, x_max, slice_count)` midpoint segments.
#[must_use]
#[verified_engine::verified]
pub fn compiled_barrier_transmission(
    k: f64,
    expr: &CompiledExpr,
    params: &[f64],
    grid: (f64, f64, usize),
    mass: f64,
    h_bar: f64,
) -> f64 {
    compiled_barrier_transmission_with_overrides(k, expr, params, grid, &[], mass, h_bar)
}

/// Zero-allocation transfer-matrix transmission across a [`CompiledExpr`] barrier with optional
/// sparse `(grid_index, potential)` brush overrides across `grid = (x_min, x_max, slice_count)`.
#[must_use]
#[verified_engine::verified]
pub fn compiled_barrier_transmission_with_overrides(
    k: f64,
    expr: &CompiledExpr,
    params: &[f64],
    grid: (f64, f64, usize),
    sparse_overrides: &[(u16, f64)],
    mass: f64,
    h_bar: f64,
) -> f64 {
    let (x_min, x_max, slice_count) = grid;
    if !k.is_finite()
        || k <= 0.0
        || !x_min.is_finite()
        || !x_max.is_finite()
        || x_max <= x_min
        || slice_count == 0
        || mass <= 0.0
        || h_bar <= 0.0
    {
        return 0.0;
    }
    let dx = (x_max - x_min) / (slice_count as f64);
    let energy = (h_bar * h_bar * k * k) / (2.0 * mass);
    let mut total = TransferMatrix2x2::identity();
    for i in 0..slice_count {
        let x_mid = x_min + (i as f64 + 0.5) * dx;
        let v_sample = lookup_sparse_override(sparse_overrides, i)
            .unwrap_or_else(|| clamp_potential(expr.eval_1d(x_mid, params)));
        let Some(step) = TransferMatrix2x2::constant_slice(energy, v_sample, dx, mass, h_bar)
        else {
            return 0.0;
        };
        let Some(next) = total.compose(step) else {
            return 0.0;
        };
        total = next;
    }
    total.transmission_for_wavenumber(k)
}

/// 12-point Gauss-Hermite wavepacket transmission expectation
/// $\langle T \rangle_\sigma(k_0, \sigma) = \pi^{-1/2} \sum_{j=0}^{11} w_j T(\max(k_0 + u_j / \sigma, 0))$.
#[must_use]
#[verified_engine::verified]
pub fn gauss_hermite_wavepacket_transmission<F>(k0: f64, sigma: f64, stationary_t: F) -> f64
where
    F: Fn(f64) -> f64,
{
    if !k0.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
        return 0.0;
    }
    let mut accum = 0.0;
    for &(u, w) in &GAUSS_HERMITE_12 {
        let k_sample = k0 + u / sigma;
        if k_sample > 0.0 {
            let tk = stationary_t(k_sample);
            if tk.is_finite() {
                accum += w * tk.clamp(0.0, 1.0);
            }
        }
    }
    (accum * INV_SQRT_PI).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use math_commons::ast::{CompiledExpr, parse_equation};
    use verified_engine::engine::VerifiedEngine;

    #[test]
    #[verified_engine::verified]
    fn test_rectangular_barrier_matches_closed_form_tunneling_and_ramsauer() {
        // Sub-barrier tunneling: E = 12.5 < V0 = 20.0, w = 0.5
        let k = 5.0;
        let v0 = 20.0;
        let w = 0.5;
        let t_tm = rectangular_barrier_transmission(k, v0, w, 1.0, 1.0);
        let energy = 0.5 * k * k;
        let kappa = (2.0 * (v0 - energy)).sqrt();
        let sinh_term = (kappa * w).sinh();
        let t_exact =
            1.0 / (1.0 + (v0 * v0 * sinh_term * sinh_term) / (4.0 * energy * (v0 - energy)));
        assert!((t_tm - t_exact).abs() < 1.0e-12);

        // Above-barrier Ramsauer-Townsend transparent resonance: q * w = pi => T = 1
        let q = std::f64::consts::PI / w;
        let k_rt = (q * q + 2.0 * 6.0_f64).sqrt();
        let t_rt = rectangular_barrier_transmission(k_rt, 6.0, w, 1.0, 1.0);
        assert!((t_rt - 1.0).abs() < 1.0e-12);
    }

    #[test]
    #[verified_engine::verified]
    fn test_double_barrier_fabry_perot_sub_barrier_resonance() {
        let t_res = symmetric_double_barrier_transmission(4.6385, 20.0, 1.0, 1.0, 1.0);
        let t_off = symmetric_double_barrier_transmission(3.80, 20.0, 1.0, 1.0, 1.0);
        assert!(
            t_res > 0.999,
            "Expected resonant transparency > 99.9%, got {t_res}"
        );
        assert!(
            t_off < 0.03,
            "Expected off-resonant opacity < 3%, got {t_off}"
        );
    }

    #[test]
    fn test_compiled_barrier_and_gauss_hermite_zero_allocations() {
        let (_, ast) = parse_equation("v0 * exp(-(x / width)^4)").expect("valid AST");
        let compiled = CompiledExpr::compile_with_vars_and_params(&ast, &["x"], &["v0", "width"])
            .expect("compiled bytecode");

        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (val, metrics) = VerifiedEngine::run_verified(|| {
            let t_ast = compiled_barrier_transmission(
                5.0,
                &compiled,
                &[11.35, 0.5],
                (-3.0, 3.0, 96),
                1.0,
                1.0,
            );
            let t_wp = gauss_hermite_wavepacket_transmission(4.6385, 0.8, |k| {
                symmetric_double_barrier_transmission(k, 20.0, 1.0, 1.0, 1.0)
            });
            t_ast + t_wp
        });

        assert!(val.is_finite() && val > 0.4);
        assert!(metrics.function_calls > 0);
        assert!(metrics.arithmetic_ops > 0);
    }
}
