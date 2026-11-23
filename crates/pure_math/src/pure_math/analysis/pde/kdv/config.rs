//! Configuration types, conserved invariant structs, and exact analytical soliton formulas
//! for 1D Korteweg-de Vries (KdV) and 1D cubic Nonlinear Schrödinger (NLS) equations.

// [cite:advanced_linear_algebra]

use num_complex::Complex;

/// Maximum periodic grid point count supported on desktop and `wasm32-unknown-unknown`.
pub const MAX_KDV_POINTS: usize = 512;

/// Maximum Integrating-Factor RK4 substeps executed per macroscopic solver step.
pub const MAX_KDV_SUBSTEPS: usize = 256;

/// Target convective Courant-like stability bound for dealiased KdV IFRK4 substeps.
pub const KDV_IFRK4_CFL_TARGET: f64 = 0.40;

/// Governing 1D dispersive wave equation simulated by [`crate::pure_math::analysis::pde::kdv::KdvSolver1D`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KdvEquationMode {
    /// Korteweg-de Vries shallow-water wave equation $u_t + 6 \alpha u u_x + u_{xxx} = 0$
    /// (`alpha = 1` nonlinear soliton regime, `alpha = 0` linear Airy dispersion $u_t + u_{xxx} = 0$).
    KdvSoliton,
    /// 1D cubic Nonlinear Schrödinger equation $i \psi_t + \psi_{xx} + 2 \sigma |\psi|^2 \psi = 0$
    /// (`sigma > 0` focusing bright envelope solitons, `sigma = 0` linear dispersion).
    CubicNls1D,
}

/// Runtime parameters controlling nonlinearity and cubic NLS coupling strength.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct KdvParams {
    /// Active 1D dispersive wave equation.
    pub mode: KdvEquationMode,
    /// Enables nonlinear steepening ($6 u u_x$ in KdV or $2 \sigma |\psi|^2 \psi$ in NLS).
    /// When `false`, evolves pure linear Airy ($u_t + u_{xxx} = 0$) or Schrödinger ($i\psi_t + \psi_{xx} = 0$) dispersion.
    pub nonlinear_enabled: bool,
    /// Focusing ($\sigma > 0$) or defocusing ($\sigma < 0$) cubic coupling for [`KdvEquationMode::CubicNls1D`].
    pub nls_coupling: f64,
}

impl Default for KdvParams {
    fn default() -> Self {
        Self {
            mode: KdvEquationMode::KdvSoliton,
            nonlinear_enabled: true,
            nls_coupling: 1.0,
        }
    }
}

/// Specification for an individual KdV $\operatorname{sech}^2$ soliton pulse.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SolitonSpec {
    /// Propagation speed $c > 0$ (peak amplitude $c / 2$, inverse width $\sqrt{c} / 2$).
    pub speed_c: f64,
    /// Initial center coordinate $x_0 \in [-L/2, L/2)$.
    pub x0: f64,
}

/// First three conserved integrals $(M, P, H)$ of the 1D KdV or cubic NLS hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct SolitonInvariants {
    /// Conserved mass $M = \int u\,dx$ (KdV) or wave norm $M = \int |\psi|^2\,dx$ (NLS).
    pub mass: f64,
    /// Conserved $L^2$ impulse $P = \frac{1}{2}\int u^2\,dx$ (KdV) or momentum $\operatorname{Im}\int \psi^* \psi_x\,dx$ (NLS).
    pub momentum: f64,
    /// Conserved Hamiltonian $H = \int (\frac{1}{2}u_x^2 - \alpha u^3)\,dx$ (KdV) or $\int (|\psi_x|^2 - \sigma |\psi|^4)\,dx$ (NLS).
    pub hamiltonian: f64,
}

impl SolitonInvariants {
    /// Relative drift $(|\Delta M|/|M_0|, |\Delta P|/|P_0|, |\Delta H|/|H_0|)$ against `baseline`.
    #[must_use]
    #[verified_engine::verified]
    pub fn relative_drift(self, baseline: Self) -> [f64; 3] {
        let rel = |now: f64, base: f64| -> f64 {
            let denom = base.abs().max(1.0e-12);
            (now - base).abs() / denom
        };
        [
            rel(self.mass, baseline.mass),
            rel(self.momentum, baseline.momentum),
            rel(self.hamiltonian, baseline.hamiltonian),
        ]
    }
}

/// Exact inverse-scattering spatial phase shifts $(\Delta_1, \Delta_2)$ after an overtaking 2-soliton collision.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct TwoSolitonPhaseShift {
    /// Forward spatial shift $\Delta_1 > 0$ of the faster soliton ($c_{\text{fast}} > c_{\text{slow}}$).
    pub delta_fast: f64,
    /// Backward spatial shift $\Delta_2 < 0$ of the slower soliton.
    pub delta_slow: f64,
}

/// Sub-cell parabolic interpolated wave peak location and amplitude.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DetectedPulsePeak {
    /// Sub-grid spatial coordinate $x \in [-L/2, L/2)$.
    pub x: f64,
    /// Parabolically refined peak value.
    pub amplitude: f64,
}

/// Wraps a spatial displacement $\Delta x$ onto the periodic interval $[-L/2, L/2)$.
#[inline]
#[must_use]
pub fn wrap_periodic_coord(dx: f64, length: f64) -> f64 {
    if !dx.is_finite() || !length.is_finite() || length <= 0.0 {
        return 0.0;
    }
    (dx + 0.5 * length).rem_euclid(length) - 0.5 * length
}

/// Evaluates a periodic 1D KdV single-soliton profile
/// $u(x) = \frac{c}{2} \operatorname{sech}^2\!\left(\frac{\sqrt{c}}{2} (x - x_0)\right)$.
///
/// # Example
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::pde::kdv::kdv_single_soliton_profile;
///
/// let peak = kdv_single_soliton_profile(-4.0, -4.0, 16.0, 40.0);
/// assert!((peak - 8.0).abs() < 1e-12);
/// ```
#[must_use]
#[verified_engine::verified]
pub fn kdv_single_soliton_profile(x: f64, x0: f64, speed_c: f64, length: f64) -> f64 {
    if speed_c <= 0.0 || !speed_c.is_finite() {
        return 0.0;
    }
    let kappa = 0.5 * speed_c.sqrt();
    let dx = wrap_periodic_coord(x - x0, length);
    let arg = (kappa * dx).clamp(-60.0, 60.0);
    let sech = 1.0 / arg.cosh();
    0.5 * speed_c * sech * sech
}

/// Computes the exact inverse-scattering spatial phase shifts $(\Delta_{\text{fast}}, \Delta_{\text{slow}})$
/// for two overtaking KdV solitons with speeds $c_1 \neq c_2 > 0$:
///
/// $$\Delta_{\text{fast}} = +\frac{2}{\sqrt{c_{\text{fast}}}} \ln\!\left(\frac{\sqrt{c_{\text{fast}}} + \sqrt{c_{\text{slow}}}}{\sqrt{c_{\text{fast}}} - \sqrt{c_{\text{slow}}}}\right), \quad
///   \Delta_{\text{slow}} = -\frac{2}{\sqrt{c_{\text{slow}}}} \ln\!\left(\frac{\sqrt{c_{\text{fast}}} + \sqrt{c_{\text{slow}}}}{\sqrt{c_{\text{fast}}} - \sqrt{c_{\text{slow}}}}\right).$$
///
/// # Example
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::pde::kdv::kdv_two_soliton_phase_shifts;
///
/// let shifts = kdv_two_soliton_phase_shifts(16.0, 4.0);
/// assert!((shifts.delta_fast - 0.5493061443340549).abs() < 1e-12);
/// assert!((shifts.delta_slow - (-1.0986122886681098)).abs() < 1e-12);
/// ```
#[must_use]
#[verified_engine::verified]
pub fn kdv_two_soliton_phase_shifts(c1: f64, c2: f64) -> TwoSolitonPhaseShift {
    if c1 <= 0.0 || c2 <= 0.0 || !c1.is_finite() || !c2.is_finite() {
        return TwoSolitonPhaseShift::default();
    }
    let (c_fast, c_slow) = if c1 >= c2 { (c1, c2) } else { (c2, c1) };
    let rt_fast = c_fast.sqrt();
    let rt_slow = c_slow.sqrt();
    let diff = (rt_fast - rt_slow).max(1.0e-12);
    let log_ratio = ((rt_fast + rt_slow) / diff).ln();
    TwoSolitonPhaseShift {
        delta_fast: (2.0 / rt_fast) * log_ratio,
        delta_slow: -(2.0 / rt_slow) * log_ratio,
    }
}

/// Evaluates a focusing 1D cubic NLS bright envelope soliton
/// $\psi(x, 0) = \eta \operatorname{sech}(\eta (x - x_0)) e^{i k_0 (x - x_0)}$ for $i\psi_t + \psi_{xx} + 2|\psi|^2\psi = 0$.
#[must_use]
#[verified_engine::verified]
pub fn nls_bright_soliton_sample(
    x: f64,
    x0: f64,
    amplitude_eta: f64,
    carrier_k: f64,
    length: f64,
) -> Complex<f64> {
    if amplitude_eta <= 0.0 || !amplitude_eta.is_finite() || !carrier_k.is_finite() {
        return Complex::new(0.0, 0.0);
    }
    let dx = wrap_periodic_coord(x - x0, length);
    let env = amplitude_eta / (amplitude_eta * dx).clamp(-60.0, 60.0).cosh();
    let phase = carrier_k * dx;
    Complex::new(env * phase.cos(), env * phase.sin())
}

pub(super) fn build_spectral_tables(n: usize, length: f64) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
    let dk = std::f64::consts::TAU / length;
    let half = n / 2;
    let mut k_deriv = Vec::with_capacity(n);
    let mut k_sq = Vec::with_capacity(n);
    let mut dealias = Vec::with_capacity(n);
    for j in 0..n {
        let signed_m: isize = if j <= half {
            j as isize
        } else {
            (j as isize) - (n as isize)
        };
        let k_full = dk * (signed_m as f64);
        k_sq.push(k_full * k_full);
        k_deriv.push(if j == half { 0.0 } else { k_full });
        dealias.push(3 * signed_m.unsigned_abs() < n && j != half);
    }
    (k_deriv, k_sq, dealias)
}
