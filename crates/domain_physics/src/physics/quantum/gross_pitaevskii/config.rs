//! Configuration types, trapping potentials, conserved Gross-Pitaevskii invariants,
//! and quantized $2\pi$ topological vortex core descriptors.

// [cite:quantum_mechanics]

use std::f64::consts::PI;

/// Maximum 2D condensate grid dimension supported on desktop and `wasm32-unknown-unknown`.
pub const MAX_GPE_DIM: usize = 128;

/// Maximum number of detected vortex cores tracked in the pre-allocated plaquette buffer.
pub const MAX_TRACKED_VORTICES: usize = 256;

/// External trapping potential geometry $V(x, y)$ confining a 2D Bose-Einstein condensate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GpeTrapKind {
    /// Isotropic harmonic trap $V(x, y) = \frac{1}{2} \omega_x^2 (x^2 + y^2)$.
    HarmonicIsotropic,
    /// Anisotropic elliptic harmonic trap $V(x, y) = \frac{1}{2} (\omega_x^2 x^2 + \omega_y^2 y^2)$.
    HarmonicAnisotropic,
    /// Harmonic trap plus a central repulsive Gaussian laser obstacle $V_0 e^{-(x^2 + y^2)/w_0^2}$.
    CentralObstacle,
    /// Uniform periodic box ($V(x, y) = 0$) for free vortex-antivortex dipole translation.
    BoxPeriodic,
}

/// Parameters defining the 2D external trapping potential $V(x, y)$.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GpeTrapConfig {
    /// Trap potential family.
    pub kind: GpeTrapKind,
    /// Horizontal harmonic trap frequency $\omega_x \ge 0$.
    pub omega_x: f64,
    /// Vertical harmonic trap frequency $\omega_y \ge 0$.
    pub omega_y: f64,
    /// Central Gaussian barrier height $V_0 \ge 0$ (used in [`GpeTrapKind::CentralObstacle`]).
    pub obstacle_height: f64,
    /// Central Gaussian barrier $1/e$ radius $w_0 > 0$.
    pub obstacle_width: f64,
}

impl Default for GpeTrapConfig {
    fn default() -> Self {
        Self {
            kind: GpeTrapKind::HarmonicIsotropic,
            omega_x: 1.0,
            omega_y: 1.0,
            obstacle_height: 12.0,
            obstacle_width: 1.1,
        }
    }
}

impl GpeTrapConfig {
    /// Evaluates the external potential $V(x, y)$ at spatial coordinates `(x, y)`.
    #[inline]
    #[must_use]
    pub fn evaluate(&self, x: f64, y: f64) -> f64 {
        match self.kind {
            GpeTrapKind::HarmonicIsotropic => 0.5 * self.omega_x * self.omega_x * (x * x + y * y),
            GpeTrapKind::HarmonicAnisotropic => {
                0.5 * (self.omega_x * self.omega_x * x * x + self.omega_y * self.omega_y * y * y)
            }
            GpeTrapKind::CentralObstacle => {
                let harm = 0.5
                    * (self.omega_x * self.omega_x * x * x + self.omega_y * self.omega_y * y * y);
                let w_sq = (self.obstacle_width * self.obstacle_width).max(1.0e-4);
                let r_sq = x * x + y * y;
                harm + self.obstacle_height * (-r_sq / w_sq).exp()
            }
            GpeTrapKind::BoxPeriodic => 0.0,
        }
    }
}

/// Evolution parameters for 2D Gross-Pitaevskii Strang split-step integration.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GpeParams2D {
    /// Mean-field contact interaction coupling $g$ ($g > 0$ repulsive superfluid).
    pub interaction_g: f64,
    /// Evolves Wick-rotated imaginary time $\tau = i t$ with norm renormalization to relax excitations.
    pub imaginary_time_cooling: bool,
}

impl Default for GpeParams2D {
    fn default() -> Self {
        Self {
            interaction_g: 220.0,
            imaginary_time_cooling: false,
        }
    }
}

/// Conserved dynamical observables $(M, \mathbf{P}, H, \mu)$ of a 2D Gross-Pitaevskii condensate.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct GpeInvariants2D {
    /// Total condensate norm / mass $M = \iint |\psi|^2\,dx\,dy$.
    pub norm: f64,
    /// Horizontal superfluid momentum $P_x = \operatorname{Im}\iint \psi^* \partial_x \psi\,dx\,dy$.
    pub momentum_x: f64,
    /// Vertical superfluid momentum $P_y = \operatorname{Im}\iint \psi^* \partial_y \psi\,dx\,dy$.
    pub momentum_y: f64,
    /// Gross-Pitaevskii energy functional $H = \iint \left(\frac{1}{2}|\nabla\psi|^2 + V|\psi|^2 + \frac{g}{2}|\psi|^4\right)dx\,dy$.
    pub energy: f64,
    /// Chemical potential $\mu = \frac{1}{M}\iint \left(\frac{1}{2}|\nabla\psi|^2 + V|\psi|^2 + g|\psi|^4\right)dx\,dy$.
    pub chemical_potential: f64,
}

impl GpeInvariants2D {
    /// Euclidean linear momentum magnitude $|\mathbf{P}| = \sqrt{P_x^2 + P_y^2}$.
    #[inline]
    #[must_use]
    pub fn momentum_magnitude(self) -> f64 {
        self.momentum_x.hypot(self.momentum_y)
    }
}

/// Quantized superfluid vortex core detected on a $2\times 2$ lattice plaquette.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct VortexCore2D {
    /// Sub-cell interpolated horizontal coordinate $x$.
    pub x: f64,
    /// Sub-cell interpolated vertical coordinate $y$.
    pub y: f64,
    /// Lower-left plaquette column index $i \in [0, W - 2]$.
    pub grid_x: usize,
    /// Lower-left plaquette row index $j \in [0, H - 2]$.
    pub grid_y: usize,
    /// Topological winding charge $q \in \mathbb{Z} \setminus \{0\}$ (`+1` or `-1` for elementary vortices).
    pub charge: i32,
    /// Quantized phase circulation $\oint \nabla \theta \cdot d\mathbf{l} = 2\pi q$.
    pub circulation: f64,
}

/// Aggregate topological vortex charge summary across the condensate bulk.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct VortexChargeSummary {
    /// Count of positively charged ($q > 0$, $+2\pi$) vortices.
    pub positive_count: usize,
    /// Count of negatively charged ($q < 0$, $-2\pi$) antivortices.
    pub negative_count: usize,
    /// Net topological winding charge $Q = \sum_k q_k$.
    pub net_charge: i32,
    /// Net quantized phase circulation $2\pi Q$.
    pub net_circulation: f64,
}

/// Computes the 2D Thomas-Fermi chemical potential $\mu_{\text{TF}} = \sqrt{g N \omega_x \omega_y / \pi}$
/// for a harmonically trapped repulsive condensate.
#[inline]
#[must_use]
pub fn thomas_fermi_chemical_potential_2d(
    interaction_g: f64,
    target_norm: f64,
    omega_x: f64,
    omega_y: f64,
) -> f64 {
    let prod = interaction_g.max(0.0) * target_norm.max(0.0) * omega_x.max(0.1) * omega_y.max(0.1);
    (prod / PI).sqrt().max(0.5 * (omega_x + omega_y))
}

pub(super) fn clamp_even_dim(dim: usize) -> usize {
    let mut n = dim.clamp(8, MAX_GPE_DIM);
    if !n.is_multiple_of(2) {
        n -= 1;
    }
    n
}

pub(super) fn sanitize_domain_len(length: f64) -> f64 {
    if length.is_finite() && length > 1.0 {
        length
    } else {
        20.0
    }
}

pub(super) fn build_1d_wavenumbers(n: usize, length: f64) -> (Vec<f64>, Vec<f64>) {
    let dk = std::f64::consts::TAU / length;
    let half = n / 2;
    let mut k_deriv = Vec::with_capacity(n);
    let mut k_sq = Vec::with_capacity(n);
    for i in 0..n {
        let signed_m: isize = if i <= half {
            i as isize
        } else {
            (i as isize) - (n as isize)
        };
        let k_val = dk * (signed_m as f64);
        k_sq.push(k_val * k_val);
        k_deriv.push(if i == half { 0.0 } else { k_val });
    }
    (k_deriv, k_sq)
}
