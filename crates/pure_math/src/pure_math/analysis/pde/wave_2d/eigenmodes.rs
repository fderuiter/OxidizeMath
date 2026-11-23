//! Zero-allocation 2D Kirchhoff-Love plate and circular Bessel drumhead eigenmode evaluators.
//!
//! Provides boundary-accurate (`Neumann`, `Clamped`, `Dirichlet`) square plate/membrane and
//! circular Dirichlet drumhead eigenfrequencies, pin-compatibility filtering, spatial mode
//! evaluation, antinodal coordinate snapping, and Lorentzian modal response spectra.

// [cite:advanced_linear_algebra]

#[path = "bessel.rs"]
mod bessel;

use std::f64::consts::PI;

use super::config::{BoundaryMode2D, Wave2DMedium, Wave2DParams, analytic_clamped_square_omega};
use super::masks::clamped_beam_eigenfunction;
use bessel::{
    BESSEL_FIRST_PEAK, MAX_BESSEL_ORDER, MAX_BESSEL_RADIAL, distance_to_nearest_pin,
    insert_sorted_mode, mode_survives_pins, snap_bessel_antinode,
};

pub use bessel::{BESSEL_JN_ZEROS, bessel_jn, bessel_jn_zero};

/// Effective staircase Dirichlet boundary offset (in grid cells) for `apply_circular_drumhead_mask`.
const DRUMHEAD_STAIRCASE_CELL_OFFSET: f64 = 0.44;
/// Maximum normalized pin leakage tolerated for an eigenmode to survive nodal clamping.
const PIN_LEAKAGE_TOLERANCE: f64 = 0.08;

/// Nodal point-clamp layout applied to 2D Chladni plates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PinConstraint2D {
    /// No interior nodal pins.
    None,
    /// Single nodal pin at the plate center `(0.5, 0.5)`.
    CenterPin,
    /// Four symmetric pins at `(0.25, 0.25)`, `(0.75, 0.25)`, `(0.25, 0.75)`, `(0.75, 0.75)`.
    QuadSymmetry,
    /// Four perimeter-adjacent pins at `(0.5, 0.04)`, `(0.5, 0.96)`, `(0.04, 0.5)`, `(0.96, 0.5)`.
    EdgeMidpoints,
}

/// Spatial basis family of a 2D standing eigenmode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PlateEigenmodeKind {
    /// Free-edge DCT-I separable mode $\cos(m\pi x)\cos(n\pi y)$.
    SquareNeumannCosine,
    /// Free-edge antisymmetric Chladni combination $\cos(m\pi x)\cos(n\pi y) - \cos(n\pi x)\cos(m\pi y)$.
    SquareChladniAntisymmetric,
    /// Free-edge symmetric Chladni combination $\cos(m\pi x)\cos(n\pi y) + \cos(n\pi x)\cos(m\pi y)$.
    SquareChladniSymmetric,
    /// Simply-supported / fixed-edge DST-I mode $\sin(m\pi x)\sin(n\pi y)$.
    SquareDirichletSine,
    /// Clamped Euler-Bernoulli beam product mode $\psi_m(x)\psi_n(y)$.
    SquareClampedBeam,
    /// Circular Dirichlet membrane Bessel mode $J_m(\alpha_{m,n} r / R)\cos(m\theta)$.
    CircularBesselDrumhead,
}

/// Descriptor for a single discrete-calibrated 2D plate or drumhead eigenmode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eigenmode2D {
    /// Primary modal index $m$ (horizontal half-waves or azimuthal Bessel order).
    pub m: usize,
    /// Secondary modal index $n$ (vertical half-waves or radial Bessel root index $\ge 1$).
    pub n: usize,
    /// Calibrated discrete eigenfrequency $f_{\nu}$ in Hz.
    pub frequency_hz: f64,
    /// Angular eigenfrequency $\omega_{\nu} = 2\pi f_{\nu}$ in rad/s.
    pub omega: f64,
    /// Physical radial wavenumber $k_{\nu}$ in rad per unit domain length.
    pub wavenumber: f64,
    /// Canonical normalized X antinode coordinate in `[0.0, 1.0]`.
    pub antinode_x: f64,
    /// Canonical normalized Y antinode coordinate in `[0.0, 1.0]`.
    pub antinode_y: f64,
    /// Spatial eigenfunction family.
    pub kind: PlateEigenmodeKind,
}

impl Default for Eigenmode2D {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            m: 1,
            n: 1,
            frequency_hz: 1.0,
            omega: 2.0 * PI,
            wavenumber: PI,
            antinode_x: 0.25,
            antinode_y: 0.25,
            kind: PlateEigenmodeKind::SquareNeumannCosine,
        }
    }
}

impl Eigenmode2D {
    /// Radial 2D DFT bin coordinate matching [`super::spectral::SpectralAnalyzer2D::compute_radial_profile`].
    #[must_use]
    #[verified_engine::verified]
    pub fn fft_radial_bin(&self, grid_size: usize, length: f64) -> f64 {
        let denom = grid_size.saturating_sub(1).max(1) as f64;
        (self.wavenumber * length.max(1.0e-9) / (2.0 * PI)) * ((grid_size as f64) / denom)
    }
}

/// Computes the discrete-calibrated square plate/membrane eigenfrequency $f_{m,n}$ (Hz) on a `grid_size x grid_size` domain.
#[must_use]
#[verified_engine::verified]
pub fn square_plate_eigenfrequency_hz(
    mode_m: usize,
    mode_n: usize,
    grid_size: usize,
    length: f64,
    params: &Wave2DParams,
) -> f64 {
    let n_cells = grid_size.max(4);
    let len = length.max(1.0e-6);
    let h = len / ((n_cells - 1) as f64);
    let c_sq = params.wave_speed.max(0.0) * params.wave_speed.max(0.0);
    let d_stiff = if params.medium == Wave2DMedium::BiharmonicPlate {
        params.plate_stiffness.max(0.0)
    } else {
        0.0
    };

    let omega = match params.boundary {
        BoundaryMode2D::Neumann | BoundaryMode2D::Dirichlet | BoundaryMode2D::Sponge => {
            let sm = (0.5 * (mode_m as f64) * PI / ((n_cells - 1) as f64)).sin();
            let sn = (0.5 * (mode_n as f64) * PI / ((n_cells - 1) as f64)).sin();
            let mu = (4.0 / (h * h)) * (sm * sm + sn * sn);
            (c_sq * mu + d_stiff * mu * mu).max(0.0).sqrt()
        }
        BoundaryMode2D::Clamped => {
            let m_eff = mode_m.max(1);
            let n_eff = mode_n.max(1);
            let omega_plate = if d_stiff > 0.0 {
                let raw = analytic_clamped_square_omega(m_eff, n_eff, len, d_stiff);
                let mode_sq = ((m_eff * m_eff + n_eff * n_eff) as f64) * PI * PI;
                let fd_corr = (1.0 - 0.14 * mode_sq * (h / len) * (h / len)).clamp(0.92, 1.0);
                raw * fd_corr
            } else {
                0.0
            };
            let km = ((m_eff as f64) + 0.35) * PI / len;
            let kn = ((n_eff as f64) + 0.35) * PI / len;
            let omega_wave_sq = c_sq * (km * km + kn * kn);
            (omega_plate * omega_plate + omega_wave_sq).sqrt()
        }
    };
    omega / (2.0 * PI)
}

/// Computes the discrete-calibrated circular Dirichlet drumhead eigenfrequency $f_{m,n}$ (Hz).
#[must_use]
#[verified_engine::verified]
pub fn circular_drumhead_eigenfrequency_hz(
    order_m: usize,
    radial_n: usize,
    grid_size: usize,
    length: f64,
    radius_norm: f64,
    params: &Wave2DParams,
) -> f64 {
    let n_cells = grid_size.max(4);
    let len = length.max(1.0e-6);
    let h = len / ((n_cells - 1) as f64);
    let r_phys = (radius_norm.clamp(0.05, 0.49) * len) + DRUMHEAD_STAIRCASE_CELL_OFFSET * h;
    let alpha = bessel_jn_zero(order_m, radial_n.max(1));
    let k = alpha / r_phys;
    let kh = k * h;
    let mu = (k * k) * (1.0 - (kh * kh) / 16.0).clamp(0.85, 1.0);
    let c_sq = params.wave_speed.max(0.0) * params.wave_speed.max(0.0);
    let d_stiff = if params.medium == Wave2DMedium::BiharmonicPlate {
        params.plate_stiffness.max(0.0)
    } else {
        0.0
    };
    (c_sq * mu + d_stiff * mu * mu).max(0.0).sqrt() / (2.0 * PI)
}

/// Evaluates the normalized spatial amplitude $\Phi_{\nu}(x, y) \in [-1, 1]$ of `mode` at `(x_norm, y_norm)`.
#[must_use]
#[verified_engine::verified]
pub fn evaluate_eigenmode_amplitude(
    mode: &Eigenmode2D,
    x_norm: f64,
    y_norm: f64,
    drumhead_radius: f64,
) -> f64 {
    let x = x_norm.clamp(0.0, 1.0);
    let y = y_norm.clamp(0.0, 1.0);
    match mode.kind {
        PlateEigenmodeKind::SquareNeumannCosine => {
            ((mode.m as f64) * PI * x).cos() * ((mode.n as f64) * PI * y).cos()
        }
        PlateEigenmodeKind::SquareChladniAntisymmetric => {
            let a = ((mode.m as f64) * PI * x).cos() * ((mode.n as f64) * PI * y).cos();
            let b = ((mode.n as f64) * PI * x).cos() * ((mode.m as f64) * PI * y).cos();
            0.5 * (a - b)
        }
        PlateEigenmodeKind::SquareChladniSymmetric => {
            let a = ((mode.m as f64) * PI * x).cos() * ((mode.n as f64) * PI * y).cos();
            let b = ((mode.n as f64) * PI * x).cos() * ((mode.m as f64) * PI * y).cos();
            0.5 * (a + b)
        }
        PlateEigenmodeKind::SquareDirichletSine => {
            ((mode.m.max(1) as f64) * PI * x).sin() * ((mode.n.max(1) as f64) * PI * y).sin()
        }
        PlateEigenmodeKind::SquareClampedBeam => {
            let raw = clamped_beam_eigenfunction(mode.m.max(1), x)
                * clamped_beam_eigenfunction(mode.n.max(1), y);
            (raw / 2.52).clamp(-1.0, 1.0)
        }
        PlateEigenmodeKind::CircularBesselDrumhead => {
            let r_lim = drumhead_radius.clamp(0.05, 0.49);
            let dx = x - 0.5;
            let dy = y - 0.5;
            let r = (dx * dx + dy * dy).sqrt();
            if r >= r_lim {
                return 0.0;
            }
            let alpha = bessel_jn_zero(mode.m, mode.n.max(1));
            let peak_arg = BESSEL_FIRST_PEAK[mode.m.min(MAX_BESSEL_ORDER)];
            let peak_val = bessel_jn(mode.m, peak_arg).abs().max(0.25);
            let radial = bessel_jn(mode.m, alpha * (r / r_lim)) / peak_val;
            let theta = dy.atan2(dx);
            (radial * ((mode.m as f64) * theta).cos()).clamp(-1.0, 1.0)
        }
    }
}

/// Snaps `(x_norm, y_norm)` onto a high-amplitude antinode of `mode` while avoiding nodal lines and pins.
#[must_use]
#[verified_engine::verified]
pub fn snap_to_mode_antinode(
    mode: &Eigenmode2D,
    x_norm: f64,
    y_norm: f64,
    pins: PinConstraint2D,
    drumhead_radius: f64,
) -> (f64, f64) {
    if mode.kind == PlateEigenmodeKind::CircularBesselDrumhead {
        return snap_bessel_antinode(mode, x_norm, y_norm, pins, drumhead_radius);
    }
    let mut best_amp = 0.0_f64;
    let mut best_pt = (0.25_f64, 0.25_f64);
    let steps = 25_usize;
    for iy in 0..steps {
        let yn = 0.06 + 0.88 * ((iy as f64) / ((steps - 1) as f64));
        for ix in 0..steps {
            let xn = 0.06 + 0.88 * ((ix as f64) / ((steps - 1) as f64));
            if distance_to_nearest_pin(xn, yn, pins) < 0.06 {
                continue;
            }
            let amp = evaluate_eigenmode_amplitude(mode, xn, yn, drumhead_radius).abs();
            if amp > best_amp {
                best_amp = amp;
                best_pt = (xn, yn);
            }
        }
    }
    let threshold = (0.88 * best_amp).max(0.15);
    let mut min_dist_sq = f64::INFINITY;
    for iy in 0..steps {
        let yn = 0.06 + 0.88 * ((iy as f64) / ((steps - 1) as f64));
        for ix in 0..steps {
            let xn = 0.06 + 0.88 * ((ix as f64) / ((steps - 1) as f64));
            if distance_to_nearest_pin(xn, yn, pins) < 0.06 {
                continue;
            }
            let amp = evaluate_eigenmode_amplitude(mode, xn, yn, drumhead_radius).abs();
            let dist_sq = (xn - x_norm) * (xn - x_norm) + (yn - y_norm) * (yn - y_norm);
            if amp >= threshold && dist_sq < min_dist_sq {
                min_dist_sq = dist_sq;
                best_pt = (xn, yn);
            }
        }
    }
    best_pt
}

/// Populates `out` with pin-compatible 2D eigenmodes sorted by ascending `frequency_hz` without heap allocation.
#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub fn populate_eigenmodes(
    is_circular_drumhead: bool,
    grid_size: usize,
    length: f64,
    drumhead_radius: f64,
    params: &Wave2DParams,
    pins: PinConstraint2D,
    out: &mut [Eigenmode2D],
) -> usize {
    let mut count = 0_usize;
    if is_circular_drumhead {
        let r_phys = drumhead_radius.clamp(0.05, 0.49) * length.max(1.0e-6);
        for m in 0..=MAX_BESSEL_ORDER {
            for n in 1..=MAX_BESSEL_RADIAL {
                let freq = circular_drumhead_eigenfrequency_hz(
                    m,
                    n,
                    grid_size,
                    length,
                    drumhead_radius,
                    params,
                );
                let mut mode = Eigenmode2D {
                    m,
                    n,
                    frequency_hz: freq,
                    omega: 2.0 * PI * freq,
                    wavenumber: bessel_jn_zero(m, n) / r_phys,
                    antinode_x: 0.5,
                    antinode_y: 0.5,
                    kind: PlateEigenmodeKind::CircularBesselDrumhead,
                };
                if mode_survives_pins(&mode, pins, drumhead_radius, PIN_LEAKAGE_TOLERANCE) {
                    let (ax, ay) = snap_bessel_antinode(&mode, 0.34, 0.5, pins, drumhead_radius);
                    mode.antinode_x = ax;
                    mode.antinode_y = ay;
                    insert_sorted_mode(out, &mut count, mode);
                }
            }
        }
        return count;
    }
    populate_square_eigenmodes(
        grid_size,
        length,
        drumhead_radius,
        params,
        pins,
        out,
        &mut count,
    );
    count
}

#[verified_engine::verified]
fn populate_square_eigenmodes(
    grid_size: usize,
    length: f64,
    drumhead_radius: f64,
    params: &Wave2DParams,
    pins: PinConstraint2D,
    out: &mut [Eigenmode2D],
    count: &mut usize,
) {
    let len = length.max(1.0e-6);
    let min_idx = usize::from(params.boundary != BoundaryMode2D::Neumann);
    for m in min_idx..=6_usize {
        for n in min_idx..=m {
            if m == 0 && n == 0 {
                continue;
            }
            let freq = square_plate_eigenfrequency_hz(m, n, grid_size, len, params);
            let k_phys = (PI / len) * (((m * m + n * n) as f64).sqrt());
            if params.boundary == BoundaryMode2D::Neumann && m > n {
                for kind in [
                    PlateEigenmodeKind::SquareChladniAntisymmetric,
                    PlateEigenmodeKind::SquareChladniSymmetric,
                ] {
                    push_checked_square_mode(
                        m,
                        n,
                        freq,
                        k_phys,
                        kind,
                        pins,
                        drumhead_radius,
                        out,
                        count,
                    );
                }
            } else {
                let kind = match params.boundary {
                    BoundaryMode2D::Neumann => PlateEigenmodeKind::SquareNeumannCosine,
                    BoundaryMode2D::Clamped => PlateEigenmodeKind::SquareClampedBeam,
                    BoundaryMode2D::Dirichlet | BoundaryMode2D::Sponge => {
                        PlateEigenmodeKind::SquareDirichletSine
                    }
                };
                push_checked_square_mode(
                    m,
                    n,
                    freq,
                    k_phys,
                    kind,
                    pins,
                    drumhead_radius,
                    out,
                    count,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn push_checked_square_mode(
    m: usize,
    n: usize,
    freq: f64,
    wavenumber: f64,
    kind: PlateEigenmodeKind,
    pins: PinConstraint2D,
    drumhead_radius: f64,
    out: &mut [Eigenmode2D],
    count: &mut usize,
) {
    let mut mode = Eigenmode2D {
        m,
        n,
        frequency_hz: freq,
        omega: 2.0 * PI * freq,
        wavenumber,
        antinode_x: 0.25,
        antinode_y: 0.25,
        kind,
    };
    if mode_survives_pins(&mode, pins, drumhead_radius, PIN_LEAKAGE_TOLERANCE) {
        let (ax, ay) = snap_to_mode_antinode(&mode, 0.25, 0.15, pins, drumhead_radius);
        mode.antinode_x = ax;
        mode.antinode_y = ay;
        insert_sorted_mode(out, count, mode);
    }
}

/// Evaluates the steady-state Lorentzian modal energy spectrum and returns `(response, dominant_index)`.
#[must_use]
#[verified_engine::verified]
pub fn lorentzian_modal_response(
    modes: &[Eigenmode2D],
    freq_hz: f64,
    damping: f64,
    driver_pos: Option<(f64, f64)>,
    drumhead_radius: f64,
) -> (f64, usize) {
    if modes.is_empty() || !freq_hz.is_finite() || freq_hz <= 0.0 {
        return (0.0, 0);
    }
    let omega = 2.0 * PI * freq_hz;
    let gamma = damping.max(0.008);
    let mut total = 0.0_f64;
    let mut best_term = -1.0_f64;
    let mut best_idx = 0_usize;

    for (idx, mode) in modes.iter().enumerate() {
        let coupling = match driver_pos {
            Some((xd, yd)) => evaluate_eigenmode_amplitude(mode, xd, yd, drumhead_radius).abs(),
            None => 1.0,
        };
        let dw = omega * omega - mode.omega * mode.omega;
        let damp_term = gamma * omega.max(mode.omega);
        let denom = (dw * dw + damp_term * damp_term).max(1.0e-12);
        let term = (coupling * coupling * mode.omega * mode.omega) / denom;
        total += term;
        if term > best_term {
            best_term = term;
            best_idx = idx;
        }
    }
    (total, best_idx)
}
