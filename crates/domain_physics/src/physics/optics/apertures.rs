//! Zero-allocation complex aperture, thin-lens, and Zernike wavefront generators.

// [cite:graph_parameters_rust]

use std::f64::consts::PI;

use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex;

/// Configuration for single-slit or multi-slit complex transmission apertures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlitApertureConfig {
    /// Number of parallel vertical slits ($N \ge 1$).
    pub slit_count: usize,
    /// Full opening width of each slit in physical length units.
    pub slit_width: f64,
    /// Center-to-center pitch between adjacent slits in physical length units.
    pub slit_pitch: f64,
    /// Vertical clear height of the slits (`<= 0.0` spans full grid height).
    pub slit_height: f64,
}

#[inline]
#[verified_engine::verified]
fn centered_coord(index: usize, dim: usize, pitch: f64) -> f64 {
    ((index as isize) - ((dim / 2) as isize)) as f64 * pitch
}

/// Fills `out` with an opaque circular disk obstacle ($U = 0$ for $r < a$) illuminated by a
/// unit plane wave, optionally tapered at the outer boundary ($R_{\text{win}} > a$) with a
/// smooth 16th-order super-Gaussian pupil to suppress finite-grid periodic wrap-around.
#[verified_engine::verified]
pub fn fill_opaque_disk_aperture_into(
    out: &mut [Complex<f64>],
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    disk_radius: f64,
    outer_window_radius: f64,
) {
    assert_eq!(out.len(), width * height, "Aperture buffer size mismatch");
    let r_disk = disk_radius.max(0.0);
    let use_window = outer_window_radius > r_disk && outer_window_radius.is_finite();

    for y in 0..height {
        let py = centered_coord(y, height, dy);
        for x in 0..width {
            let px = centered_coord(x, width, dx);
            let r = px.hypot(py);
            let idx = flatten_2d_index(x, y, width);
            let disk_trans = if r <= r_disk { 0.0 } else { 1.0 };
            let env = if use_window {
                let ratio = r / outer_window_radius;
                (-ratio.powi(12)).exp()
            } else {
                1.0
            };
            out[idx] = Complex::new(disk_trans * env, 0.0);
        }
    }
}

/// Fills `out` with a clear circular pupil ($U = 1$ for $r \le R$, $U = 0$ outside).
#[verified_engine::verified]
pub fn fill_circular_pupil_into(
    out: &mut [Complex<f64>],
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    pupil_radius: f64,
) {
    assert_eq!(out.len(), width * height, "Pupil buffer size mismatch");
    let r_max = pupil_radius.max(0.0);
    for y in 0..height {
        let py = centered_coord(y, height, dy);
        for x in 0..width {
            let px = centered_coord(x, width, dx);
            let idx = flatten_2d_index(x, y, width);
            let amp = if px.hypot(py) <= r_max { 1.0 } else { 0.0 };
            out[idx] = Complex::new(amp, 0.0);
        }
    }
}

/// Fills `out` with a single-slit or multi-slit vertical transmission mask centered on the origin.
#[verified_engine::verified]
pub fn fill_slit_aperture_into(
    out: &mut [Complex<f64>],
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    config: SlitApertureConfig,
) {
    assert_eq!(out.len(), width * height, "Slit buffer size mismatch");
    let count = config.slit_count.max(1);
    let half_w = 0.5 * config.slit_width.max(0.0);
    let half_h = 0.5 * config.slit_height;
    let span_offset = 0.5 * ((count - 1) as f64) * config.slit_pitch;

    for y in 0..height {
        let py = centered_coord(y, height, dy);
        let y_open = config.slit_height <= 0.0 || py.abs() <= half_h;
        for x in 0..width {
            let px = centered_coord(x, width, dx);
            let mut open = false;
            if y_open {
                for s in 0..count {
                    let center_x = (s as f64) * config.slit_pitch - span_offset;
                    if (px - center_x).abs() <= half_w {
                        open = true;
                        break;
                    }
                }
            }
            out[flatten_2d_index(x, y, width)] = Complex::new(if open { 1.0 } else { 0.0 }, 0.0);
        }
    }
}

/// Fills `out` with a periodic 1D Ronchi grating of spatial `period` and `duty_cycle` $\in (0, 1)$.
///
/// When `phase_grating` is `false`, transmits binary amplitude $\{0, 1\}$; when `true`,
/// transmits binary $\pi$-phase $\{+1, -1\}$ with unit modulus everywhere.
#[verified_engine::verified]
pub fn fill_ronchi_grating_into(
    out: &mut [Complex<f64>],
    width: usize,
    height: usize,
    dx: f64,
    period: f64,
    duty_cycle: f64,
    phase_grating: bool,
) {
    assert_eq!(out.len(), width * height, "Grating buffer size mismatch");
    let safe_period = period.abs().max(1.0e-12);
    let duty = duty_cycle.clamp(0.05, 0.95);

    for x in 0..width {
        let px = centered_coord(x, width, dx);
        let phase_frac = (px / safe_period + 0.5 * duty).rem_euclid(1.0);
        let high = phase_frac < duty;
        let val = if phase_grating {
            Complex::new(if high { 1.0 } else { -1.0 }, 0.0)
        } else {
            Complex::new(if high { 1.0 } else { 0.0 }, 0.0)
        };
        for y in 0..height {
            out[flatten_2d_index(x, y, width)] = val;
        }
    }
}

/// Fills `out` with a single-harmonic cosinusoidal amplitude grating $U(x, y) = \frac{1}{2}[1 + m \cos(2\pi x / p)]$.
#[verified_engine::verified]
pub fn fill_cosinusoidal_grating_into(
    out: &mut [Complex<f64>],
    width: usize,
    height: usize,
    dx: f64,
    period: f64,
    modulation: f64,
) {
    assert_eq!(out.len(), width * height, "Grating buffer size mismatch");
    let safe_period = period.abs().max(1.0e-12);
    let m = modulation.clamp(0.0, 1.0);
    for x in 0..width {
        let px = centered_coord(x, width, dx);
        let amp = 0.5 * (1.0 + m * (2.0 * PI * px / safe_period).cos());
        let sample = Complex::new(amp, 0.0);
        for y in 0..height {
            out[flatten_2d_index(x, y, width)] = sample;
        }
    }
}

/// Multiplies `field` in-place by a thin quadratic lens transmittance $\exp(-i k (x^2 + y^2) / (2 f))$
/// within `pupil_radius` (zeroing rays outside `pupil_radius` when `pupil_radius > 0.0`).
#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub fn apply_thin_lens_phase_into(
    field: &mut [Complex<f64>],
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    wavelength: f64,
    focal_length: f64,
    pupil_radius: f64,
) {
    assert_eq!(field.len(), width * height, "Lens field size mismatch");
    if !wavelength.is_finite()
        || wavelength <= 0.0
        || !focal_length.is_finite()
        || focal_length.abs() < 1.0e-12
    {
        return;
    }
    let k_over_2f = PI / (wavelength * focal_length);
    for y in 0..height {
        let py = centered_coord(y, height, dy);
        for x in 0..width {
            let px = centered_coord(x, width, dx);
            let r2 = px * px + py * py;
            let idx = flatten_2d_index(x, y, width);
            if pupil_radius > 0.0 && r2 > pupil_radius * pupil_radius {
                field[idx] = Complex::new(0.0, 0.0);
            } else {
                let phi = -k_over_2f * r2;
                field[idx] *= Complex::new(phi.cos(), phi.sin());
            }
        }
    }
}

#[inline]
#[verified_engine::verified]
fn binomial_f64(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k_eff = k.min(n - k);
    let mut acc = 1.0_f64;
    for i in 0..k_eff {
        acc *= ((n - i) as f64) / ((i + 1) as f64);
    }
    acc
}

/// Evaluates the radial Zernike polynomial $R_n^{|m|}(\rho)$ on $\rho \in [0, 1]$ without recursion.
#[must_use]
#[verified_engine::verified]
pub fn zernike_radial(n: usize, m_abs: usize, rho: f64) -> f64 {
    if m_abs > n || !(n - m_abs).is_multiple_of(2) {
        return 0.0;
    }
    let r_clamped = rho.clamp(0.0, 1.0);
    let half_diff = (n - m_abs) / 2;
    let mut sum = 0.0_f64;
    for s in 0..=half_diff {
        let sign = if s.is_multiple_of(2) { 1.0 } else { -1.0 };
        let c1 = binomial_f64(n - s, s);
        let c2 = binomial_f64(n - 2 * s, half_diff - s);
        let power = (n - 2 * s) as i32;
        sum += sign * c1 * c2 * r_clamped.powi(power);
    }
    sum
}

/// Evaluates the unnormalized OSA/ANSI Zernike polynomial $Z_n^m(\rho, \theta)$ on the unit disk.
#[must_use]
#[verified_engine::verified]
pub fn zernike_osa(n: usize, m: isize, rho: f64, theta: f64) -> f64 {
    let m_abs = m.unsigned_abs();
    let radial = zernike_radial(n, m_abs, rho);
    if m > 0 {
        radial * ((m_abs as f64) * theta).cos()
    } else if m < 0 {
        radial * ((m_abs as f64) * theta).sin()
    } else {
        radial
    }
}

/// Multiplies `field` in-place by the wavefront aberration phasor $\exp(i 2\pi w \, Z_n^m(\rho, \theta))$
/// inside `pupil_radius` (in waves `waves_coeff`).
#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub fn apply_zernike_aberration_into(
    field: &mut [Complex<f64>],
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    pupil_radius: f64,
    n: usize,
    m: isize,
    waves_coeff: f64,
) {
    assert_eq!(field.len(), width * height, "Zernike field size mismatch");
    if pupil_radius <= 0.0 || !waves_coeff.is_finite() || waves_coeff.abs() < 1.0e-15 {
        return;
    }
    let inv_r = 1.0 / pupil_radius;
    for y in 0..height {
        let py = centered_coord(y, height, dy);
        for x in 0..width {
            let px = centered_coord(x, width, dx);
            let rho = px.hypot(py) * inv_r;
            if rho <= 1.0 {
                let theta = py.atan2(px);
                let phase = 2.0 * PI * waves_coeff * zernike_osa(n, m, rho, theta);
                let idx = flatten_2d_index(x, y, width);
                field[idx] *= Complex::new(phase.cos(), phase.sin());
            }
        }
    }
}
