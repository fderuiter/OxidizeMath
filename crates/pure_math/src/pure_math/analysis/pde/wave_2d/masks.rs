//! Mask builders and localized wave forcing helpers for [`crate::pure_math::analysis::pde::wave_2d::Wave2DSolver`].

// [cite:advanced_linear_algebra]

use math_commons::math_kernel::types::flatten_2d_index;

use super::DriverConfig2D;

/// Evaluates localized spot or line-source forcing acceleration at grid cell `(x, y)`.
#[verified_engine::verified]
pub(super) fn driver_force_at(
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    driver: &DriverConfig2D,
    osc: f64,
) -> f64 {
    assert!(width >= 2 && height >= 2, "Grid must be at least 2x2");
    let cx = driver.x_norm.clamp(0.0, 1.0) * ((width - 1) as f64);
    let r = driver.radius_cells.max(0.75);
    let inv_two_r_sq = 0.5 / (r * r);
    if driver.line_source {
        let dx = (x as f64) - cx;
        let d2 = dx * dx;
        if d2 > 9.0 * r * r {
            return 0.0;
        }
        let edge_pad = 6.min(height / 4);
        if y < edge_pad || y + edge_pad >= height {
            return 0.0;
        }
        driver.amplitude * (-d2 * inv_two_r_sq).exp() * osc
    } else {
        let cy = driver.y_norm.clamp(0.0, 1.0) * ((height - 1) as f64);
        let dx = (x as f64) - cx;
        let dy = (y as f64) - cy;
        let d2 = dx * dx + dy * dy;
        if d2 > 12.0 * r * r {
            return 0.0;
        }
        driver.amplitude * (-d2 * inv_two_r_sq).exp() * osc
    }
}

/// Injects an additive 2D Gaussian displacement pulse into `u_curr` and `u_prev` without allocation.
#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub(super) fn inject_gaussian_pulse_into(
    u_curr: &mut [f64],
    u_prev: &mut [f64],
    wall_mask: &[f64],
    width: usize,
    height: usize,
    cx_norm: f64,
    cy_norm: f64,
    sigma_norm: f64,
    amplitude: f64,
) {
    assert_eq!(u_curr.len(), width * height);
    assert_eq!(u_prev.len(), width * height);
    let sig = sigma_norm.abs().max(1.0e-3);
    let inv_two_sig_sq = 0.5 / (sig * sig);
    let w_denom = (width - 1).max(1) as f64;
    let h_denom = (height - 1).max(1) as f64;
    for y in 1..(height - 1) {
        let yn = (y as f64) / h_denom;
        let dy = yn - cy_norm;
        for x in 1..(width - 1) {
            let idx = flatten_2d_index(x, y, width);
            if wall_mask[idx] > 0.5 {
                let xn = (x as f64) / w_denom;
                let dx = xn - cx_norm;
                let pulse = amplitude * (-(dx * dx + dy * dy) * inv_two_sig_sq).exp();
                u_curr[idx] += pulse;
                u_prev[idx] += pulse;
            }
        }
    }
}

/// Populates `sponge_mask` with a smooth cubic absorbing boundary profile of `thickness` cells.
#[verified_engine::verified]
pub fn apply_sponge_layer(
    sponge_mask: &mut [f64],
    width: usize,
    height: usize,
    thickness: usize,
    max_damping: f64,
) {
    assert_eq!(sponge_mask.len(), width * height);
    sponge_mask.fill(0.0);
    if thickness == 0 || max_damping <= 0.0 {
        return;
    }
    let thick_f = thickness as f64;
    for y in 0..height {
        let dist_y = y.min(height - 1 - y);
        for x in 0..width {
            let dist_x = x.min(width - 1 - x);
            let dist = dist_x.min(dist_y);
            if dist < thickness {
                let s = ((thickness - dist) as f64) / thick_f;
                sponge_mask[flatten_2d_index(x, y, width)] = max_damping * s * s * s;
            }
        }
    }
}

/// Clamps cells outside a centered circular drumhead of normalized radius `radius_norm` (`<= 0.5`).
#[verified_engine::verified]
pub fn apply_circular_drumhead_mask(
    wall_mask: &mut [f64],
    width: usize,
    height: usize,
    radius_norm: f64,
) {
    assert_eq!(wall_mask.len(), width * height);
    let r_lim = radius_norm.clamp(0.05, 0.49);
    let r_sq = r_lim * r_lim;
    let w_denom = (width - 1).max(1) as f64;
    let h_denom = (height - 1).max(1) as f64;
    for y in 0..height {
        let dy = (y as f64) / h_denom - 0.5;
        for x in 0..width {
            let dx = (x as f64) / w_denom - 0.5;
            let idx = flatten_2d_index(x, y, width);
            wall_mask[idx] = if dx * dx + dy * dy <= r_sq { 1.0 } else { 0.0 };
        }
    }
}

/// Places a localized circular nodal clamp (`wall_mask = 0.0`) on a plate at normalized `(cx_norm, cy_norm)`.
#[verified_engine::verified]
pub fn apply_point_clamp(
    wall_mask: &mut [f64],
    width: usize,
    height: usize,
    cx_norm: f64,
    cy_norm: f64,
    radius_cells: f64,
) {
    assert_eq!(wall_mask.len(), width * height);
    let cx = cx_norm.clamp(0.0, 1.0) * ((width - 1) as f64);
    let cy = cy_norm.clamp(0.0, 1.0) * ((height - 1) as f64);
    let r_sq = radius_cells.max(0.5) * radius_cells.max(0.5);
    for y in 0..height {
        let dy = (y as f64) - cy;
        for x in 0..width {
            let dx = (x as f64) - cx;
            if dx * dx + dy * dy <= r_sq {
                wall_mask[flatten_2d_index(x, y, width)] = 0.0;
            }
        }
    }
}

/// Places a vertical barrier wall of `wall_thickness` cells at `barrier_x` pierced by `slit_count`
/// centered apertures of width `slit_width` and center-to-center spacing `slit_separation`.
#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub fn apply_multi_slit_barrier(
    wall_mask: &mut [f64],
    width: usize,
    height: usize,
    barrier_x: usize,
    wall_thickness: usize,
    slit_count: usize,
    slit_width: usize,
    slit_separation: f64,
) {
    assert_eq!(wall_mask.len(), width * height);
    let x0 = barrier_x.min(width - 2);
    let x1 = (x0 + wall_thickness.max(1)).min(width - 1);
    let center_y = ((height - 1) as f64) * 0.5;
    let half_w = (slit_width.max(1) as f64) * 0.5;
    let n_slits = slit_count.clamp(1, 7);
    let span_offset = ((n_slits - 1) as f64) * 0.5;

    for y in 0..height {
        let yf = y as f64;
        let mut in_slit = false;
        for s in 0..n_slits {
            let sy = center_y + ((s as f64) - span_offset) * slit_separation;
            if (yf - sy).abs() <= half_w {
                in_slit = true;
            }
        }
        if !in_slit {
            for x in x0..x1 {
                wall_mask[flatten_2d_index(x, y, width)] = 0.0;
            }
        }
    }
}

/// Populates `wave_speed_mask` with a biconvex acoustic/optical converging lens of `refractive_index`
/// ($s = 1 / n_{\text{lens}}$) centered at `(center_x, height / 2)`.
#[verified_engine::verified]
pub fn apply_biconvex_lens(
    wave_speed_mask: &mut [f64],
    width: usize,
    height: usize,
    center_x: usize,
    half_aperture: f64,
    half_thickness: f64,
    refractive_index: f64,
) {
    assert_eq!(wave_speed_mask.len(), width * height);
    wave_speed_mask.fill(1.0);
    let n_eff = refractive_index.clamp(0.5, 4.0);
    let lens_speed = 1.0 / n_eff;
    let cx = center_x as f64;
    let cy = ((height - 1) as f64) * 0.5;
    let ap = half_aperture.max(2.0);
    let th = half_thickness.max(1.0);

    for y in 0..height {
        let yn = ((y as f64) - cy) / ap;
        if yn.abs() <= 1.0 {
            let local_half_x = th * (1.0 - yn * yn);
            for x in 0..width {
                let dx = ((x as f64) - cx).abs();
                if dx <= local_half_x {
                    wave_speed_mask[flatten_2d_index(x, y, width)] = lens_speed;
                }
            }
        }
    }
}

/// Evaluates the 1D clamped-clamped Euler-Bernoulli beam mode $\psi_m(\xi)$ for $\xi \in [0, 1]$,
/// used to initialize clean 2D square clamped plate eigenmodes $\psi_m(x)\psi_n(y)$.
#[verified_engine::verified]
pub fn clamped_beam_eigenfunction(mode: usize, xi: f64) -> f64 {
    assert!(mode >= 1, "Beam mode index must be >= 1");
    let x = xi.clamp(0.0, 1.0);
    let beta = match mode {
        1 => 4.730_040_744_862_704,
        2 => 7.853_204_624_095_838,
        3 => 10.995_607_838_001_671,
        4 => 14.137_165_491_257_464,
        _ => ((mode as f64) + 0.5) * std::f64::consts::PI,
    };
    let sigma = (beta.cosh() - beta.cos()) / (beta.sinh() - beta.sin());
    let bx = beta * x;
    (bx.cosh() - bx.cos()) - sigma * (bx.sinh() - bx.sin())
}
