//! Conservative 2D donor-cell upwind advection, dimensionless Courant and Péclet
//! diagnostics, and combined advective-diffusive explicit CFL stability bounds.

use super::fused_stepper::StencilOperators;
use math_commons::math_kernel::types::{StepSize, flatten_2d_index};

/// Evaluates the conservative donor-cell flux across a single interface from the
/// upstream (`c_minus`) to the downstream (`c_plus`) cell given normal velocity `u_face`.
#[inline(always)]
#[verified_engine::verified]
pub fn donor_cell_face_flux(u_face: f64, c_minus: f64, c_plus: f64) -> f64 {
    if u_face >= 0.0 {
        u_face * c_minus
    } else {
        u_face * c_plus
    }
}

/// Evaluates the 2D conservative donor-cell upwind advection tendency
/// $-\nabla \cdot (\mathbf{u} c)$ at a cell given neighbor concentrations
/// `[c_left, c_right, c_up, c_down]` and interface velocities
/// `[u_west, u_east, v_north, v_south]`.
#[inline(always)]
#[verified_engine::verified]
pub fn upwind_advection_cell_2d(
    c_curr: f64,
    c_neighbors: [f64; 4],
    face_vel: [f64; 4],
    inv_dx: f64,
    inv_dy: f64,
) -> f64 {
    let [c_left, c_right, c_up, c_down] = c_neighbors;
    let [u_west, u_east, v_north, v_south] = face_vel;
    let flux_west = donor_cell_face_flux(u_west, c_left, c_curr);
    let flux_east = donor_cell_face_flux(u_east, c_curr, c_right);
    let flux_north = donor_cell_face_flux(v_north, c_up, c_curr);
    let flux_south = donor_cell_face_flux(v_south, c_curr, c_down);
    -(flux_east - flux_west) * inv_dx - (flux_south - flux_north) * inv_dy
}

#[inline(always)]
fn safe_face_avg(va: f64, vb: f64) -> f64 {
    let a = if va.is_finite() { va } else { 0.0 };
    let b = if vb.is_finite() { vb } else { 0.0 };
    0.5 * (a + b)
}

/// Evaluates 2D conservative donor-cell upwind advection $-\nabla \cdot (\mathbf{u} c)$
/// over a row-major `width x height` scalar grid into `out` with zero-normal-flux outer
/// walls ($\mathbf{u} \cdot \mathbf{n} = 0$ on domain boundaries) and zero heap allocations.
#[verified_engine::verified]
pub fn upwind_advection_2d(
    dim: (usize, usize),
    src: &[f64],
    ux: &[f64],
    uy: &[f64],
    out: &mut [f64],
    ops: &StencilOperators,
) {
    let (width, height) = dim;
    let total = width.saturating_mul(height);
    if total == 0 {
        return;
    }
    assert!(
        src.len() >= total && ux.len() >= total && uy.len() >= total && out.len() >= total,
        "upwind_advection_2d buffer length mismatch for {}x{} grid",
        width,
        height
    );

    for y in 0..height {
        for x in 0..width {
            let idx = flatten_2d_index(x, y, width);
            let c_curr = src[idx];
            let (c_left, u_west) = if x > 0 {
                let il = idx - 1;
                (src[il], safe_face_avg(ux[il], ux[idx]))
            } else {
                (c_curr, 0.0)
            };
            let (c_right, u_east) = if x + 1 < width {
                let ir = idx + 1;
                (src[ir], safe_face_avg(ux[idx], ux[ir]))
            } else {
                (c_curr, 0.0)
            };
            let (c_up, v_north) = if y > 0 {
                let iu = idx - width;
                (src[iu], safe_face_avg(uy[iu], uy[idx]))
            } else {
                (c_curr, 0.0)
            };
            let (c_down, v_south) = if y + 1 < height {
                let id = idx + width;
                (src[id], safe_face_avg(uy[idx], uy[id]))
            } else {
                (c_curr, 0.0)
            };

            out[idx] = upwind_advection_cell_2d(
                c_curr,
                [c_left, c_right, c_up, c_down],
                [u_west, u_east, v_north, v_south],
                ops.inv_dx,
                ops.inv_dy,
            );
        }
    }
}

/// Computes the 2D directional advective Courant number
/// $\text{Co} = \Delta t \left(\frac{|u_x|_{\max}}{\Delta x} + \frac{|u_y|_{\max}}{\Delta y}\right)$.
#[inline]
#[verified_engine::verified]
pub fn advection_courant_number(
    dt: f64,
    dx: StepSize,
    dy: StepSize,
    max_abs_ux: f64,
    max_abs_uy: f64,
) -> f64 {
    if !dt.is_finite()
        || dt <= 0.0
        || !dx.is_finite()
        || dx.get() <= 0.0
        || !dy.is_finite()
        || dy.get() <= 0.0
        || !max_abs_ux.is_finite()
        || !max_abs_uy.is_finite()
    {
        return 0.0;
    }
    dt * (max_abs_ux.abs() / dx.get() + max_abs_uy.abs() / dy.get())
}

/// Computes the maximum explicit Forward Euler time step $\Delta t_{\max}$ satisfying
/// $\Delta t \left(2 D_{\max}(\Delta x^{-2} + \Delta y^{-2}) + |u_x|_{\max}\Delta x^{-1} + |u_y|_{\max}\Delta y^{-1}\right) < 1$.
///
/// For isotropic spacing $\Delta x = \Delta y = h$, this equals
/// $\left(4 D_{\max} / h^2 + (|u_x|_{\max} + |u_y|_{\max}) / h\right)^{-1}$.
#[inline]
#[verified_engine::verified]
pub fn advection_diffusion_cfl_limit(
    dx: StepSize,
    dy: StepSize,
    d_max: f64,
    max_abs_ux: f64,
    max_abs_uy: f64,
) -> f64 {
    if !dx.is_finite()
        || dx.get() <= 0.0
        || !dy.is_finite()
        || dy.get() <= 0.0
        || !d_max.is_finite()
        || !max_abs_ux.is_finite()
        || !max_abs_uy.is_finite()
    {
        return 0.0;
    }
    let inv_dx = 1.0 / dx.get();
    let inv_dy = 1.0 / dy.get();
    let diff_rate = 2.0 * d_max.max(0.0) * (inv_dx * inv_dx + inv_dy * inv_dy);
    let adv_rate = max_abs_ux.abs() * inv_dx + max_abs_uy.abs() * inv_dy;
    let denom = diff_rate + adv_rate;
    if denom <= 0.0 {
        f64::INFINITY
    } else {
        1.0 / denom
    }
}

/// Computes the dimensionless grid Péclet number $\text{Pe}_g = \frac{|\mathbf{u}| \, h}{\max(D, 10^{-12})}$.
#[inline]
#[verified_engine::verified]
pub fn grid_peclet_number(speed: f64, grid_spacing: f64, diffusivity: f64) -> f64 {
    if !speed.is_finite()
        || !grid_spacing.is_finite()
        || !diffusivity.is_finite()
        || grid_spacing <= 0.0
    {
        return 0.0;
    }
    (speed.abs() * grid_spacing) / diffusivity.max(1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anisotropic_upwind_and_zero_velocity() {
        let (w, h) = (6, 5);
        let ops = StencilOperators::new_2d(StepSize(0.5), StepSize(0.25));
        let mut src = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                src[flatten_2d_index(x, y, w)] = 2.0 * (x as f64 * 0.5) + 3.0 * (y as f64 * 0.25);
            }
        }
        let zero_u = vec![0.0; w * h];
        let mut out = vec![1.0; w * h];
        upwind_advection_2d((w, h), &src, &zero_u, &zero_u, &mut out, &ops);
        assert!(out.iter().all(|&v| v.abs() < 1e-12));

        let ux = vec![1.5; w * h];
        let uy = vec![-0.8; w * h];
        upwind_advection_2d((w, h), &src, &ux, &uy, &mut out, &ops);
        // Interior cell (2, 2): -ux * dc/dx - uy * dc/dy = -1.5 * 2.0 - (-0.8) * 3.0 = -0.6
        let interior = out[flatten_2d_index(2, 2, w)];
        assert!((interior - (-0.6)).abs() < 1e-12, "got {interior}");
    }

    #[test]
    fn test_gaussian_pulse_translation_mass_conservation_and_zero_alloc() {
        let (w, h) = (32, 16);
        let ops = StencilOperators::new_2d(StepSize(1.0), StepSize(1.0));
        let mut c = vec![0.0; w * h];
        let mut tendency = vec![0.0; w * h];
        let ux = vec![0.12; w * h];
        let uy = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let dx = x as f64 - 8.0;
                let dy = y as f64 - 8.0;
                c[flatten_2d_index(x, y, w)] = (-0.15 * (dx * dx + dy * dy)).exp();
            }
        }
        let initial_mass: f64 = c.iter().sum();
        let centroid_x = |field: &[f64]| -> f64 {
            let m: f64 = field.iter().sum();
            let mut wx = 0.0;
            for y in 0..h {
                for x in 0..w {
                    wx += (x as f64) * field[flatten_2d_index(x, y, w)];
                }
            }
            wx / m
        };
        let x0 = centroid_x(&c);

        let ((), _metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            for _ in 0..40 {
                upwind_advection_2d((w, h), &c, &ux, &uy, &mut tendency, &ops);
                for i in 0..(w * h) {
                    c[i] += 1.0 * tendency[i];
                }
            }
        });
        let final_mass: f64 = c.iter().sum();
        let x1 = centroid_x(&c);
        assert!((final_mass - initial_mass).abs() < 1e-11);
        assert!(x1 > x0 + 3.5, "Expected downstream drift: {x0} -> {x1}");
    }

    #[test]
    fn test_cfl_limit_courant_and_peclet_helpers() {
        let limit = advection_diffusion_cfl_limit(StepSize(1.0), StepSize(1.0), 0.16, 0.12, 0.04);
        let expected = 1.0 / (4.0 * 0.16 + 0.16);
        assert!((limit - expected).abs() < 1e-12);
        let co = advection_courant_number(0.5, StepSize(1.0), StepSize(2.0), 0.12, 0.08);
        assert!((co - 0.5 * (0.12 + 0.04)).abs() < 1e-12);
        let pe = grid_peclet_number(0.12, 1.0, 0.08);
        assert!((pe - 1.5).abs() < 1e-12);
        assert_eq!(grid_peclet_number(f64::NAN, 1.0, 0.1), 0.0);
        assert_eq!(
            advection_diffusion_cfl_limit(StepSize(1.0), StepSize(1.0), f64::NAN, 0.1, 0.0),
            0.0
        );
        assert_eq!(
            advection_courant_number(0.5, StepSize::new(0.0), StepSize::new(1.0), 0.1, 0.1),
            0.0
        );
    }
}

// [cite:graph_parameters_rust]
