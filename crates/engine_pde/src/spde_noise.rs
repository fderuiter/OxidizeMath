// [cite:engine_pde]
use crate::boundary::BoundaryCondition;
use math_commons::math_kernel::types::{
    Scalar, SolverWorkspace, fill_standard_normal, flatten_2d_index,
};
use rand_core::RngCore;

/// Spatio-temporal stochastic forcing structure for continuum SPDE steppers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpdeNoiseKind {
    /// Deterministic evolution without stochastic forcing.
    None,
    /// Non-conserved additive white noise $\sqrt{2 D_{\text{noise}} \Delta t / \Delta V}\,\xi_i$
    /// (Model A / Allen-Cahn / Edwards-Wilkinson / KPZ).
    Additive {
        /// Noise variance intensity $D_{\text{noise}} \ge 0$.
        intensity: f64,
    },
    /// Conserved divergence-of-flux noise $\nabla \cdot \left(\sqrt{2 M k_B T \Delta t / \Delta V}\,\vec{\xi}\right)$
    /// discretized on staggered cell faces so total mass $\sum_i u_i$ is conserved to machine precision
    /// (Model B / Cahn-Hilliard).
    ConservedFlux {
        /// Fluctuation-dissipation product $M k_B T \ge 0$.
        mobility_temp: f64,
    },
}

/// Injects non-conserved 1D additive Gaussian white noise into `field` using `workspace` and `rng`.
#[verified_engine::verified]
pub fn inject_additive_noise_1d<R: RngCore + ?Sized>(
    field: &mut [f64],
    dx: f64,
    dt: f64,
    intensity: f64,
    bc: BoundaryCondition,
    workspace: &mut SolverWorkspace<'_, f64>,
    rng: &mut R,
) {
    let n = field.len();
    assert!(dx > 0.0 && dt >= 0.0);
    if n == 0 || intensity <= 0.0 || dt <= 0.0 {
        return;
    }
    let scale = <f64 as Scalar>::sqrt(2.0 * intensity * dt / dx);
    let mut ws = workspace.reborrow();
    let z = ws.take_slice(n);
    fill_standard_normal(z, rng);
    for i in 0..n {
        if let BoundaryCondition::Dirichlet(val) = bc
            && (i == 0 || i + 1 == n)
        {
            field[i] = val;
            continue;
        }
        field[i] += scale * z[i];
    }
}

/// Injects non-conserved 2D additive Gaussian white noise into `field` (`width x height`).
#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub fn inject_additive_noise_2d<R: RngCore + ?Sized>(
    field: &mut [f64],
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    dt: f64,
    intensity: f64,
    bc: BoundaryCondition,
    workspace: &mut SolverWorkspace<'_, f64>,
    rng: &mut R,
) {
    let total = width * height;
    assert!(field.len() >= total && dx > 0.0 && dy > 0.0 && dt >= 0.0);
    if total == 0 || intensity <= 0.0 || dt <= 0.0 {
        return;
    }
    let scale = <f64 as Scalar>::sqrt(2.0 * intensity * dt / (dx * dy));
    let mut ws = workspace.reborrow();
    let z = ws.take_slice(total);
    fill_standard_normal(z, rng);
    for y in 0..height {
        for x in 0..width {
            let idx = flatten_2d_index(x, y, width);
            if let BoundaryCondition::Dirichlet(val) = bc
                && (x == 0 || x + 1 == width || y == 0 || y + 1 == height)
            {
                field[idx] = val;
                continue;
            }
            field[idx] += scale * z[idx];
        }
    }
}

/// Injects 1D staggered-face conserved-flux noise $\Delta t \, \partial_x \eta$ into `field`,
/// strictly preserving $\sum_i u_i$ under Periodic and Neumann boundaries.
#[verified_engine::verified]
pub fn inject_conserved_flux_noise_1d<R: RngCore + ?Sized>(
    field: &mut [f64],
    dx: f64,
    dt: f64,
    mobility_temp: f64,
    bc: BoundaryCondition,
    workspace: &mut SolverWorkspace<'_, f64>,
    rng: &mut R,
) {
    let n = field.len();
    assert!(dx > 0.0 && dt >= 0.0);
    if n < 2 || mobility_temp <= 0.0 || dt <= 0.0 {
        return;
    }
    let flux_scale = <f64 as Scalar>::sqrt(2.0 * mobility_temp * dt / dx) / dx;
    let mut ws = workspace.reborrow();
    let faces = ws.take_slice(n);
    fill_standard_normal(faces, rng);
    for f in faces.iter_mut() {
        *f *= flux_scale;
    }
    if bc != BoundaryCondition::Periodic {
        faces[n - 1] = 0.0;
    }
    for i in 0..n {
        if let BoundaryCondition::Dirichlet(val) = bc
            && (i == 0 || i + 1 == n)
        {
            field[i] = val;
            continue;
        }
        let j_right = faces[i];
        let j_left = if i > 0 {
            faces[i - 1]
        } else if bc == BoundaryCondition::Periodic {
            faces[n - 1]
        } else {
            0.0
        };
        field[i] += j_right - j_left;
    }
}

/// Injects 2D staggered-face conserved-flux noise $\Delta t \, \nabla \cdot \vec{\eta}$ into `field`,
/// strictly preserving $\sum_{x, y} u_{x, y}$ under Periodic and Neumann boundaries.
#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub fn inject_conserved_flux_noise_2d<R: RngCore + ?Sized>(
    field: &mut [f64],
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    dt: f64,
    mobility_temp: f64,
    bc: BoundaryCondition,
    workspace: &mut SolverWorkspace<'_, f64>,
    rng: &mut R,
) {
    let total = width * height;
    assert!(field.len() >= total && dx > 0.0 && dy > 0.0 && dt >= 0.0);
    if width < 2 || height < 2 || mobility_temp <= 0.0 || dt <= 0.0 {
        return;
    }
    let amp = <f64 as Scalar>::sqrt(2.0 * mobility_temp * dt / (dx * dy));
    let scale_x = amp / dx;
    let scale_y = amp / dy;
    let mut ws = workspace.reborrow();
    let fx = ws.take_slice(total);
    let fy = ws.take_slice(total);
    fill_standard_normal(fx, rng);
    fill_standard_normal(fy, rng);

    if bc != BoundaryCondition::Periodic {
        zero_boundary_normal_fluxes_2d(fx, fy, width, height);
    }

    for y in 0..height {
        for x in 0..width {
            let idx = flatten_2d_index(x, y, width);
            if let BoundaryCondition::Dirichlet(val) = bc
                && (x == 0 || x + 1 == width || y == 0 || y + 1 == height)
            {
                field[idx] = val;
                continue;
            }
            let jx_r = fx[idx] * scale_x;
            let jx_l = if x > 0 {
                fx[flatten_2d_index(x - 1, y, width)] * scale_x
            } else if bc == BoundaryCondition::Periodic {
                fx[flatten_2d_index(width - 1, y, width)] * scale_x
            } else {
                0.0
            };
            let jy_d = fy[idx] * scale_y;
            let jy_u = if y > 0 {
                fy[flatten_2d_index(x, y - 1, width)] * scale_y
            } else if bc == BoundaryCondition::Periodic {
                fy[flatten_2d_index(x, height - 1, width)] * scale_y
            } else {
                0.0
            };
            field[idx] += (jx_r - jx_l) + (jy_d - jy_u);
        }
    }
}

fn zero_boundary_normal_fluxes_2d(fx: &mut [f64], fy: &mut [f64], width: usize, height: usize) {
    for y in 0..height {
        fx[flatten_2d_index(width - 1, y, width)] = 0.0;
    }
    for x in 0..width {
        fy[flatten_2d_index(x, height - 1, width)] = 0.0;
    }
}

pub(crate) fn apply_noise_1d<R: RngCore + ?Sized>(
    u: &mut [f64],
    dx: f64,
    dt: f64,
    bc: BoundaryCondition,
    noise: SpdeNoiseKind,
    workspace: &mut SolverWorkspace<'_, f64>,
    rng: &mut R,
) {
    match noise {
        SpdeNoiseKind::None => {}
        SpdeNoiseKind::Additive { intensity } => {
            inject_additive_noise_1d(u, dx, dt, intensity, bc, workspace, rng);
        }
        SpdeNoiseKind::ConservedFlux { mobility_temp } => {
            inject_conserved_flux_noise_1d(u, dx, dt, mobility_temp, bc, workspace, rng);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_noise_2d<R: RngCore + ?Sized>(
    u: &mut [f64],
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    dt: f64,
    bc: BoundaryCondition,
    noise: SpdeNoiseKind,
    workspace: &mut SolverWorkspace<'_, f64>,
    rng: &mut R,
) {
    match noise {
        SpdeNoiseKind::None => {}
        SpdeNoiseKind::Additive { intensity } => {
            inject_additive_noise_2d(u, width, height, dx, dy, dt, intensity, bc, workspace, rng);
        }
        SpdeNoiseKind::ConservedFlux { mobility_temp } => {
            inject_conserved_flux_noise_2d(
                u,
                width,
                height,
                dx,
                dy,
                dt,
                mobility_temp,
                bc,
                workspace,
                rng,
            );
        }
    }
}
