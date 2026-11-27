// [cite:engine_pde]
use crate::boundary::BoundaryCondition;
use crate::error::PdeError;
use crate::stencil::StencilOperators;
use math_commons::math_kernel::types::flatten_2d_index;

#[allow(clippy::too_many_arguments)]
pub(super) fn advance_cahn_hilliard_stencil_2d(
    ops: &StencilOperators,
    boundary: BoundaryCondition,
    src: &[f64],
    bulk_mu: &[f64],
    lap_phi: &[f64],
    phi: &mut [f64],
    width: usize,
    height: usize,
    mobility: f64,
    kappa: f64,
    dt: f64,
) -> Result<(), PdeError> {
    for y in 0..height {
        for x in 0..width {
            let idx = flatten_2d_index(x, y, width);
            if let BoundaryCondition::Dirichlet(val) = boundary
                && (x == 0 || x + 1 == width || y == 0 || y + 1 == height)
            {
                phi[idx] = val;
                continue;
            }
            let lap_bulk = ops.try_laplacian_2d_at(bulk_mu, width, height, x, y, boundary)?;
            let biharm =
                ops.try_eval_biharmonic_2d_from_laplacian(lap_phi, width, height, x, y, boundary)?;
            phi[idx] = src[idx] + dt * mobility * (lap_bulk - kappa * biharm);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn advance_diffusion_substeps_2d(
    ops: &StencilOperators,
    boundary: BoundaryCondition,
    substeps: usize,
    src: &mut [f64],
    u: &mut [f64],
    width: usize,
    height: usize,
    diffusivity: f64,
    sub_dt: f64,
) -> Result<(), PdeError> {
    for _ in 0..substeps {
        src.copy_from_slice(u);
        for y in 0..height {
            for x in 0..width {
                let idx = flatten_2d_index(x, y, width);
                if let BoundaryCondition::Dirichlet(val) = boundary
                    && (x == 0 || x + 1 == width || y == 0 || y + 1 == height)
                {
                    u[idx] = val;
                    continue;
                }
                let lap = ops.try_laplacian_2d_at(src, width, height, x, y, boundary)?;
                u[idx] = src[idx] + sub_dt * diffusivity * lap;
            }
        }
    }
    Ok(())
}

pub(super) fn apply_local_reaction_2d<F: FnMut(f64) -> f64>(
    u: &mut [f64],
    width: usize,
    height: usize,
    dt_react: f64,
    bc: BoundaryCondition,
    reaction: &mut F,
) {
    for y in 0..height {
        for x in 0..width {
            let idx = flatten_2d_index(x, y, width);
            if let BoundaryCondition::Dirichlet(val) = bc
                && (x == 0 || x + 1 == width || y == 0 || y + 1 == height)
            {
                u[idx] = val;
                continue;
            }
            u[idx] += dt_react * reaction(u[idx]);
        }
    }
}
