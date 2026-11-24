//! Positivity-preserving 2D donor-cell upwind advection and zero-flux obstacle-masked
//! reaction-diffusion stepping for [`FiniteDifference2D`].

use super::FiniteDifference2D;
use crate::biology::diffusion::SpatialDiffusion;
use crate::biology::morphogenesis::reaction::ReactionKinetics;
use math_commons::math_kernel::types::flatten_2d_index;

/// Row-major 2D velocity field and optional solid-obstacle mask for advective-diffusive transport.
///
/// Slices must have length `width * height` indexed via `flatten_2d_index(x, y, width)`.
/// Cells marked `true` in `obstacles` act as impermeable solids: interior concentrations remain
/// `0.0` and adjacent fluid cells enforce zero-flux Neumann reflection ($\nabla c \cdot \mathbf{n} = 0$).
///
/// # Examples
///
/// ```rust
/// # use domain_biology::__doctest_facade as math_explorer;
/// use math_explorer::biology::diffusion::AdvectionField2D;
///
/// let ux = [0.05; 16];
/// let uy = [0.0; 16];
/// let obstacles = [false; 16];
/// let field = AdvectionField2D::with_obstacles(&ux, &uy, &obstacles);
/// assert_eq!(field.ux.len(), 16);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AdvectionField2D<'a> {
    /// Horizontal velocity component `u_x` in row-major order (`y * width + x`).
    pub ux: &'a [f64],
    /// Vertical velocity component `u_y` in row-major order (`y * width + x`).
    pub uy: &'a [f64],
    /// Optional boolean solid-obstacle mask in row-major order (`true` = impermeable solid).
    pub obstacles: Option<&'a [bool]>,
    /// Whether horizontal boundaries (`x = 0` and `x = width - 1`) wrap periodically.
    pub periodic_x: bool,
}

impl<'a> AdvectionField2D<'a> {
    /// Creates an obstacle-free row-major 2D velocity field with zero-flux outer boundaries.
    #[verified_engine::verified]
    pub fn new(ux: &'a [f64], uy: &'a [f64]) -> Self {
        Self {
            ux,
            uy,
            obstacles: None,
            periodic_x: false,
        }
    }

    /// Creates a row-major 2D velocity field paired with a boolean solid-obstacle mask.
    #[verified_engine::verified]
    pub fn with_obstacles(ux: &'a [f64], uy: &'a [f64], obstacles: &'a [bool]) -> Self {
        Self {
            ux,
            uy,
            obstacles: Some(obstacles),
            periodic_x: false,
        }
    }

    /// Configures whether streamwise (`x`) domain boundaries wrap periodically.
    #[verified_engine::verified]
    pub fn with_periodic_x(mut self, periodic_x: bool) -> Self {
        self.periodic_x = periodic_x;
        self
    }
}

#[inline(always)]
#[verified_engine::verified]
fn is_cell_obstacle(obstacles: Option<&[bool]>, idx: usize) -> bool {
    match obstacles {
        Some(mask) => mask[idx],
        None => false,
    }
}

#[inline(always)]
#[verified_engine::verified]
fn neighbor_indices_zero_flux(
    pos: (usize, usize),
    dim: (usize, usize),
    idx: usize,
    obstacles: Option<&[bool]>,
    periodic_x: bool,
) -> (usize, usize, usize, usize) {
    let (x, y) = pos;
    let (width, height) = dim;
    let idx_l = if x == 0 {
        let wrap = idx + width - 1;
        if periodic_x && width > 1 && !is_cell_obstacle(obstacles, wrap) {
            wrap
        } else {
            idx
        }
    } else if is_cell_obstacle(obstacles, idx - 1) {
        idx
    } else {
        idx - 1
    };
    let idx_r = if x + 1 >= width {
        let wrap = idx + 1 - width;
        if periodic_x && width > 1 && !is_cell_obstacle(obstacles, wrap) {
            wrap
        } else {
            idx
        }
    } else if is_cell_obstacle(obstacles, idx + 1) {
        idx
    } else {
        idx + 1
    };
    let idx_u = if y == 0 || is_cell_obstacle(obstacles, idx - width) {
        idx
    } else {
        idx - width
    };
    let idx_d = if y + 1 >= height || is_cell_obstacle(obstacles, idx + width) {
        idx
    } else {
        idx + width
    };
    (idx_l, idx_r, idx_u, idx_d)
}

impl FiniteDifference2D {
    /// Advances an `N`-species reaction-advection-diffusion system by one explicit step using
    /// 2D central Laplacian diffusion, [`pure_math::pure_math::analysis::pde::fused_stepper::StencilOperators::upwind_2d`]
    /// positivity-preserving advection, and zero-flux Neumann reflection across domain walls and
    /// internal boolean obstacle masks.
    #[verified_engine::verified]
    pub fn step_advection_diffusion_masked<const N: usize, K: ReactionKinetics<N>>(
        &self,
        state: [&[f64]; N],
        mut next_state: [&mut [f64]; N],
        field: AdvectionField2D<'_>,
        dt: f64,
        coeffs: [f64; N],
        kinetics: &K,
    ) {
        let width = *self.geometry.width;
        let height = *self.geometry.height;
        let n = width * height;
        if N == 0 || n == 0 {
            return;
        }
        assert!(field.ux.len() >= n, "AdvectionField2D ux slice too small");
        assert!(field.uy.len() >= n, "AdvectionField2D uy slice too small");
        if let Some(mask) = field.obstacles {
            assert!(mask.len() >= n, "AdvectionField2D obstacle mask too small");
        }
        for s in 0..N {
            assert!(state[s].len() >= n, "State buffer too small");
            assert!(next_state[s].len() >= n, "Next-state buffer too small");
        }

        let ops = SpatialDiffusion::<N>::stepper(self).ops;
        for y in 0..height {
            for x in 0..width {
                let idx = flatten_2d_index(x, y, width);
                if is_cell_obstacle(field.obstacles, idx) {
                    for dst in next_state.iter_mut().take(N) {
                        dst[idx] = 0.0;
                    }
                    continue;
                }

                let (idx_l, idx_r, idx_u, idx_d) = neighbor_indices_zero_flux(
                    (x, y),
                    (width, height),
                    idx,
                    field.obstacles,
                    field.periodic_x,
                );
                let mut curr = [0.0; N];
                for s in 0..N {
                    curr[s] = state[s][idx];
                }
                let rates = kinetics.reaction(curr);
                let vel = (field.ux[idx], field.uy[idx]);

                for s in 0..N {
                    let src = state[s];
                    let lap = ops.central_diff_2nd_2d(
                        curr[s], src[idx_l], src[idx_r], src[idx_u], src[idx_d],
                    );
                    let adv =
                        ops.upwind_2d(vel, curr[s], src[idx_l], src[idx_r], src[idx_u], src[idx_d]);
                    next_state[s][idx] = curr[s] + dt * (coeffs[s] * lap - adv + rates[s]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use math_commons::math_kernel::types::{Dimension, StepSize};

    struct ZeroReaction;
    impl ReactionKinetics<2> for ZeroReaction {
        fn reaction(&self, _concentrations: [f64; 2]) -> [f64; 2] {
            [0.0, 0.0]
        }
    }

    fn assert_masked_invariants(
        u: &[f64],
        v: &[f64],
        obstacles: &[bool],
        width: usize,
        height: usize,
    ) {
        let mut downstream_mass = 0.0;
        for y in 0..height {
            for x in 0..width {
                let idx = flatten_2d_index(x, y, width);
                assert!(
                    u[idx] >= 0.0 && v[idx] >= 0.0,
                    "Positivity violated at ({x}, {y}): u={}, v={}",
                    u[idx],
                    v[idx]
                );
                assert!(
                    u[idx] <= 1.0 + 1e-12,
                    "Maximum principle violated at ({x}, {y}): u={}",
                    u[idx]
                );
                if obstacles[idx] {
                    assert_eq!(u[idx], 0.0, "Morphogen u leaked into solid obstacle");
                    assert_eq!(v[idx], 0.0, "Morphogen v leaked into solid obstacle");
                } else if x > width / 3 {
                    downstream_mass += u[idx];
                }
            }
        }
        assert!(
            downstream_mass > 1.0,
            "Advected morphogen plume should wrap past solid obstacle"
        );
    }

    fn run_masked_grid_check(width: usize, height: usize) {
        let fd = FiniteDifference2D::new(
            Dimension(width),
            Dimension(height),
            StepSize(1.0),
            StepSize(1.0),
        );
        let n = width * height;
        let mut u = vec![0.0; n];
        let mut v = vec![0.0; n];
        let mut next_u = vec![0.0; n];
        let mut next_v = vec![0.0; n];
        let mut ux = vec![0.10; n];
        let mut uy = vec![0.03; n];
        let mut obstacles = vec![false; n];

        let cx = (width / 3) as isize;
        let cy = (height / 2) as isize;
        for y in 0..height {
            for x in 0..width {
                let idx = flatten_2d_index(x, y, width);
                let dx = x as isize - cx;
                let dy = y as isize - cy;
                if dx * dx + dy * dy <= 25 {
                    obstacles[idx] = true;
                    ux[idx] = 0.0;
                    uy[idx] = 0.0;
                } else if x < width / 4 && y > height / 4 && y < (3 * height) / 4 {
                    u[idx] = 1.0;
                    v[idx] = 0.5;
                }
            }
        }

        let dt = 0.25;
        let coeffs = [0.2, 0.05];
        for _ in 0..120 {
            let field = AdvectionField2D::with_obstacles(&ux, &uy, &obstacles);
            fd.step_advection_diffusion_masked(
                [u.as_slice(), v.as_slice()],
                [next_u.as_mut_slice(), next_v.as_mut_slice()],
                field,
                dt,
                coeffs,
                &ZeroReaction,
            );
            std::mem::swap(&mut u, &mut next_u);
            std::mem::swap(&mut v, &mut next_v);
        }
        assert_masked_invariants(&u, &v, &obstacles, width, height);
    }

    #[test]
    fn test_masked_advection_positivity_and_no_leakage_100x50() {
        run_masked_grid_check(100, 50);
    }

    #[test]
    fn test_masked_advection_positivity_and_no_leakage_100x100() {
        run_masked_grid_check(100, 100);
    }
}
