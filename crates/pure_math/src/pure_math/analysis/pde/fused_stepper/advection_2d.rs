use super::{FusedStencilStepper, StencilOperators};
use math_commons::math_kernel::types::flatten_2d_index;

/// Computes `(idx, left, right, up, down)` row-major indices with zero-flux reflection at domain
/// edges and solid `obstacles` cells.
#[inline(always)]
#[verified_engine::verified]
pub fn reflect_2d_neighbors(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    obstacles: &[bool],
) -> (usize, usize, usize, usize, usize) {
    let idx = flatten_2d_index(x, y, width);
    let cand_l = if x > 0 { idx - 1 } else { idx };
    let cand_r = if x + 1 < width { idx + 1 } else { idx };
    let cand_u = if y > 0 { idx - width } else { idx };
    let cand_d = if y + 1 < height { idx + width } else { idx };
    let is_obs = |i: usize| i < obstacles.len() && obstacles[i];
    let idx_l = if is_obs(cand_l) { idx } else { cand_l };
    let idx_r = if is_obs(cand_r) { idx } else { cand_r };
    let idx_u = if is_obs(cand_u) { idx } else { cand_u };
    let idx_d = if is_obs(cand_d) { idx } else { cand_d };
    (idx, idx_l, idx_r, idx_u, idx_d)
}

/// Computes `[west, east, up, down]` normal face velocities, enforcing zero normal velocity at
/// reflected domain boundaries and solid obstacle interfaces.
#[inline(always)]
#[verified_engine::verified]
pub fn advection_face_velocities(
    idx: usize,
    idx_l: usize,
    idx_r: usize,
    idx_u: usize,
    idx_d: usize,
    vx: &[f64],
    vy: &[f64],
) -> [f64; 4] {
    let ux_c = if idx < vx.len() { vx[idx] } else { 0.0 };
    let uy_c = if idx < vy.len() { vy[idx] } else { 0.0 };
    let vw = if idx_l == idx || idx_l >= vx.len() {
        0.0
    } else {
        0.5 * (vx[idx_l] + ux_c)
    };
    let ve = if idx_r == idx || idx_r >= vx.len() {
        0.0
    } else {
        0.5 * (ux_c + vx[idx_r])
    };
    let vn = if idx_u == idx || idx_u >= vy.len() {
        0.0
    } else {
        0.5 * (vy[idx_u] + uy_c)
    };
    let vs = if idx_d == idx || idx_d >= vy.len() {
        0.0
    } else {
        0.5 * (uy_c + vy[idx_d])
    };
    [vw, ve, vn, vs]
}

impl FusedStencilStepper {
    /// Applies a 2D fused advection-diffusion-reaction stencil pass with zero-flux reflection at
    /// domain boundaries and solid `obstacles` cells.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_2d_coupled_advection_neumann<const N: usize, F>(
        &self,
        dim: (usize, usize),
        src: [&[f64]; N],
        dst: [&mut [f64]; N],
        vx: &[f64],
        vy: &[f64],
        obstacles: &[bool],
        dt: f64,
        dir: f64,
        mut op: F,
    ) where
        F: FnMut(
            usize,
            [f64; 4],
            [f64; N],
            [f64; N],
            [f64; N],
            [f64; N],
            [f64; N],
            &StencilOperators,
        ) -> [f64; N],
    {
        let (width, height) = dim;
        if width == 0 || height == 0 {
            return;
        }
        let get_state = |idx: usize| {
            let mut val = [0.0; N];
            for s in 0..N {
                if idx < src[s].len() {
                    val[s] = src[s][idx];
                }
            }
            val
        };
        for y in 0..height {
            for x in 0..width {
                let (idx, idx_l, idx_r, idx_u, idx_d) =
                    reflect_2d_neighbors(x, y, width, height, obstacles);
                let curr = get_state(idx);
                if idx < obstacles.len() && obstacles[idx] {
                    for s in 0..N {
                        if idx < dst[s].len() {
                            dst[s][idx] = curr[s];
                        }
                    }
                    continue;
                }
                let face_vel = advection_face_velocities(idx, idx_l, idx_r, idx_u, idx_d, vx, vy);
                let rhs = op(
                    idx,
                    face_vel,
                    curr,
                    get_state(idx_l),
                    get_state(idx_r),
                    get_state(idx_u),
                    get_state(idx_d),
                    &self.ops,
                );
                for s in 0..N {
                    if idx < dst[s].len() {
                        dst[s][idx] = curr[s] + dir * dt * rhs[s];
                    }
                }
            }
        }
    }
}
