use math_commons::math_kernel::types::{flatten_2d_index, flatten_3d_index};

use super::{FusedStencilStepper, StencilNeighbors3D, StencilOperators};

impl FusedStencilStepper {
    /// Applies a 2D fused stencil pass over multiple coupled arrays with Neumann boundaries.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_2d_coupled_neumann<const N: usize, F>(
        &self,
        dim: (usize, usize),
        src: [&[f64]; N],
        dst: [&mut [f64]; N],
        dt: f64,
        dir: f64,
        mut op: F,
    ) where
        F: FnMut(
            usize,
            [f64; N],
            [f64; N],
            [f64; N],
            [f64; N],
            [f64; N],
            &StencilOperators,
        ) -> [f64; N],
    {
        let (width, height) = dim;
        let n = width * height;
        if n == 0 {
            return;
        }

        let get_state = |idx| {
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
                let idx = flatten_2d_index(x, y, width);

                let idx_l = if x > 0 { idx - 1 } else { idx };
                let idx_r = if x < width - 1 { idx + 1 } else { idx };
                let idx_u = if y > 0 { idx - width } else { idx };
                let idx_d = if y < height - 1 { idx + width } else { idx };

                let curr = get_state(idx);
                let left = get_state(idx_l);
                let right = get_state(idx_r);
                let up = get_state(idx_u);
                let down = get_state(idx_d);

                let rhs = op(idx, curr, left, right, up, down, &self.ops);

                for s in 0..N {
                    if idx < dst[s].len() {
                        dst[s][idx] = curr[s] + dir * dt * rhs[s];
                    }
                }
            }
        }
    }

    /// Applies a 2D fused stencil pass over multiple coupled arrays with toroidal periodic boundaries.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_2d_coupled_periodic<const N: usize, F>(
        &self,
        dim: (usize, usize),
        src: [&[f64]; N],
        dst: [&mut [f64]; N],
        dt: f64,
        dir: f64,
        mut op: F,
    ) where
        F: FnMut(
            usize,
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
            let y_u = if y > 0 { y - 1 } else { height - 1 };
            let y_d = if y + 1 < height { y + 1 } else { 0 };
            for x in 0..width {
                let x_l = if x > 0 { x - 1 } else { width - 1 };
                let x_r = if x + 1 < width { x + 1 } else { 0 };
                let idx = flatten_2d_index(x, y, width);
                let curr = get_state(idx);
                let left = get_state(flatten_2d_index(x_l, y, width));
                let right = get_state(flatten_2d_index(x_r, y, width));
                let up = get_state(flatten_2d_index(x, y_u, width));
                let down = get_state(flatten_2d_index(x, y_d, width));
                let rhs = op(idx, curr, left, right, up, down, &self.ops);
                for s in 0..N {
                    if idx < dst[s].len() {
                        dst[s][idx] = curr[s] + dir * dt * rhs[s];
                    }
                }
            }
        }
    }

    /// Applies a 3D 7-point fused stencil pass over coupled species with zero-flux Neumann boundaries.
    #[verified_engine::verified]
    pub fn step_3d_coupled_neumann<const N: usize, F>(
        &self,
        dim: (usize, usize, usize),
        src: [&[f64]; N],
        dst: [&mut [f64]; N],
        dt: f64,
        dir: f64,
        mut op: F,
    ) where
        F: FnMut(usize, [f64; N], StencilNeighbors3D<N>, &StencilOperators) -> [f64; N],
    {
        let (width, height, depth) = dim;
        let slice_stride = width * height;
        if slice_stride * depth == 0 {
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
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    let idx = flatten_3d_index(x, y, z, width, height);
                    let neighbors = StencilNeighbors3D {
                        xm: get_state(if x > 0 { idx - 1 } else { idx }),
                        xp: get_state(if x + 1 < width { idx + 1 } else { idx }),
                        ym: get_state(if y > 0 { idx - width } else { idx }),
                        yp: get_state(if y + 1 < height { idx + width } else { idx }),
                        zm: get_state(if z > 0 { idx - slice_stride } else { idx }),
                        zp: get_state(if z + 1 < depth {
                            idx + slice_stride
                        } else {
                            idx
                        }),
                    };
                    let curr = get_state(idx);
                    let rhs = op(idx, curr, neighbors, &self.ops);
                    for s in 0..N {
                        if idx < dst[s].len() {
                            dst[s][idx] = curr[s] + dir * dt * rhs[s];
                        }
                    }
                }
            }
        }
    }
}
