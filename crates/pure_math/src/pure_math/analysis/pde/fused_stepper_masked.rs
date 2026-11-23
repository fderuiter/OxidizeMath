//! Masked 2D zero-flux and 3D neighbor stencil helpers for [`super::FusedStencilStepper`].

use super::{FusedStencilStepper, StencilOperators};
use math_commons::math_kernel::types::{flatten_2d_index, flatten_3d_index};

/// Four-neighbor 2D axial state values and open-face flags around a central grid cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StencilNeighbors2D<const N: usize> {
    /// Neighbor at `(x - 1, y)` (reflected to `curr` at `x == 0` or solid obstacle).
    pub left: [f64; N],
    /// Neighbor at `(x + 1, y)` (reflected to `curr` at `x == width - 1` or solid obstacle).
    pub right: [f64; N],
    /// Neighbor at `(x, y - 1)` (reflected to `curr` at `y == 0` or solid obstacle).
    pub up: [f64; N],
    /// Neighbor at `(x, y + 1)` (reflected to `curr` at `y == height - 1` or solid obstacle).
    pub down: [f64; N],
    /// Open-face flags `[west, east, north, south]` (`false` at domain borders or solid obstacles).
    pub face_open: [bool; 4],
}

/// Six-neighbor 3D axial state values around a central grid cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StencilNeighbors3D<const N: usize> {
    /// Neighbor at `(x - 1, y, z)` (clamped at `x == 0`).
    pub xm: [f64; N],
    /// Neighbor at `(x + 1, y, z)` (clamped at `x == width - 1`).
    pub xp: [f64; N],
    /// Neighbor at `(x, y - 1, z)` (clamped at `y == 0`).
    pub ym: [f64; N],
    /// Neighbor at `(x, y + 1, z)` (clamped at `y == height - 1`).
    pub yp: [f64; N],
    /// Neighbor at `(x, y, z - 1)` (clamped at `z == 0`).
    pub zm: [f64; N],
    /// Neighbor at `(x, y, z + 1)` (clamped at `z == depth - 1`).
    pub zp: [f64; N],
}

impl FusedStencilStepper {
    /// Applies a 2D fused stencil pass over coupled species with outer Neumann borders and internal zero-flux solid masks.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_2d_coupled_masked_neumann<const N: usize, F>(
        &self,
        dim: (usize, usize),
        solid_mask: &[bool],
        src: [&[f64]; N],
        dst: [&mut [f64]; N],
        dt: f64,
        dir: f64,
        mut op: F,
    ) where
        F: FnMut(usize, [f64; N], StencilNeighbors2D<N>, &StencilOperators) -> [f64; N],
    {
        let (width, height) = dim;
        if width * height == 0 {
            return;
        }

        for y in 0..height {
            for x in 0..width {
                let idx = flatten_2d_index(x, y, width);
                let curr = read_coupled_cell(&src, idx);
                if solid_mask.get(idx).copied().unwrap_or(false) {
                    for s in 0..N {
                        if idx < dst[s].len() {
                            dst[s][idx] = curr[s];
                        }
                    }
                    continue;
                }
                let neighbors = sample_2d_masked_neighbors(&src, solid_mask, (x, y), dim, curr);
                let rhs = op(idx, curr, neighbors, &self.ops);
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
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    let idx = flatten_3d_index(x, y, z, width, height);
                    let neighbors = StencilNeighbors3D {
                        xm: read_coupled_cell(&src, if x > 0 { idx - 1 } else { idx }),
                        xp: read_coupled_cell(&src, if x + 1 < width { idx + 1 } else { idx }),
                        ym: read_coupled_cell(&src, if y > 0 { idx - width } else { idx }),
                        yp: read_coupled_cell(&src, if y + 1 < height { idx + width } else { idx }),
                        zm: read_coupled_cell(&src, if z > 0 { idx - slice_stride } else { idx }),
                        zp: read_coupled_cell(
                            &src,
                            if z + 1 < depth {
                                idx + slice_stride
                            } else {
                                idx
                            },
                        ),
                    };
                    let curr = read_coupled_cell(&src, idx);
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

#[inline(always)]
#[verified_engine::verified]
pub(super) fn read_coupled_cell<const N: usize>(src: &[&[f64]; N], idx: usize) -> [f64; N] {
    let mut val = [0.0; N];
    for (v, s) in val.iter_mut().zip(src.iter()) {
        if idx < s.len() {
            *v = s[idx];
        }
    }
    val
}

#[inline(always)]
#[verified_engine::verified]
fn sample_2d_masked_neighbors<const N: usize>(
    src: &[&[f64]; N],
    solid_mask: &[bool],
    pos: (usize, usize),
    dim: (usize, usize),
    curr: [f64; N],
) -> StencilNeighbors2D<N> {
    let (x, y) = pos;
    let (width, height) = dim;
    let idx = flatten_2d_index(x, y, width);
    let is_solid = |nb: usize| solid_mask.get(nb).copied().unwrap_or(false);

    let w_open = x > 0 && !is_solid(idx - 1);
    let e_open = x + 1 < width && !is_solid(idx + 1);
    let n_open = y > 0 && !is_solid(idx - width);
    let s_open = y + 1 < height && !is_solid(idx + width);

    StencilNeighbors2D {
        left: if w_open {
            read_coupled_cell(src, idx - 1)
        } else {
            curr
        },
        right: if e_open {
            read_coupled_cell(src, idx + 1)
        } else {
            curr
        },
        up: if n_open {
            read_coupled_cell(src, idx - width)
        } else {
            curr
        },
        down: if s_open {
            read_coupled_cell(src, idx + width)
        } else {
            curr
        },
        face_open: [w_open, e_open, n_open, s_open],
    }
}
