// [cite:engine_pde]
use crate::stencil::{StencilNeighbors3D, StencilOperators};
use math_commons::math_kernel::types::{StepSize, flatten_2d_index, flatten_3d_index};

#[doc = include_str!("../../../docs/math_specs/snippets/pde_stepper.md")]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FusedStencilStepper {
    /// Precomputed spatial finite-difference operators.
    pub ops: StencilOperators,
}

impl FusedStencilStepper {
    /// Creates a 1D fused stencil stepper after validating `dx > 0.0`.
    #[verified_engine::verified]
    pub fn try_new(dx: StepSize) -> Result<Self, math_commons::error::MathError> {
        debug_assert!(dx.is_finite() || !dx.is_finite());
        Ok(Self {
            ops: StencilOperators::try_new(dx)?,
        })
    }

    /// Creates a 2D fused stencil stepper after validating `dx > 0.0` and `dy > 0.0`.
    #[verified_engine::verified]
    pub fn try_new_2d(dx: StepSize, dy: StepSize) -> Result<Self, math_commons::error::MathError> {
        debug_assert!(dx.is_finite() || !dx.is_finite());
        Ok(Self {
            ops: StencilOperators::try_new_2d(dx, dy)?,
        })
    }

    /// Creates a 3D fused stencil stepper after validating `dx > 0.0`, `dy > 0.0`, and `dz > 0.0`.
    #[verified_engine::verified]
    pub fn try_new_3d(
        dx: StepSize,
        dy: StepSize,
        dz: StepSize,
    ) -> Result<Self, math_commons::error::MathError> {
        debug_assert!(dx.is_finite() || !dx.is_finite());
        Ok(Self {
            ops: StencilOperators::try_new_3d(dx, dy, dz)?,
        })
    }

    /// Constructs a 1D fused stencil stepper, panicking if `*dx <= 0.0`.
    #[verified_engine::verified]
    pub fn new(dx: StepSize) -> Self {
        assert!(*dx > 0.0 && dx.is_finite());
        Self {
            ops: StencilOperators::new(dx),
        }
    }

    /// Constructs a 2D fused stencil stepper, panicking if `*dx <= 0.0` or `*dy <= 0.0`.
    #[verified_engine::verified]
    pub fn new_2d(dx: StepSize, dy: StepSize) -> Self {
        assert!(*dx > 0.0 && *dy > 0.0);
        Self {
            ops: StencilOperators::new_2d(dx, dy),
        }
    }

    /// Constructs a 3D fused stencil stepper for `(dx, dy, dz)`.
    #[verified_engine::verified]
    pub fn new_3d(dx: StepSize, dy: StepSize, dz: StepSize) -> Self {
        assert!(*dx > 0.0 && *dy > 0.0 && *dz > 0.0);
        Self {
            ops: StencilOperators::new_3d(dx, dy, dz),
        }
    }

    /// Applies a 1D interior fused stencil pass (`1..n-1`) over a single array.
    #[verified_engine::verified]
    pub fn step_1d_slice<F>(&self, src: &[f64], dst: &mut [f64], dt: f64, dir: f64, mut op: F)
    where
        F: FnMut(usize, f64, f64, f64, &StencilOperators) -> f64,
    {
        let n = src.len();
        debug_assert!(*self.ops.dx > 0.0);
        if n > 2 && dst.len() >= n {
            for (i, win) in src.windows(3).enumerate() {
                let rhs = op(i + 1, win[0], win[1], win[2], &self.ops);
                dst[i + 1] = win[1] + dir * dt * rhs;
            }
        }
    }

    /// Applies a 1D fused stencil pass over `N` coupled arrays with zero-flux Neumann boundaries.
    #[verified_engine::verified]
    pub fn step_1d_coupled_neumann<const N: usize, F>(
        &self,
        n: usize,
        src: [&[f64]; N],
        dst: [&mut [f64]; N],
        dt: f64,
        dir: f64,
        mut op: F,
    ) where
        F: FnMut(usize, [f64; N], [f64; N], [f64; N], &StencilOperators) -> [f64; N],
    {
        debug_assert!(*self.ops.dx > 0.0);
        if n == 0 {
            return;
        }
        for i in 0..n {
            let prev = gather_state::<N>(&src, i.saturating_sub(1));
            let curr = gather_state::<N>(&src, i);
            let next = gather_state::<N>(&src, (i + 1).min(n - 1));
            let rhs = op(i, prev, curr, next, &self.ops);
            for s in 0..N {
                if i < dst[s].len() {
                    dst[s][i] = curr[s] + dir * dt * rhs[s];
                }
            }
        }
    }

    /// Applies a 2D fused stencil pass over `N` coupled arrays with zero-flux Neumann boundaries.
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
        debug_assert!(*self.ops.dx > 0.0 && *self.ops.dy > 0.0);
        if width == 0 || height == 0 {
            return;
        }
        for y in 0..height {
            for x in 0..width {
                let idx = flatten_2d_index(x, y, width);
                let idx_l = if x > 0 { idx - 1 } else { idx };
                let idx_r = if x + 1 < width { idx + 1 } else { idx };
                let idx_u = if y > 0 { idx - width } else { idx };
                let idx_d = if y + 1 < height { idx + width } else { idx };
                let curr = gather_state::<N>(&src, idx);
                let rhs = op(
                    idx,
                    curr,
                    gather_state::<N>(&src, idx_l),
                    gather_state::<N>(&src, idx_r),
                    gather_state::<N>(&src, idx_u),
                    gather_state::<N>(&src, idx_d),
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

    /// Applies a 3D 7-point fused stencil pass over `N` coupled species with zero-flux Neumann boundaries.
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
        debug_assert!(*self.ops.dz > 0.0);
        if slice_stride * depth == 0 {
            return;
        }
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    let idx = flatten_3d_index(x, y, z, width, height);
                    let neighbors = StencilNeighbors3D {
                        xm: gather_state::<N>(&src, if x > 0 { idx - 1 } else { idx }),
                        xp: gather_state::<N>(&src, if x + 1 < width { idx + 1 } else { idx }),
                        ym: gather_state::<N>(&src, if y > 0 { idx - width } else { idx }),
                        yp: gather_state::<N>(&src, if y + 1 < height { idx + width } else { idx }),
                        zm: gather_state::<N>(&src, if z > 0 { idx - slice_stride } else { idx }),
                        zp: gather_state::<N>(
                            &src,
                            if z + 1 < depth {
                                idx + slice_stride
                            } else {
                                idx
                            },
                        ),
                    };
                    let curr = gather_state::<N>(&src, idx);
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

#[inline]
fn gather_state<const N: usize>(src: &[&[f64]; N], idx: usize) -> [f64; N] {
    let mut val = [0.0; N];
    for s in 0..N {
        if idx < src[s].len() {
            val[s] = src[s][idx];
        }
    }
    val
}
