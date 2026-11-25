// [cite:engine_pde]
use math_commons::math_kernel::types::{flatten_2d_index, flatten_3d_index};

/// Spatial boundary condition applied along regular Cartesian grid boundaries.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BoundaryCondition<T = f64> {
    /// Periodic wrap-around ($u(-1) = u(N - 1)$, $u(N) = u(0)$).
    Periodic,
    /// Cell-centered zero-flux Neumann boundary ($u(-1) = u(0)$, $u(N) = u(N - 1)$),
    /// strictly conserving discrete finite-volume sums.
    Neumann,
    /// Node-centered symmetric reflection Neumann boundary ($u(-1) = u(1)$, $u(N) = u(N - 2)$).
    SymmetricNeumann,
    /// Fixed exterior/boundary Dirichlet value $u_{\partial\Omega} = c$.
    Dirichlet(T),
}

impl<T> Default for BoundaryCondition<T> {
    #[inline]
    fn default() -> Self {
        Self::Periodic
    }
}

impl<T: Copy> BoundaryCondition<T> {
    /// Returns `true` when boundary nodes `(0, n - 1)` are pinned to a fixed value.
    #[inline]
    pub const fn is_dirichlet(self) -> bool {
        matches!(self, Self::Dirichlet(_))
    }

    /// Resolves `(prev_val, curr_val, next_val)` around 1D cell `x` of a slice `u` of length `n`.
    #[inline]
    #[verified_engine::verified]
    pub fn sample_1d(self, u: &[T], n: usize, x: usize) -> (T, T, T) {
        debug_assert!(n > 0 && x < n && u.len() >= n);
        let curr = u[x];
        if n == 1 {
            return match self {
                Self::Dirichlet(val) => (val, val, val),
                _ => (curr, curr, curr),
            };
        }
        match self {
            Self::Periodic => {
                let xm = if x > 0 { x - 1 } else { n - 1 };
                let xp = if x + 1 < n { x + 1 } else { 0 };
                (u[xm], curr, u[xp])
            }
            Self::Neumann => {
                let xm = x.saturating_sub(1);
                let xp = (x + 1).min(n - 1);
                (u[xm], curr, u[xp])
            }
            Self::SymmetricNeumann => {
                let xm = if x > 0 { x - 1 } else { 1 };
                let xp = if x + 1 < n { x + 1 } else { n - 2 };
                (u[xm], curr, u[xp])
            }
            Self::Dirichlet(val) => {
                let prev = if x > 0 { u[x - 1] } else { val };
                let next = if x + 1 < n { u[x + 1] } else { val };
                let center = if x == 0 || x + 1 == n { val } else { curr };
                (prev, center, next)
            }
        }
    }

    /// Resolves `(curr, left, right, up, down)` around 2D cell `(x, y)` on a `width x height` grid.
    #[inline]
    #[verified_engine::verified]
    pub fn sample_2d(
        self,
        u: &[T],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
    ) -> (T, T, T, T, T) {
        debug_assert!(width > 0 && height > 0 && x < width && y < height);
        debug_assert!(u.len() >= width * height);
        let curr = u[flatten_2d_index(x, y, width)];
        match self {
            Self::Periodic => {
                let xm = if x > 0 { x - 1 } else { width - 1 };
                let xp = if x + 1 < width { x + 1 } else { 0 };
                let ym = if y > 0 { y - 1 } else { height - 1 };
                let yp = if y + 1 < height { y + 1 } else { 0 };
                (
                    curr,
                    u[flatten_2d_index(xm, y, width)],
                    u[flatten_2d_index(xp, y, width)],
                    u[flatten_2d_index(x, ym, width)],
                    u[flatten_2d_index(x, yp, width)],
                )
            }
            Self::Neumann => {
                let xm = x.saturating_sub(1);
                let xp = (x + 1).min(width - 1);
                let ym = y.saturating_sub(1);
                let yp = (y + 1).min(height - 1);
                (
                    curr,
                    u[flatten_2d_index(xm, y, width)],
                    u[flatten_2d_index(xp, y, width)],
                    u[flatten_2d_index(x, ym, width)],
                    u[flatten_2d_index(x, yp, width)],
                )
            }
            Self::SymmetricNeumann => {
                let xm = if x > 0 { x - 1 } else { usize::from(width > 1) };
                let xp = if x + 1 < width {
                    x + 1
                } else {
                    width.saturating_sub(2)
                };
                let ym = if y > 0 {
                    y - 1
                } else {
                    usize::from(height > 1)
                };
                let yp = if y + 1 < height {
                    y + 1
                } else {
                    height.saturating_sub(2)
                };
                (
                    curr,
                    u[flatten_2d_index(xm, y, width)],
                    u[flatten_2d_index(xp, y, width)],
                    u[flatten_2d_index(x, ym, width)],
                    u[flatten_2d_index(x, yp, width)],
                )
            }
            Self::Dirichlet(val) => {
                let on_edge = x == 0 || x + 1 == width || y == 0 || y + 1 == height;
                let center = if on_edge { val } else { curr };
                let left = if x > 0 {
                    u[flatten_2d_index(x - 1, y, width)]
                } else {
                    val
                };
                let right = if x + 1 < width {
                    u[flatten_2d_index(x + 1, y, width)]
                } else {
                    val
                };
                let up = if y > 0 {
                    u[flatten_2d_index(x, y - 1, width)]
                } else {
                    val
                };
                let down = if y + 1 < height {
                    u[flatten_2d_index(x, y + 1, width)]
                } else {
                    val
                };
                (center, left, right, up, down)
            }
        }
    }

    /// Resolves `(curr, [xm, xp, ym, yp, zm, zp])` around 3D cell `(x, y, z)` on a `width x height x depth` grid.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[verified_engine::verified]
    pub fn sample_3d(
        self,
        u: &[T],
        width: usize,
        height: usize,
        depth: usize,
        x: usize,
        y: usize,
        z: usize,
    ) -> (T, [T; 6]) {
        debug_assert!(width > 0 && height > 0 && depth > 0);
        debug_assert!(x < width && y < height && z < depth);
        debug_assert!(u.len() >= width * height * depth);
        let curr = u[flatten_3d_index(x, y, z, width, height)];
        if let Self::Dirichlet(val) = self {
            let on_edge =
                x == 0 || x + 1 == width || y == 0 || y + 1 == height || z == 0 || z + 1 == depth;
            let c = if on_edge { val } else { curr };
            let xm = if x > 0 {
                u[flatten_3d_index(x - 1, y, z, width, height)]
            } else {
                val
            };
            let xp = if x + 1 < width {
                u[flatten_3d_index(x + 1, y, z, width, height)]
            } else {
                val
            };
            let ym = if y > 0 {
                u[flatten_3d_index(x, y - 1, z, width, height)]
            } else {
                val
            };
            let yp = if y + 1 < height {
                u[flatten_3d_index(x, y + 1, z, width, height)]
            } else {
                val
            };
            let zm = if z > 0 {
                u[flatten_3d_index(x, y, z - 1, width, height)]
            } else {
                val
            };
            let zp = if z + 1 < depth {
                u[flatten_3d_index(x, y, z + 1, width, height)]
            } else {
                val
            };
            return (c, [xm, xp, ym, yp, zm, zp]);
        }

        let (xm_i, xp_i) = axis_indices(self, x, width);
        let (ym_i, yp_i) = axis_indices(self, y, height);
        let (zm_i, zp_i) = axis_indices(self, z, depth);
        (
            curr,
            [
                u[flatten_3d_index(xm_i, y, z, width, height)],
                u[flatten_3d_index(xp_i, y, z, width, height)],
                u[flatten_3d_index(x, ym_i, z, width, height)],
                u[flatten_3d_index(x, yp_i, z, width, height)],
                u[flatten_3d_index(x, y, zm_i, width, height)],
                u[flatten_3d_index(x, y, zp_i, width, height)],
            ],
        )
    }
}

#[inline]
fn axis_indices<T: Copy>(bc: BoundaryCondition<T>, i: usize, len: usize) -> (usize, usize) {
    match bc {
        BoundaryCondition::Periodic => (
            if i > 0 { i - 1 } else { len - 1 },
            if i + 1 < len { i + 1 } else { 0 },
        ),
        BoundaryCondition::SymmetricNeumann => (
            if i > 0 { i - 1 } else { usize::from(len > 1) },
            if i + 1 < len {
                i + 1
            } else {
                len.saturating_sub(2)
            },
        ),
        _ => (i.saturating_sub(1), (i + 1).min(len - 1)),
    }
}
