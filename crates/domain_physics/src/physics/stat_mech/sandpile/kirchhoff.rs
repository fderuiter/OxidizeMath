//! Exact Kirchhoff Laplacian Bareiss Determinants and Compile-Time Lattice Kernels.
//!
//! Provides zero-allocation integer linear algebra for counting the order
//! $|G| = \det(\Delta)$ of the Abelian sandpile group on open-boundary $W \times H$
//! grids via fraction-free Bareiss Gaussian elimination, alongside stack-resident
//! toppling, Creutz identity, and Dhar burning kernels for [`super::SandpileElement`].

use diagnostics::{Diagnostic, Severity};
use math_commons::math_kernel::types::flatten_2d_index;
use std::collections::HashMap;
use thiserror::Error;

use super::{CRITICAL_HEIGHT, MAX_STABLE_HEIGHT};

/// Maximum number of lattice sites supported by stack-resident Bareiss elimination (`5x5 = 25`).
pub const MAX_BAREISS_CELLS: usize = 25;

/// Errors returned by Abelian sandpile lattice and group operations.
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandpileError {
    /// Grid width or height is zero or exceeds the interactive bound.
    #[error("sandpile dimensions {width}x{height} must each lie in 1..={max_dim}")]
    InvalidDimensions {
        /// Requested lattice width.
        width: usize,
        /// Requested lattice height.
        height: usize,
        /// Maximum supported dimension per axis.
        max_dim: usize,
    },
    /// Lattice coordinates lie outside `0..width` or `0..height`.
    #[error("cell ({x}, {y}) is outside sandpile bounds {width}x{height}")]
    OutOfBounds {
        /// Requested horizontal coordinate.
        x: usize,
        /// Requested vertical coordinate.
        y: usize,
        /// Lattice width.
        width: usize,
        /// Lattice height.
        height: usize,
    },
    /// Total site count exceeds the stack-allocated Bareiss determinant limit.
    #[error("Kirchhoff Bareiss solver supports up to {max_cells} cells, got {cells}")]
    BareissGridTooLarge {
        /// Requested `width * height` site count.
        cells: usize,
        /// Maximum supported site count.
        max_cells: usize,
    },
    /// Slice length did not match `width * height`.
    #[error("buffer length {actual} does not match grid cell count {expected}")]
    BufferLengthMismatch {
        ///Expected cell count `width * height`.
        expected: usize,
        /// Provided slice length.
        actual: usize,
    },
    /// Exact integer determinant overflowed `i128` arithmetic.
    #[error("exact Bareiss determinant overflowed 128-bit integer precision")]
    ArithmeticOverflow,
}

impl Diagnostic for SandpileError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "SandpileError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

/// Computes the exact number of recurrent configurations $|G| = \det(\Delta)$ on an
/// open-boundary $W \times H$ grid using fraction-free Bareiss integer elimination.
///
/// By Kirchhoff's Matrix-Tree Theorem and Dhar's Abelian sandpile theorem (1990),
/// the order of the recurrent sandpile group equals the determinant of the reduced
/// graph Laplacian $\Delta$ ($\Delta_{ii} = 4$, $\Delta_{ij} = -1$ for orthogonal neighbors).
///
/// # Examples
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::stat_mech::sandpile::kirchhoff_determinant_bareiss;
///
/// assert_eq!(kirchhoff_determinant_bareiss(2, 2).unwrap(), 192);
/// assert_eq!(kirchhoff_determinant_bareiss(3, 3).unwrap(), 100_352);
/// ```
#[verified_engine::verified]
pub fn kirchhoff_determinant_bareiss(width: usize, height: usize) -> Result<i128, SandpileError> {
    if width == 0 || height == 0 {
        return Err(SandpileError::InvalidDimensions {
            width,
            height,
            max_dim: MAX_BAREISS_CELLS,
        });
    }
    let n = width.saturating_mul(height);
    if n > MAX_BAREISS_CELLS {
        return Err(SandpileError::BareissGridTooLarge {
            cells: n,
            max_cells: MAX_BAREISS_CELLS,
        });
    }

    let mut mat = [[0_i128; MAX_BAREISS_CELLS]; MAX_BAREISS_CELLS];
    populate_reduced_laplacian(width, height, &mut mat);
    bareiss_determinant_in_place(&mut mat, n)
}

fn populate_reduced_laplacian(
    width: usize,
    height: usize,
    mat: &mut [[i128; MAX_BAREISS_CELLS]; MAX_BAREISS_CELLS],
) {
    for y in 0..height {
        for x in 0..width {
            let i = flatten_2d_index(x, y, width);
            mat[i][i] = i128::from(CRITICAL_HEIGHT);
            if x > 0 {
                mat[i][flatten_2d_index(x - 1, y, width)] = -1;
            }
            if x + 1 < width {
                mat[i][flatten_2d_index(x + 1, y, width)] = -1;
            }
            if y > 0 {
                mat[i][flatten_2d_index(x, y - 1, width)] = -1;
            }
            if y + 1 < height {
                mat[i][flatten_2d_index(x, y + 1, width)] = -1;
            }
        }
    }
}

fn bareiss_determinant_in_place(
    mat: &mut [[i128; MAX_BAREISS_CELLS]; MAX_BAREISS_CELLS],
    n: usize,
) -> Result<i128, SandpileError> {
    let mut sign = 1_i128;
    let mut prev_pivot = 1_i128;

    for k in 0..n {
        if mat[k][k] == 0 {
            let Some(swap_row) = ((k + 1)..n).find(|&r| mat[r][k] != 0) else {
                return Ok(0);
            };
            mat.swap(k, swap_row);
            sign = -sign;
        }
        let pivot = mat[k][k];
        for i in (k + 1)..n {
            let row_k_factor = mat[i][k];
            for j in (k + 1)..n {
                let lhs = mat[i][j]
                    .checked_mul(pivot)
                    .ok_or(SandpileError::ArithmeticOverflow)?;
                let rhs = row_k_factor
                    .checked_mul(mat[k][j])
                    .ok_or(SandpileError::ArithmeticOverflow)?;
                let diff = lhs
                    .checked_sub(rhs)
                    .ok_or(SandpileError::ArithmeticOverflow)?;
                mat[i][j] = diff / prev_pivot;
            }
            mat[i][k] = 0;
        }
        prev_pivot = pivot;
    }

    sign.checked_mul(mat[n - 1][n - 1])
        .ok_or(SandpileError::ArithmeticOverflow)
}

/// Stabilizes a compile-time `W x H` sandpile array in place and returns total topples.
pub(crate) fn stabilize_grid_array<const W: usize, const H: usize>(
    grid: &mut [[u32; W]; H],
) -> u64 {
    assert!(W > 0 && H > 0, "SandpileElement dimensions must be > 0");
    let mut total_topples = 0_u64;
    loop {
        let mut changed = false;
        for y in 0..H {
            for x in 0..W {
                let h = grid[y][x];
                if h >= CRITICAL_HEIGHT {
                    let topples = h / CRITICAL_HEIGHT;
                    grid[y][x] = h % CRITICAL_HEIGHT;
                    if x > 0 {
                        grid[y][x - 1] += topples;
                    }
                    if x + 1 < W {
                        grid[y][x + 1] += topples;
                    }
                    if y > 0 {
                        grid[y - 1][x] += topples;
                    }
                    if y + 1 < H {
                        grid[y + 1][x] += topples;
                    }
                    total_topples += u64::from(topples);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    total_topples
}

/// Computes the super-recurrent zero representative $z_0 = 2 z_{\max} - (2 z_{\max})^\circ$ on `W x H`.
pub(crate) fn creutz_dual_zero_array<const W: usize, const H: usize>() -> [[u32; W]; H] {
    let two_z_max = 2 * MAX_STABLE_HEIGHT;
    let mut relaxed = [[two_z_max; W]; H];
    let _ = stabilize_grid_array(&mut relaxed);
    let mut z0 = [[0_u32; W]; H];
    for y in 0..H {
        for x in 0..W {
            z0[y][x] = two_z_max - relaxed[y][x];
        }
    }
    z0
}

/// Computes Creutz's exact recurrent identity $e = (2 z_{\max} - (2 z_{\max})^\circ)^\circ$ on `W x H`.
pub(crate) fn creutz_identity_array<const W: usize, const H: usize>() -> [[u32; W]; H] {
    let mut e = creutz_dual_zero_array::<W, H>();
    let _ = stabilize_grid_array(&mut e);
    e
}

/// Runs Dhar's burning algorithm on a compile-time `W x H` configuration, returning
/// `(is_recurrent, wave_depths, max_depth)`.
pub(crate) fn dhar_burning_array<const W: usize, const H: usize>(
    grid: &[[u32; W]; H],
) -> (bool, [[u16; W]; H], u16) {
    let mut depths = [[0_u16; W]; H];
    let mut rem_deg = [[0_u8; W]; H];
    for (y, row) in rem_deg.iter_mut().enumerate() {
        for (x, deg) in row.iter_mut().enumerate() {
            let mut d = 0_u8;
            if x > 0 {
                d += 1;
            }
            if x + 1 < W {
                d += 1;
            }
            if y > 0 {
                d += 1;
            }
            if y + 1 < H {
                d += 1;
            }
            *deg = d;
        }
    }

    let mut burned = 0_usize;
    let mut wave = 1_u16;
    loop {
        let mut newly_burned = false;
        let prev_wave = wave - 1;
        for y in 0..H {
            for x in 0..W {
                if grid[y][x] >= CRITICAL_HEIGHT {
                    return (false, depths, 0);
                }
                if depths[y][x] == 0 && grid[y][x] >= u32::from(rem_deg[y][x]) {
                    // Enforce synchronous BFS wave depth: wave 1 burns boundary-eligible cells,
                    // subsequent waves burn cells unlocked during `prev_wave`.
                    if wave == 1 || has_neighbor_at_wave(x, y, prev_wave, &depths) {
                        depths[y][x] = wave;
                        newly_burned = true;
                        burned += 1;
                    }
                }
            }
        }
        if !newly_burned {
            break;
        }
        decrement_burned_wave_neighbors(wave, &depths, &mut rem_deg);
        wave = wave.saturating_add(1);
    }

    let max_depth = wave.saturating_sub(1);
    (burned == W * H, depths, max_depth)
}

fn has_neighbor_at_wave<const W: usize, const H: usize>(
    x: usize,
    y: usize,
    target_wave: u16,
    depths: &[[u16; W]; H],
) -> bool {
    (x > 0 && depths[y][x - 1] == target_wave)
        || (x + 1 < W && depths[y][x + 1] == target_wave)
        || (y > 0 && depths[y - 1][x] == target_wave)
        || (y + 1 < H && depths[y + 1][x] == target_wave)
}

fn decrement_burned_wave_neighbors<const W: usize, const H: usize>(
    burned_wave: u16,
    depths: &[[u16; W]; H],
    rem_deg: &mut [[u8; W]; H],
) {
    for y in 0..H {
        for x in 0..W {
            if depths[y][x] == burned_wave {
                if x > 0 && depths[y][x - 1] == 0 {
                    rem_deg[y][x - 1] = rem_deg[y][x - 1].saturating_sub(1);
                }
                if x + 1 < W && depths[y][x + 1] == 0 {
                    rem_deg[y][x + 1] = rem_deg[y][x + 1].saturating_sub(1);
                }
                if y > 0 && depths[y - 1][x] == 0 {
                    rem_deg[y - 1][x] = rem_deg[y - 1][x].saturating_sub(1);
                }
                if y + 1 < H && depths[y + 1][x] == 0 {
                    rem_deg[y + 1][x] = rem_deg[y + 1][x].saturating_sub(1);
                }
            }
        }
    }
}
