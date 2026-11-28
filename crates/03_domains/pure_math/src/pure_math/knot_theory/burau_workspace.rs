//! Pre-allocated zero-allocation numerical workspace for reduced Burau braid stepping.

// [cite:knot_braid_topology]

use super::KnotError;
use crate::pure_math::algebra::{BraidWord, MAX_BRAID_STRANDS};
use math_commons::math_kernel::types::flatten_2d_index;
use math_memory::{ArenaLayout, ArenaRegion, OwnedByteArena};

/// Pre-allocated $(n - 1) \times (n - 1)$ reduced Burau matrix workspace enabling
/// zero-allocation generator stepping inside `VerifiedEngine::run_verified`.
#[derive(Clone, Debug, PartialEq)]
pub struct BurauWorkspace {
    strands: usize,
    dim: usize,
    arena: OwnedByteArena,
    matrix: ArenaRegion<f64>,
    scratch: ArenaRegion<f64>,
}

impl BurauWorkspace {
    /// Allocates a workspace for `strands`-strand braids (`1..=MAX_BRAID_STRANDS`),
    /// initialized to the identity matrix $I_{n-1}$.
    ///
    /// # Errors
    /// Returns [`KnotError::InvalidParameters`] if `strands` is outside `1..=MAX_BRAID_STRANDS`.
    #[verified_engine::verified]
    pub fn new(strands: usize) -> Result<Self, KnotError> {
        if !(1..=MAX_BRAID_STRANDS).contains(&strands) {
            return Err(KnotError::InvalidParameters {
                reason: "Burau workspace strand count must lie in 1..=16",
            });
        }
        let dim = strands - 1;
        let len = dim * dim;
        let mut layout = ArenaLayout::new();
        let matrix = layout
            .reserve::<f64>(len)
            .expect("Burau matrix region layout valid");
        let scratch = layout
            .reserve::<f64>(len)
            .expect("Burau scratch region layout valid");
        let mut ws = Self {
            strands,
            dim,
            arena: OwnedByteArena::from_layout(&layout),
            matrix,
            scratch,
        };
        ws.reset_identity();
        assert_eq!(ws.arena.region_slice(ws.matrix).len(), len);
        assert_eq!(ws.arena.region_slice(ws.scratch).len(), len);
        Ok(ws)
    }

    /// Resets the internal $(n - 1) \times (n - 1)$ matrix to $I_{n-1}$ with zero heap allocations.
    #[verified_engine::verified]
    pub fn reset_identity(&mut self) {
        let dim = self.dim;
        let matrix = self.arena.region_slice_mut(self.matrix);
        matrix.fill(0.0);
        for i in 0..dim {
            matrix[flatten_2d_index(i, i, dim)] = 1.0;
        }
        debug_assert!(dim == 0 || matrix[0] == 1.0);
    }

    /// Dimension $n - 1$ of the reduced Burau representation.
    #[must_use]
    #[verified_engine::verified]
    pub const fn dimension(&self) -> usize {
        self.dim
    }

    /// Read-only flat row-major view of $\bar{\psi}(\beta)(t)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn matrix_slice(&self) -> &[f64] {
        self.arena.region_slice(self.matrix)
    }

    /// Multiplies the current matrix on the right by $\bar{\psi}(\sigma_i^{\pm 1})(t)$ in place
    /// with **zero heap allocations**.
    ///
    /// # Errors
    /// Returns [`KnotError::InvalidParameters`] if `generator` is invalid or `t == 0.0`.
    #[verified_engine::verified]
    pub fn step_generator_in_place(&mut self, generator: i32, t: f64) -> Result<(), KnotError> {
        let idx = generator.unsigned_abs() as usize;
        if idx == 0 || idx >= self.strands {
            return Err(KnotError::InvalidParameters {
                reason: "Burau generator index out of range for strand count",
            });
        }
        if !t.is_finite() || t.abs() < 1e-15 {
            return Err(KnotError::InvalidParameters {
                reason: "Burau parameter t must be finite and non-zero",
            });
        }
        let k = idx - 1;
        let dim = self.dim;
        let inv_t = 1.0 / t;
        let matrix = self.arena.region_slice_mut(self.matrix);

        for r in 0..dim {
            let rk = flatten_2d_index(k, r, dim);
            let m_rk = matrix[rk];
            if m_rk == 0.0 {
                continue;
            }
            if generator > 0 {
                if k > 0 {
                    matrix[flatten_2d_index(k - 1, r, dim)] += t * m_rk;
                }
                matrix[rk] = -t * m_rk;
                if k + 1 < dim {
                    matrix[flatten_2d_index(k + 1, r, dim)] += m_rk;
                }
            } else {
                if k > 0 {
                    matrix[flatten_2d_index(k - 1, r, dim)] += m_rk;
                }
                matrix[rk] = -inv_t * m_rk;
                if k + 1 < dim {
                    matrix[flatten_2d_index(k + 1, r, dim)] += inv_t * m_rk;
                }
            }
        }
        Ok(())
    }

    /// Evaluates an entire [`BraidWord`] at non-zero parameter `t` and returns the unnormalized
    /// numerical Alexander value $\det(I_{n-1} - \bar{\psi}(\beta)(t)) / (1 + t + \dots + t^{n-1})$
    /// with **zero heap allocations**.
    ///
    /// # Errors
    /// Returns [`KnotError::InvalidParameters`] if strand counts mismatch or $1 + t + \dots + t^{n-1} \approx 0$.
    #[verified_engine::verified]
    pub fn eval_alexander_in_place(&mut self, braid: &BraidWord, t: f64) -> Result<f64, KnotError> {
        if braid.strands() != self.strands {
            return Err(KnotError::InvalidParameters {
                reason: "braid strand count does not match BurauWorkspace",
            });
        }
        self.reset_identity();
        for &g in braid.generators() {
            self.step_generator_in_place(g, t)?;
        }
        if self.dim == 0 {
            return Ok(1.0);
        }
        let mut denom = 0.0;
        let mut p = 1.0;
        for _ in 0..self.strands {
            denom += p;
            p *= t;
        }
        if denom.abs() < 1e-12 {
            return Err(KnotError::InvalidParameters {
                reason: "cyclotomic denominator vanishes at chosen parameter t",
            });
        }
        let dim = self.dim;
        let mat_reg = self.matrix;
        let scr_reg = self.scratch;
        let mut cur = self.arena.raw_cursor();
        let matrix = cur.region_mut(mat_reg);
        let scratch = cur.region_mut(scr_reg);
        for i in 0..dim {
            for j in 0..dim {
                let ij = flatten_2d_index(j, i, dim);
                scratch[ij] = if i == j {
                    1.0 - matrix[ij]
                } else {
                    -matrix[ij]
                };
            }
        }
        let val = gaussian_det_in_place(scratch, dim) / denom;
        assert!(val.is_finite());
        Ok(val)
    }
}

#[verified_engine::verified]
fn gaussian_det_in_place(buf: &mut [f64], n: usize) -> f64 {
    let mut det_sign = 1.0;
    for k in 0..n {
        let mut piv_row = k;
        let mut max_abs = buf[flatten_2d_index(k, k, n)].abs();
        for r in (k + 1)..n {
            let v = buf[flatten_2d_index(k, r, n)].abs();
            if v > max_abs {
                max_abs = v;
                piv_row = r;
            }
        }
        if max_abs < 1e-15 {
            return 0.0;
        }
        if piv_row != k {
            for c in k..n {
                buf.swap(flatten_2d_index(c, k, n), flatten_2d_index(c, piv_row, n));
            }
            det_sign = -det_sign;
        }
        let piv = buf[flatten_2d_index(k, k, n)];
        for i in (k + 1)..n {
            let factor = buf[flatten_2d_index(k, i, n)] / piv;
            for j in (k + 1)..n {
                let kj = buf[flatten_2d_index(j, k, n)];
                buf[flatten_2d_index(j, i, n)] -= factor * kj;
            }
        }
    }
    let mut prod = det_sign;
    for i in 0..n {
        prod *= buf[flatten_2d_index(i, i, n)];
    }
    debug_assert!(prod.is_finite());
    prod
}
