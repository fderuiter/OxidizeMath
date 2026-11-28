//! Pre-allocated zero-allocation numerical workspace for reduced Burau braid stepping.

// [cite:knot_braid_topology]

use super::KnotError;
use crate::pure_math::algebra::{BraidWord, MAX_BRAID_STRANDS};
use math_commons::math_kernel::types::flatten_2d_index;
use math_memory::{ArenaLayout, ArenaRegion, OwnedByteArena};
use math_scalars::Dual;

/// Pre-allocated $(n - 1) \times (n - 1)$ reduced Burau matrix workspace enabling
/// zero-allocation generator stepping and singular L'Hopital evaluation at $t = -1.0$
/// inside `VerifiedEngine::run_verified`.
#[derive(Clone, Debug, PartialEq)]
pub struct BurauWorkspace {
    strands: usize,
    dim: usize,
    arena: OwnedByteArena,
    matrix: ArenaRegion<f64>,
    scratch: ArenaRegion<f64>,
    dual_scratch: ArenaRegion<[f64; 2]>,
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
        let dual_scratch = layout
            .reserve::<[f64; 2]>(len)
            .expect("Burau dual scratch region layout valid");
        let mut ws = Self {
            strands,
            dim,
            arena: OwnedByteArena::from_layout(&layout),
            matrix,
            scratch,
            dual_scratch,
        };
        ws.reset_identity();
        assert_eq!(ws.arena.region_slice(ws.matrix).len(), len);
        assert_eq!(ws.arena.region_slice(ws.scratch).len(), len);
        assert_eq!(ws.arena.region_slice(ws.dual_scratch).len(), len);
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
    /// with **zero heap allocations**, resolving the removable root $t = -1.0$ on even strand counts
    /// via first-order dual Gaussian elimination $N'(-1) / (n / 2)$.
    ///
    /// # Errors
    /// Returns [`KnotError::InvalidParameters`] if strand counts mismatch or `t` is invalid.
    #[verified_engine::verified]
    pub fn eval_alexander_in_place(&mut self, braid: &BraidWord, t: f64) -> Result<f64, KnotError> {
        if braid.strands() != self.strands {
            return Err(KnotError::InvalidParameters {
                reason: "braid strand count does not match BurauWorkspace",
            });
        }
        if !t.is_finite() || t.abs() < 1e-15 {
            return Err(KnotError::InvalidParameters {
                reason: "Burau parameter t must be finite and non-zero",
            });
        }
        self.reset_identity();
        for &g in braid.generators() {
            self.step_generator_in_place(g, t)?;
        }
        if self.dim == 0 {
            return Ok(1.0);
        }
        let (denom, denom_deriv) = eval_cyclotomic_with_deriv(self.strands, t);
        if denom.abs() < 1e-12 {
            if denom_deriv.abs() < 1e-12 {
                return Err(KnotError::InvalidParameters {
                    reason: "cyclotomic denominator and derivative vanish at chosen parameter t",
                });
            }
            let num_deriv = self.eval_singular_det_deriv_in_place(braid, t);
            let val = num_deriv / denom_deriv;
            assert!(val.is_finite());
            return Ok(val);
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

    #[verified_engine::verified]
    fn eval_singular_det_deriv_in_place(&mut self, braid: &BraidWord, t: f64) -> f64 {
        let dim = self.dim;
        let dual_buf = self.arena.region_slice_mut(self.dual_scratch);
        dual_buf.fill([0.0, 0.0]);
        for i in 0..dim {
            dual_buf[flatten_2d_index(i, i, dim)] = [1.0, 0.0];
        }
        let t_dual = Dual::new(t, 1.0);
        let inv_t = Dual::new(1.0, 0.0) / t_dual;

        for &g in braid.generators() {
            step_dual_generator_in_place(dual_buf, dim, g, t_dual, inv_t);
        }
        let mut norm = 1.0_f64;
        for i in 0..dim {
            for j in 0..dim {
                let ij = flatten_2d_index(j, i, dim);
                let cur = to_dual(dual_buf[ij]);
                let v = if i == j {
                    Dual::new(1.0, 0.0) - cur
                } else {
                    -cur
                };
                norm = norm.max(v.re.abs());
                dual_buf[ij] = from_dual(v);
            }
        }
        dual_gaussian_det_deriv_in_place(dual_buf, dim, 1e-11 * norm)
    }
}

#[inline]
const fn to_dual(pair: [f64; 2]) -> Dual<f64> {
    Dual::new(pair[0], pair[1])
}

#[inline]
const fn from_dual(d: Dual<f64>) -> [f64; 2] {
    [d.re, d.eps]
}

#[verified_engine::verified]
fn step_dual_generator_in_place(
    buf: &mut [[f64; 2]],
    dim: usize,
    g: i32,
    t_dual: Dual<f64>,
    inv_t: Dual<f64>,
) {
    let k = (g.unsigned_abs() as usize) - 1;
    let neg_t = -t_dual;
    let neg_inv_t = -inv_t;
    for r in 0..dim {
        let rk = flatten_2d_index(k, r, dim);
        let m_rk = to_dual(buf[rk]);
        if m_rk.re == 0.0 && m_rk.eps == 0.0 {
            continue;
        }
        if g > 0 {
            if k > 0 {
                let idx_l = flatten_2d_index(k - 1, r, dim);
                buf[idx_l] = from_dual(to_dual(buf[idx_l]) + t_dual * m_rk);
            }
            buf[rk] = from_dual(neg_t * m_rk);
            if k + 1 < dim {
                let idx_r = flatten_2d_index(k + 1, r, dim);
                buf[idx_r] = from_dual(to_dual(buf[idx_r]) + m_rk);
            }
        } else {
            if k > 0 {
                let idx_l = flatten_2d_index(k - 1, r, dim);
                buf[idx_l] = from_dual(to_dual(buf[idx_l]) + m_rk);
            }
            buf[rk] = from_dual(neg_inv_t * m_rk);
            if k + 1 < dim {
                let idx_r = flatten_2d_index(k + 1, r, dim);
                buf[idx_r] = from_dual(to_dual(buf[idx_r]) + inv_t * m_rk);
            }
        }
    }
}

#[verified_engine::verified]
fn eval_cyclotomic_with_deriv(strands: usize, t: f64) -> (f64, f64) {
    let t_dual = Dual::new(t, 1.0);
    let mut sum = Dual::new(0.0, 0.0);
    let mut pow = Dual::new(1.0, 0.0);
    for _ in 0..strands {
        sum += pow;
        pow *= t_dual;
    }
    (sum.re, sum.eps)
}

#[verified_engine::verified]
fn dual_gaussian_det_deriv_in_place(buf: &mut [[f64; 2]], n: usize, tol: f64) -> f64 {
    let mut det_sign = 1.0_f64;
    for k in 0..n {
        let (piv_r, piv_c, max_re) = find_full_dual_pivot(buf, n, k);
        if max_re <= tol {
            if k + 1 < n {
                return 0.0;
            }
            let mut primal_prod = det_sign;
            for i in 0..(n - 1) {
                primal_prod *= buf[flatten_2d_index(i, i, n)][0];
            }
            return primal_prod * buf[flatten_2d_index(n - 1, n - 1, n)][1];
        }
        det_sign *= swap_dual_pivot(buf, n, k, piv_r, piv_c);
        let piv = to_dual(buf[flatten_2d_index(k, k, n)]);
        for r in (k + 1)..n {
            let factor = to_dual(buf[flatten_2d_index(k, r, n)]) / piv;
            for c in (k + 1)..n {
                let kc = to_dual(buf[flatten_2d_index(c, k, n)]);
                let idx = flatten_2d_index(c, r, n);
                buf[idx] = from_dual(to_dual(buf[idx]) - factor * kc);
            }
        }
    }
    let mut prod = Dual::new(det_sign, 0.0);
    for i in 0..n {
        prod *= to_dual(buf[flatten_2d_index(i, i, n)]);
    }
    prod.eps
}

#[verified_engine::verified]
fn swap_dual_pivot(buf: &mut [[f64; 2]], n: usize, k: usize, piv_r: usize, piv_c: usize) -> f64 {
    let mut sign = 1.0_f64;
    if piv_r != k {
        for c in k..n {
            buf.swap(flatten_2d_index(c, k, n), flatten_2d_index(c, piv_r, n));
        }
        sign = -sign;
    }
    if piv_c != k {
        for r in k..n {
            buf.swap(flatten_2d_index(k, r, n), flatten_2d_index(piv_c, r, n));
        }
        sign = -sign;
    }
    sign
}

#[verified_engine::verified]
fn find_full_dual_pivot(buf: &[[f64; 2]], n: usize, k: usize) -> (usize, usize, f64) {
    let mut piv_r = k;
    let mut piv_c = k;
    let mut max_re = buf[flatten_2d_index(k, k, n)][0].abs();
    for r in k..n {
        for c in k..n {
            let v = buf[flatten_2d_index(c, r, n)][0].abs();
            if v > max_re {
                max_re = v;
                piv_r = r;
                piv_c = c;
            }
        }
    }
    (piv_r, piv_c, max_re)
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
