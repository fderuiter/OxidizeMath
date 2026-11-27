//! Pre-allocated zero-allocation numerical workspace for reduced Burau braid stepping.

// [cite:knot_braid_topology]

use super::KnotError;
use crate::pure_math::algebra::{BraidWord, MAX_BRAID_STRANDS};
use math_commons::math_kernel::types::flatten_2d_index;
use math_scalars::Dual;

/// Pre-allocated $(n - 1) \times (n - 1)$ reduced Burau matrix workspace enabling
/// zero-allocation generator stepping and singular L'Hopital evaluation at $t = -1.0$
/// inside `VerifiedEngine::run_verified`.
#[derive(Clone, Debug, PartialEq)]
pub struct BurauWorkspace {
    strands: usize,
    dim: usize,
    matrix: Vec<f64>,
    scratch: Vec<f64>,
    dual_scratch: Vec<Dual<f64>>,
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
        let mut ws = Self {
            strands,
            dim,
            matrix: vec![0.0; dim * dim],
            scratch: vec![0.0; dim * dim],
            dual_scratch: vec![Dual::new(0.0, 0.0); dim * dim],
        };
        ws.reset_identity();
        assert_eq!(ws.matrix.len(), dim * dim);
        assert_eq!(ws.scratch.len(), dim * dim);
        assert_eq!(ws.dual_scratch.len(), dim * dim);
        Ok(ws)
    }

    /// Resets the internal $(n - 1) \times (n - 1)$ matrix to $I_{n-1}$ with zero heap allocations.
    #[verified_engine::verified]
    pub fn reset_identity(&mut self) {
        self.matrix.fill(0.0);
        for i in 0..self.dim {
            self.matrix[flatten_2d_index(i, i, self.dim)] = 1.0;
        }
        debug_assert!(self.dim == 0 || self.matrix[0] == 1.0);
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
        &self.matrix
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

        for r in 0..dim {
            let rk = flatten_2d_index(k, r, dim);
            let m_rk = self.matrix[rk];
            if m_rk == 0.0 {
                continue;
            }
            if generator > 0 {
                if k > 0 {
                    self.matrix[flatten_2d_index(k - 1, r, dim)] += t * m_rk;
                }
                self.matrix[rk] = -t * m_rk;
                if k + 1 < dim {
                    self.matrix[flatten_2d_index(k + 1, r, dim)] += m_rk;
                }
            } else {
                if k > 0 {
                    self.matrix[flatten_2d_index(k - 1, r, dim)] += m_rk;
                }
                self.matrix[rk] = -inv_t * m_rk;
                if k + 1 < dim {
                    self.matrix[flatten_2d_index(k + 1, r, dim)] += inv_t * m_rk;
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
        for i in 0..self.dim {
            for j in 0..self.dim {
                let ij = flatten_2d_index(j, i, self.dim);
                self.scratch[ij] = if i == j {
                    1.0 - self.matrix[ij]
                } else {
                    -self.matrix[ij]
                };
            }
        }
        let val = gaussian_det_in_place(&mut self.scratch, self.dim) / denom;
        assert!(val.is_finite());
        Ok(val)
    }

    #[verified_engine::verified]
    fn eval_singular_det_deriv_in_place(&mut self, braid: &BraidWord, t: f64) -> f64 {
        let dim = self.dim;
        self.dual_scratch.fill(Dual::new(0.0, 0.0));
        for i in 0..dim {
            self.dual_scratch[flatten_2d_index(i, i, dim)] = Dual::new(1.0, 0.0);
        }
        let t_dual = Dual::new(t, 1.0);
        let neg_t = -t_dual;
        let inv_t = Dual::new(1.0, 0.0) / t_dual;
        let neg_inv_t = -inv_t;

        for &g in braid.generators() {
            let k = (g.unsigned_abs() as usize) - 1;
            for r in 0..dim {
                let rk = flatten_2d_index(k, r, dim);
                let m_rk = self.dual_scratch[rk];
                if m_rk.re == 0.0 && m_rk.eps == 0.0 {
                    continue;
                }
                if g > 0 {
                    if k > 0 {
                        let idx_l = flatten_2d_index(k - 1, r, dim);
                        self.dual_scratch[idx_l] += t_dual * m_rk;
                    }
                    self.dual_scratch[rk] = neg_t * m_rk;
                    if k + 1 < dim {
                        let idx_r = flatten_2d_index(k + 1, r, dim);
                        self.dual_scratch[idx_r] += m_rk;
                    }
                } else {
                    if k > 0 {
                        let idx_l = flatten_2d_index(k - 1, r, dim);
                        self.dual_scratch[idx_l] += m_rk;
                    }
                    self.dual_scratch[rk] = neg_inv_t * m_rk;
                    if k + 1 < dim {
                        let idx_r = flatten_2d_index(k + 1, r, dim);
                        self.dual_scratch[idx_r] += inv_t * m_rk;
                    }
                }
            }
        }
        let mut norm = 1.0_f64;
        for i in 0..dim {
            for j in 0..dim {
                let ij = flatten_2d_index(j, i, dim);
                let v = if i == j {
                    Dual::new(1.0, 0.0) - self.dual_scratch[ij]
                } else {
                    -self.dual_scratch[ij]
                };
                norm = norm.max(v.re.abs());
                self.dual_scratch[ij] = v;
            }
        }
        dual_gaussian_det_deriv_in_place(&mut self.dual_scratch, dim, 1e-11 * norm)
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
fn dual_gaussian_det_deriv_in_place(buf: &mut [Dual<f64>], n: usize, tol: f64) -> f64 {
    let mut det_sign = 1.0_f64;
    for k in 0..n {
        let (piv_r, piv_c, max_re) = find_full_dual_pivot(buf, n, k);
        if max_re <= tol {
            if k + 1 < n {
                return 0.0;
            }
            let mut primal_prod = det_sign;
            for i in 0..(n - 1) {
                primal_prod *= buf[flatten_2d_index(i, i, n)].re;
            }
            return primal_prod * buf[flatten_2d_index(n - 1, n - 1, n)].eps;
        }
        if piv_r != k {
            for c in k..n {
                buf.swap(flatten_2d_index(c, k, n), flatten_2d_index(c, piv_r, n));
            }
            det_sign = -det_sign;
        }
        if piv_c != k {
            for r in k..n {
                buf.swap(flatten_2d_index(k, r, n), flatten_2d_index(piv_c, r, n));
            }
            det_sign = -det_sign;
        }
        let piv = buf[flatten_2d_index(k, k, n)];
        for r in (k + 1)..n {
            let factor = buf[flatten_2d_index(k, r, n)] / piv;
            for c in (k + 1)..n {
                let kc = buf[flatten_2d_index(c, k, n)];
                buf[flatten_2d_index(c, r, n)] -= factor * kc;
            }
        }
    }
    let mut prod = Dual::new(det_sign, 0.0);
    for i in 0..n {
        prod *= buf[flatten_2d_index(i, i, n)];
    }
    prod.eps
}

#[verified_engine::verified]
fn find_full_dual_pivot(buf: &[Dual<f64>], n: usize, k: usize) -> (usize, usize, f64) {
    let mut piv_r = k;
    let mut piv_c = k;
    let mut max_re = buf[flatten_2d_index(k, k, n)].re.abs();
    for r in k..n {
        for c in k..n {
            let v = buf[flatten_2d_index(c, r, n)].re.abs();
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
