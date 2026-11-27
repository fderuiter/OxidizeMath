// [cite:quantum_mechanics]
//! Preallocated complex Thomas-sweep workspace for 1D Crank-Nicolson Cayley propagators.

use num_complex::Complex;
use pure_math::pure_math::analysis::pde::tridiagonal::{
    apply_tridiagonal, factor_tridiagonal, solve_tridiagonal_factored,
};

/// Preallocated complex band and Thomas-sweep scratch buffers for zero-allocation 1D Cayley stepping.
#[derive(Clone, Debug)]
pub struct SolverWorkspace {
    lhs_off: Vec<Complex<f64>>,
    lhs_main: Vec<Complex<f64>>,
    rhs_off: Vec<Complex<f64>>,
    rhs_main: Vec<Complex<f64>>,
    c_prime: Vec<Complex<f64>>,
    inv_denom: Vec<Complex<f64>>,
    rhs_vec: Vec<Complex<f64>>,
    sweep_scratch: Vec<Complex<f64>>,
}

impl SolverWorkspace {
    /// Preallocates all complex tridiagonal bands and Thomas scratch buffers for up to `max_points`.
    #[verified_engine::verified]
    #[must_use]
    pub fn with_capacity(max_points: usize) -> Self {
        let zero = Complex::new(0.0, 0.0);
        let off_cap = max_points.saturating_sub(1);
        debug_assert_eq!(off_cap, max_points.saturating_sub(1));
        Self {
            lhs_off: vec![zero; off_cap],
            lhs_main: vec![zero; max_points],
            rhs_off: vec![zero; off_cap],
            rhs_main: vec![zero; max_points],
            c_prime: vec![zero; off_cap],
            inv_denom: vec![zero; max_points],
            rhs_vec: vec![zero; max_points],
            sweep_scratch: vec![zero; max_points],
        }
    }

    /// Maximum grid capacity currently allocated without triggering heap reallocation.
    #[verified_engine::verified]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.lhs_main.len()
    }

    /// Ensures workspace buffers hold at least `n` grid points (zero heap allocations if `n <= capacity()`).
    #[verified_engine::verified]
    pub fn ensure_capacity(&mut self, n: usize) {
        let zero = Complex::new(0.0, 0.0);
        if n > self.lhs_main.len() {
            self.lhs_main.resize(n, zero);
            self.rhs_main.resize(n, zero);
            self.inv_denom.resize(n, zero);
            self.rhs_vec.resize(n, zero);
            self.sweep_scratch.resize(n, zero);
        }
        let n_off = n.saturating_sub(1);
        if n_off > self.lhs_off.len() {
            self.lhs_off.resize(n_off, zero);
            self.rhs_off.resize(n_off, zero);
            self.c_prime.resize(n_off, zero);
        }
        debug_assert!(self.lhs_main.len() >= n && self.lhs_off.len() >= n_off);
    }

    #[verified_engine::verified]
    pub(crate) fn poison(&mut self, n: usize) {
        let nan = Complex::new(f64::NAN, f64::NAN);
        let zero = Complex::new(0.0, 0.0);
        self.lhs_main[..n].fill(nan);
        self.rhs_main[..n].fill(nan);
        self.inv_denom[..n].fill(nan);
        let n_off = n.saturating_sub(1);
        self.lhs_off[..n_off].fill(zero);
        self.rhs_off[..n_off].fill(zero);
        self.c_prime[..n_off].fill(zero);
    }

    #[verified_engine::verified]
    pub(crate) fn factor_cayley(&mut self, diag: &[f64], off: &[f64], alpha: f64) -> bool {
        let n = diag.len();
        self.ensure_capacity(n);
        if n == 0 {
            return true;
        }
        let n_off = n - 1;
        if off.len() != n_off || !alpha.is_finite() {
            self.poison(n);
            return false;
        }
        for (i, &d) in diag.iter().enumerate() {
            if !d.is_finite() {
                self.poison(n);
                return false;
            }
            self.lhs_main[i] = Complex::new(1.0, alpha * d);
            self.rhs_main[i] = Complex::new(1.0, -alpha * d);
        }
        for (i, &e) in off.iter().enumerate() {
            if !e.is_finite() {
                self.poison(n);
                return false;
            }
            self.lhs_off[i] = Complex::new(0.0, alpha * e);
            self.rhs_off[i] = Complex::new(0.0, -alpha * e);
        }
        let ok = factor_tridiagonal(
            &self.lhs_off[..n_off],
            &self.lhs_main[..n],
            &self.lhs_off[..n_off],
            &mut self.c_prime[..n_off],
            &mut self.inv_denom[..n],
        );
        if !ok {
            self.poison(n);
        }
        ok
    }

    #[verified_engine::verified]
    pub(crate) fn sweep_cayley(&mut self, n: usize, psi: &mut [Complex<f64>]) -> bool {
        if psi.len() != n {
            return false;
        }
        if n == 0 {
            return true;
        }
        if n == 1 {
            psi[0] = (self.rhs_main[0] * psi[0]) * self.inv_denom[0];
            return psi[0].re.is_finite() && psi[0].im.is_finite();
        }
        let n_off = n - 1;
        if !apply_tridiagonal(
            &self.rhs_off[..n_off],
            &self.rhs_main[..n],
            &self.rhs_off[..n_off],
            psi,
            &mut self.rhs_vec[..n],
        ) {
            psi.fill(Complex::new(f64::NAN, f64::NAN));
            return false;
        }
        let ok = solve_tridiagonal_factored(
            &self.lhs_off[..n_off],
            &self.c_prime[..n_off],
            &self.inv_denom[..n],
            &self.rhs_vec[..n],
            &mut self.sweep_scratch[..n],
            psi,
        );
        if !ok {
            psi.fill(Complex::new(f64::NAN, f64::NAN));
        }
        ok
    }
}
