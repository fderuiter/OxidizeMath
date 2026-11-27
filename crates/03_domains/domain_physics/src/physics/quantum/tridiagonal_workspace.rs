// [cite:quantum_mechanics]
//! Preallocated complex Thomas-sweep workspace for 1D Crank-Nicolson Cayley propagators.

use math_memory::{ArenaLayout, ArenaRegion, OwnedByteArena};
use num_complex::Complex;
use pure_math::pure_math::analysis::pde::tridiagonal::{
    apply_tridiagonal, factor_tridiagonal, solve_tridiagonal_factored,
};

/// Preallocated complex band and Thomas-sweep scratch buffers for zero-allocation 1D Cayley stepping.
#[derive(Clone, Debug)]
pub struct TridiagonalWorkspace {
    arena: OwnedByteArena,
    max_points: usize,
    lhs_off: ArenaRegion<Complex<f64>>,
    lhs_main: ArenaRegion<Complex<f64>>,
    rhs_off: ArenaRegion<Complex<f64>>,
    rhs_main: ArenaRegion<Complex<f64>>,
    c_prime: ArenaRegion<Complex<f64>>,
    inv_denom: ArenaRegion<Complex<f64>>,
    rhs_vec: ArenaRegion<Complex<f64>>,
    sweep_scratch: ArenaRegion<Complex<f64>>,
}

impl TridiagonalWorkspace {
    /// Preallocates all complex tridiagonal bands and Thomas scratch buffers in a single [`OwnedByteArena`].
    #[verified_engine::verified]
    #[must_use]
    pub fn with_capacity(max_points: usize) -> Self {
        let off_cap = max_points.saturating_sub(1);
        let mut layout = ArenaLayout::new();
        let lhs_off = layout
            .reserve::<Complex<f64>>(off_cap)
            .expect("lhs_off layout");
        let lhs_main = layout
            .reserve::<Complex<f64>>(max_points)
            .expect("lhs_main layout");
        let rhs_off = layout
            .reserve::<Complex<f64>>(off_cap)
            .expect("rhs_off layout");
        let rhs_main = layout
            .reserve::<Complex<f64>>(max_points)
            .expect("rhs_main layout");
        let c_prime = layout
            .reserve::<Complex<f64>>(off_cap)
            .expect("c_prime layout");
        let inv_denom = layout
            .reserve::<Complex<f64>>(max_points)
            .expect("inv_denom layout");
        let rhs_vec = layout
            .reserve::<Complex<f64>>(max_points)
            .expect("rhs_vec layout");
        let sweep_scratch = layout
            .reserve::<Complex<f64>>(max_points)
            .expect("sweep_scratch layout");
        Self {
            arena: OwnedByteArena::from_layout(&layout),
            max_points,
            lhs_off,
            lhs_main,
            rhs_off,
            rhs_main,
            c_prime,
            inv_denom,
            rhs_vec,
            sweep_scratch,
        }
    }

    /// Maximum grid capacity currently allocated without triggering heap reallocation.
    #[verified_engine::verified]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.max_points
    }

    /// Ensures workspace buffers hold at least `n` grid points (zero heap allocations if `n <= capacity()`).
    #[verified_engine::verified]
    pub fn ensure_capacity(&mut self, n: usize) {
        if n > self.max_points {
            *self = Self::with_capacity(n);
        }
        debug_assert!(self.max_points >= n);
    }

    #[verified_engine::verified]
    pub(crate) fn poison(&mut self, n: usize) {
        let nan = Complex::new(f64::NAN, f64::NAN);
        let zero = Complex::new(0.0, 0.0);
        let n_off = n.saturating_sub(1);
        let mut cur = self.arena.raw_cursor();
        cur.region_mut(self.lhs_off)[..n_off].fill(zero);
        cur.region_mut(self.lhs_main)[..n].fill(nan);
        cur.region_mut(self.rhs_off)[..n_off].fill(zero);
        cur.region_mut(self.rhs_main)[..n].fill(nan);
        cur.region_mut(self.c_prime)[..n_off].fill(zero);
        cur.region_mut(self.inv_denom)[..n].fill(nan);
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
        let ok = {
            let mut cur = self.arena.raw_cursor();
            let lhs_off = &mut cur.region_mut(self.lhs_off)[..n_off];
            let lhs_main = &mut cur.region_mut(self.lhs_main)[..n];
            let rhs_off = &mut cur.region_mut(self.rhs_off)[..n_off];
            let rhs_main = &mut cur.region_mut(self.rhs_main)[..n];
            let c_prime = &mut cur.region_mut(self.c_prime)[..n_off];
            let inv_denom = &mut cur.region_mut(self.inv_denom)[..n];
            for (i, &d) in diag.iter().enumerate() {
                if !d.is_finite() {
                    return self.poison_false(n);
                }
                lhs_main[i] = Complex::new(1.0, alpha * d);
                rhs_main[i] = Complex::new(1.0, -alpha * d);
            }
            for (i, &e) in off.iter().enumerate() {
                if !e.is_finite() {
                    return self.poison_false(n);
                }
                lhs_off[i] = Complex::new(0.0, alpha * e);
                rhs_off[i] = Complex::new(0.0, -alpha * e);
            }
            factor_tridiagonal(lhs_off, lhs_main, lhs_off, c_prime, inv_denom)
        };
        if !ok {
            self.poison(n);
        }
        ok
    }

    #[inline]
    #[verified_engine::verified]
    fn poison_false(&mut self, n: usize) -> bool {
        self.poison(n);
        false
    }

    #[verified_engine::verified]
    pub(crate) fn sweep_cayley(&mut self, n: usize, psi: &mut [Complex<f64>]) -> bool {
        if psi.len() != n {
            return false;
        }
        if n == 0 {
            return true;
        }
        let mut cur = self.arena.raw_cursor();
        let lhs_off = cur.region_mut(self.lhs_off);
        let _lhs_main = cur.region_mut(self.lhs_main);
        let rhs_off = cur.region_mut(self.rhs_off);
        let rhs_main = cur.region_mut(self.rhs_main);
        let c_prime = cur.region_mut(self.c_prime);
        let inv_denom = cur.region_mut(self.inv_denom);
        let rhs_vec = cur.region_mut(self.rhs_vec);
        let sweep_scratch = cur.region_mut(self.sweep_scratch);
        if n == 1 {
            psi[0] = (rhs_main[0] * psi[0]) * inv_denom[0];
            return psi[0].re.is_finite() && psi[0].im.is_finite();
        }
        let n_off = n - 1;
        if !apply_tridiagonal(
            &rhs_off[..n_off],
            &rhs_main[..n],
            &rhs_off[..n_off],
            psi,
            &mut rhs_vec[..n],
        ) {
            psi.fill(Complex::new(f64::NAN, f64::NAN));
            return false;
        }
        let ok = solve_tridiagonal_factored(
            &lhs_off[..n_off],
            &c_prime[..n_off],
            &inv_denom[..n],
            &rhs_vec[..n],
            &mut sweep_scratch[..n],
            psi,
        );
        if !ok {
            psi.fill(Complex::new(f64::NAN, f64::NAN));
        }
        ok
    }
}
