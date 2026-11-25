// [cite:quantum_mechanics]
//! Preallocated complex Thomas-sweep workspace backed by [`OwnedByteArena`] for 1D Cayley propagators.

use math_memory::{ArenaLayout, OwnedByteArena};
use num_complex::Complex;
use pure_math::pure_math::analysis::pde::tridiagonal::{
    apply_tridiagonal, factor_tridiagonal, solve_tridiagonal_factored,
};

/// Preallocated complex band and Thomas-sweep scratch buffers backed by [`OwnedByteArena`]
/// for zero-allocation 1D Cayley stepping.
#[derive(Clone, Debug)]
pub struct TridiagonalWorkspace {
    arena: OwnedByteArena,
    max_points: usize,
}

impl TridiagonalWorkspace {
    fn required_bytes(max_points: usize) -> usize {
        let off_cap = max_points.saturating_sub(1);
        let mut layout = ArenaLayout::new();
        for len in [
            off_cap, max_points, off_cap, max_points, off_cap, max_points, max_points, max_points,
        ] {
            let _ = layout
                .add_slice::<Complex<f64>>(len)
                .expect("TridiagonalWorkspace layout");
        }
        layout.total_bytes()
    }

    /// Preallocates all complex tridiagonal bands and Thomas scratch buffers in a single [`OwnedByteArena`].
    #[verified_engine::verified]
    #[must_use]
    pub fn with_capacity(max_points: usize) -> Self {
        Self {
            arena: OwnedByteArena::with_capacity(Self::required_bytes(max_points)),
            max_points,
        }
    }

    /// Maximum grid capacity currently allocated without triggering heap reallocation.
    #[verified_engine::verified]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.max_points
    }

    /// Ensures workspace buffers hold at least `n` grid points (zero heap allocations if `n <= capacity()`).
    #[verified_engine::verified]
    pub fn ensure_capacity(&mut self, n: usize) {
        if n > self.max_points {
            self.arena.ensure_capacity(Self::required_bytes(n));
            self.max_points = n;
        }
        debug_assert!(self.max_points >= n);
    }

    #[allow(clippy::type_complexity)]
    fn slices_mut(
        &mut self,
    ) -> (
        &mut [Complex<f64>],
        &mut [Complex<f64>],
        &mut [Complex<f64>],
        &mut [Complex<f64>],
        &mut [Complex<f64>],
        &mut [Complex<f64>],
        &mut [Complex<f64>],
        &mut [Complex<f64>],
    ) {
        let cap = self.max_points;
        let off_cap = cap.saturating_sub(1);
        let mut scope = self.arena.reborrow();
        let lhs_off = scope
            .retain_slice::<Complex<f64>>(off_cap)
            .expect("lhs_off");
        let lhs_main = scope.retain_slice::<Complex<f64>>(cap).expect("lhs_main");
        let rhs_off = scope
            .retain_slice::<Complex<f64>>(off_cap)
            .expect("rhs_off");
        let rhs_main = scope.retain_slice::<Complex<f64>>(cap).expect("rhs_main");
        let c_prime = scope
            .retain_slice::<Complex<f64>>(off_cap)
            .expect("c_prime");
        let inv_denom = scope.retain_slice::<Complex<f64>>(cap).expect("inv_denom");
        let rhs_vec = scope.retain_slice::<Complex<f64>>(cap).expect("rhs_vec");
        let sweep_scratch = scope
            .retain_slice::<Complex<f64>>(cap)
            .expect("sweep_scratch");
        (
            lhs_off,
            lhs_main,
            rhs_off,
            rhs_main,
            c_prime,
            inv_denom,
            rhs_vec,
            sweep_scratch,
        )
    }

    #[verified_engine::verified]
    pub(crate) fn poison(&mut self, n: usize) {
        self.ensure_capacity(n);
        let nan = Complex::new(f64::NAN, f64::NAN);
        let zero = Complex::new(0.0, 0.0);
        let n_off = n.saturating_sub(1);
        let (lhs_off, lhs_main, rhs_off, rhs_main, c_prime, inv_denom, _, _) = self.slices_mut();
        lhs_main[..n].fill(nan);
        rhs_main[..n].fill(nan);
        inv_denom[..n].fill(nan);
        lhs_off[..n_off].fill(zero);
        rhs_off[..n_off].fill(zero);
        c_prime[..n_off].fill(zero);
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
        let (lhs_off, lhs_main, rhs_off, rhs_main, c_prime, inv_denom, _, _) = self.slices_mut();
        for (i, &d) in diag.iter().enumerate() {
            if !d.is_finite() {
                self.poison(n);
                return false;
            }
            lhs_main[i] = Complex::new(1.0, alpha * d);
            rhs_main[i] = Complex::new(1.0, -alpha * d);
        }
        for (i, &e) in off.iter().enumerate() {
            if !e.is_finite() {
                self.poison(n);
                return false;
            }
            lhs_off[i] = Complex::new(0.0, alpha * e);
            rhs_off[i] = Complex::new(0.0, -alpha * e);
        }
        let ok = factor_tridiagonal(
            &lhs_off[..n_off],
            &lhs_main[..n],
            &lhs_off[..n_off],
            &mut c_prime[..n_off],
            &mut inv_denom[..n],
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
        let (lhs_off, _lhs_main, rhs_off, rhs_main, c_prime, inv_denom, rhs_vec, sweep_scratch) =
            self.slices_mut();
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
