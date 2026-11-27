//! Zero-allocation, `no_std`-compatible (`core`-slice) Thomas algorithm for complex tridiagonal systems.
//!
//! Solves $N \times N$ tridiagonal linear systems $A \mathbf{x} = \mathbf{b}$ in $O(N)$ time and
//! $O(1)$ stack overhead using caller-supplied `&[Complex<f64>]` and `&mut [Complex<f64>]` slices:
//!
//! $$
//! \begin{pmatrix}
//! b_0 & c_0 & 0 & \cdots & 0 \\
//! a_0 & b_1 & c_1 & \ddots & \vdots \\
//! 0 & a_1 & b_2 & \ddots & 0 \\
//! \vdots & \ddots & \ddots & \ddots & c_{N-2} \\
//! 0 & \cdots & 0 & a_{N-2} & b_{N-1}
//! \end{pmatrix}
//! \begin{pmatrix} x_0 \\ x_1 \\ x_2 \\ \vdots \\ x_{N-1} \end{pmatrix}
//! =
//! \begin{pmatrix} r_0 \\ r_1 \\ r_2 \\ \vdots \\ r_{N-1} \end{pmatrix}
//! $$

use num_complex::Complex;

#[inline]
#[verified_engine::verified]
fn is_finite_complex(z: Complex<f64>) -> bool {
    z.re.is_finite() && z.im.is_finite()
}

#[inline]
#[verified_engine::verified]
fn all_finite_complex(slice: &[Complex<f64>]) -> bool {
    let mut i = 0;
    let n = slice.len();
    while i < n {
        if !is_finite_complex(slice[i]) {
            return false;
        }
        i += 1;
    }
    true
}

#[inline]
#[verified_engine::verified]
fn invert_pivot(pivot: Complex<f64>) -> Option<Complex<f64>> {
    if !is_finite_complex(pivot) {
        return None;
    }
    let scale = pivot.re.abs().max(pivot.im.abs());
    if scale <= f64::MIN_POSITIVE {
        return None;
    }
    let inv_scale = 1.0 / scale;
    let sr = pivot.re * inv_scale;
    let si = pivot.im * inv_scale;
    let denom = sr * sr + si * si;
    if denom <= 0.0 || !denom.is_finite() {
        return None;
    }
    let factor = inv_scale / denom;
    let inv = Complex::new(sr * factor, -si * factor);
    if is_finite_complex(inv) {
        Some(inv)
    } else {
        None
    }
}

/// Precomputes Thomas LU elimination factors `(c_prime, inv_denom)` for a complex tridiagonal matrix.
///
/// Operates exclusively over caller-supplied `core` slices with zero heap allocations. Returns
/// `false` before performing any division by zero when slice dimensions mismatch, any entry is
/// non-finite, or a zero/underflowing pivot is encountered.
///
/// # Example
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::num_complex::Complex;
/// use math_explorer::pure_math::analysis::pde::tridiagonal::{
///     factor_tridiagonal, solve_tridiagonal_factored,
/// };
///
/// let lower = [Complex::new(-1.0, 0.0); 2];
/// let main = [Complex::new(2.0, 0.5); 3];
/// let upper = [Complex::new(-1.0, 0.0); 2];
/// let rhs = [
///     Complex::new(1.0, 0.0),
///     Complex::new(0.0, 1.0),
///     Complex::new(1.0, 0.0),
/// ];
/// let mut c_prime = [Complex::new(0.0, 0.0); 2];
/// let mut inv_denom = [Complex::new(0.0, 0.0); 3];
/// let mut scratch = [Complex::new(0.0, 0.0); 3];
/// let mut x = [Complex::new(0.0, 0.0); 3];
///
/// assert!(factor_tridiagonal(&lower, &main, &upper, &mut c_prime, &mut inv_denom));
/// assert!(solve_tridiagonal_factored(&lower, &c_prime, &inv_denom, &rhs, &mut scratch, &mut x));
/// assert!(((main[0] * x[0] + upper[0] * x[1]) - rhs[0]).norm() < 1e-12);
/// ```
#[verified_engine::verified]
pub fn factor_tridiagonal(
    lower: &[Complex<f64>],
    main: &[Complex<f64>],
    upper: &[Complex<f64>],
    c_prime: &mut [Complex<f64>],
    inv_denom: &mut [Complex<f64>],
) -> bool {
    let n = main.len();
    if n == 0 {
        return true;
    }
    let n_off = n - 1;
    if lower.len() < n_off
        || upper.len() < n_off
        || c_prime.len() < n_off
        || inv_denom.len() < n
        || !all_finite_complex(main)
        || !all_finite_complex(&lower[..n_off])
        || !all_finite_complex(&upper[..n_off])
    {
        return false;
    }

    let Some(inv0) = invert_pivot(main[0]) else {
        return false;
    };
    inv_denom[0] = inv0;
    if n > 1 {
        c_prime[0] = upper[0] * inv0;
    }

    for i in 1..n {
        let pivot = main[i] - lower[i - 1] * c_prime[i - 1];
        let Some(inv_p) = invert_pivot(pivot) else {
            return false;
        };
        inv_denom[i] = inv_p;
        if i < n_off {
            c_prime[i] = upper[i] * inv_p;
        }
    }
    true
}

#[verified_engine::verified]
fn forward_sweep(
    lower: &[Complex<f64>],
    c_prime: &[Complex<f64>],
    inv_denom: &[Complex<f64>],
    rhs: &[Complex<f64>],
    scratch: &mut [Complex<f64>],
) -> bool {
    let n = rhs.len();
    let n_off = n - 1;
    if lower.len() < n_off
        || c_prime.len() < n_off
        || inv_denom.len() < n
        || scratch.len() < n
        || !all_finite_complex(rhs)
        || !all_finite_complex(&inv_denom[..n])
        || !all_finite_complex(&c_prime[..n_off])
    {
        return false;
    }
    let mut d_prev = rhs[0] * inv_denom[0];
    scratch[0] = d_prev;
    for i in 1..n {
        d_prev = (rhs[i] - lower[i - 1] * d_prev) * inv_denom[i];
        scratch[i] = d_prev;
    }
    true
}

#[verified_engine::verified]
fn backward_sweep(
    c_prime: &[Complex<f64>],
    scratch: &[Complex<f64>],
    out: &mut [Complex<f64>],
) -> bool {
    let n = scratch.len();
    let n_off = n - 1;
    let mut x_next = scratch[n_off];
    out[n_off] = x_next;
    for i in (0..n_off).rev() {
        x_next = scratch[i] - c_prime[i] * x_next;
        out[i] = x_next;
    }
    all_finite_complex(&out[..n])
}

/// Solves the tridiagonal system $a_{i-1} x_{i-1} + b_i x_i + c_i x_{i+1} = d_i$ ($A \mathbf{x} = \mathbf{d}$) in $O(N)$ time using precomputed Thomas factors from [`factor_tridiagonal`].
///
/// Reference: Higham (2002), DOI: 10.1137/1.9780898718027; [`docs/math_specs/pure_math.md`](../../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Unpivoted Thomas forward/backward sweeps are unconditionally stable on diagonally dominant and unitary Cayley $(I + i\alpha H)$ operators; non-finite intermediates fail closed returning `false`.
///
/// # Domain of Validity
/// Requires finite complex slices with `solution.len() >= rhs.len()`, `scratch.len() >= rhs.len()`, `inv_denom.len() >= rhs.len()`, and off-diagonal factor slices of length $\ge \max(0, N - 1)$.
#[verified_engine::verified]
pub fn solve_tridiagonal_factored(
    lower: &[Complex<f64>],
    c_prime: &[Complex<f64>],
    inv_denom: &[Complex<f64>],
    rhs: &[Complex<f64>],
    scratch: &mut [Complex<f64>],
    solution: &mut [Complex<f64>],
) -> bool {
    let n = rhs.len();
    if n == 0 {
        return true;
    }
    if solution.len() < n || !forward_sweep(lower, c_prime, inv_denom, rhs, scratch) {
        return false;
    }
    backward_sweep(c_prime, &scratch[..n], &mut solution[..n])
}

/// Solves $a_{i-1} x_{i-1} + b_i x_i + c_i x_{i+1} = d_i$ ($A \mathbf{x} = \mathbf{d}$) in place over `rhs_and_sol` using precomputed Thomas factors and `scratch`.
///
/// Reference: Higham (2002), DOI: 10.1137/1.9780898718027; [`docs/math_specs/pure_math.md`](../../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Preserves $O(N)$ backward error for diagonally dominant and Crank-Nicolson tridiagonal operators with zero heap allocations, returning `false` on non-finite sweep values.
///
/// # Domain of Validity
/// Requires finite complex slices with `scratch.len() >= rhs_and_sol.len()`, `inv_denom.len() >= rhs_and_sol.len()`, and `lower.len(), c_prime.len() >= rhs_and_sol.len().saturating_sub(1)`.
#[verified_engine::verified]
pub fn solve_tridiagonal_factored_in_place(
    lower: &[Complex<f64>],
    c_prime: &[Complex<f64>],
    inv_denom: &[Complex<f64>],
    rhs_and_sol: &mut [Complex<f64>],
    scratch: &mut [Complex<f64>],
) -> bool {
    let n = rhs_and_sol.len();
    if n == 0 {
        return true;
    }
    if !forward_sweep(lower, c_prime, inv_denom, rhs_and_sol, scratch) {
        return false;
    }
    backward_sweep(c_prime, &scratch[..n], rhs_and_sol)
}

/// Factors `(lower, main, upper)` via $w_i = b_i - a_{i-1} c'_{i-1}$ and solves $A \mathbf{x} = \mathbf{b}$ in place over `rhs_and_sol`.
///
/// Reference: Higham (2002), DOI: 10.1137/1.9780898718027; [`docs/math_specs/pure_math.md`](../../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Rejects singular or ill-conditioned Schur pivots ($|w_i|^2 \le 10^{-300}$) and non-finite entries in [`factor_tridiagonal`], returning `false` without panicking or allocating.
///
/// # Domain of Validity
/// Requires `rhs_and_sol.len() == main.len()`, `inv_denom.len() >= N`, `scratch.len() >= N`, off-diagonal buffers of length $\ge \max(0, N - 1)$, and non-vanishing leading principal minors.
#[verified_engine::verified]
pub fn solve_tridiagonal_in_place(
    lower: &[Complex<f64>],
    main: &[Complex<f64>],
    upper: &[Complex<f64>],
    rhs_and_sol: &mut [Complex<f64>],
    c_prime: &mut [Complex<f64>],
    inv_denom: &mut [Complex<f64>],
    scratch: &mut [Complex<f64>],
) -> bool {
    let n = main.len();
    if rhs_and_sol.len() != n {
        return false;
    }
    if !factor_tridiagonal(lower, main, upper, c_prime, inv_denom) {
        return false;
    }
    solve_tridiagonal_factored_in_place(lower, c_prime, inv_denom, rhs_and_sol, scratch)
}

/// Evaluates the tridiagonal matrix-vector product $\mathbf{y} = B \mathbf{x}$ into `out` in $O(N)$ time.
///
/// Handles $N = 0$, $N = 1$, and $N \ge 2$ grids with zero heap allocations.
#[verified_engine::verified]
pub fn apply_tridiagonal(
    lower: &[Complex<f64>],
    main: &[Complex<f64>],
    upper: &[Complex<f64>],
    x: &[Complex<f64>],
    out: &mut [Complex<f64>],
) -> bool {
    let n = main.len();
    if x.len() != n || out.len() < n {
        return false;
    }
    if n == 0 {
        return true;
    }
    let n_off = n - 1;
    if lower.len() < n_off || upper.len() < n_off {
        return false;
    }
    if n == 1 {
        out[0] = main[0] * x[0];
        return is_finite_complex(out[0]);
    }
    out[0] = main[0] * x[0] + upper[0] * x[1];
    for i in 1..n_off {
        out[i] = lower[i - 1] * x[i - 1] + main[i] * x[i] + upper[i] * x[i + 1];
    }
    out[n_off] = lower[n_off - 1] * x[n_off - 1] + main[n_off] * x[n_off];
    all_finite_complex(&out[..n])
}

#[cfg(test)]
#[path = "tridiagonal_tests.rs"]
mod tests;
