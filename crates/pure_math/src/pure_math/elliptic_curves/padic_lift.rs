// [cite:modular_polynomials_review]
//! Zero-allocation $p$-adic Newton-Hensel ordinate lifting and Breuer valuation checks on short-Weierstrass curves.

use crate::pure_math::algebra::fields::Fp;
pub use math_scalars::{Qp, ScalarError, eval_int_poly_qp, hensel_lift_root};

/// Lifts a simple affine point $(x, y_0) \in E(\mathbb{F}_P)$ on $y^2 = x^3 + ax + b$
/// to its unique $p$-adic ordinate $y \in \mathbb{Q}_P$ up to precision $O(P^N)$ via
/// zero-allocation Newton-Hensel iteration (Hensel 1908, DOI: `10.1515/9783112342404`).
///
/// # Stability
/// Quadratically convergent in $\lceil \log_2 N \rceil + 1$ steps in the ultrametric norm $|\cdot|_P$
/// whenever $2 y_0 \not\equiv 0 \pmod P$.
///
/// # Domain of Validity
/// Requires odd prime $P > 2$, $N \ge 1$, $y_0^2 \equiv x^3 + ax + b \pmod P$, and $y_0 \not\equiv 0 \pmod P$.
///
/// # Errors
/// Returns [`ScalarError::RootNotFound`] if $(x, y_0)$ is not on $E(\mathbb{F}_P)$ or
/// [`ScalarError::SingularDerivative`] at 2-torsion points ($2 y_0 \equiv 0 \pmod P$).
#[verified_engine::verified]
pub fn lift_weierstrass_y_qp<const P: u64, const N: usize>(
    a: i64,
    b: i64,
    x: i64,
    y0_mod_p: u64,
) -> Result<Qp<P, N>, ScalarError> {
    let x_qp = Qp::<P, N>::from_i64(x);
    let rhs_qp = x_qp * x_qp * x_qp + Qp::<P, N>::from_i64(a) * x_qp + Qp::<P, N>::from_i64(b);
    let y0_fp = Fp::<P>::from_u64(y0_mod_p);
    let x_fp = Fp::<P>::new(x);
    let rhs_fp = x_fp * x_fp * x_fp + Fp::<P>::new(a) * x_fp + Fp::<P>::new(b);
    if y0_fp * y0_fp != rhs_fp {
        return Err(ScalarError::RootNotFound);
    }
    let two_y0 = Fp::<P>::new(2) * y0_fp;
    if two_y0.is_zero() {
        return Err(ScalarError::SingularDerivative);
    }
    let mut y = Qp::<P, N>::from_u64(y0_fp.value());
    let two = Qp::<P, N>::from_u64(2);
    let steps = (usize::BITS - N.leading_zeros()) as usize + 2;
    for _ in 0..steps {
        let residual = y * y - rhs_qp;
        if residual.is_zero() || residual.valuation().is_some_and(|v| v >= (N as i32)) {
            break;
        }
        y -= residual.checked_div(two * y)?;
    }
    Ok(y)
}

/// Verifies whether a $p$-adic coefficient $c \in \mathbb{Q}_P$ satisfies a Breuer valuation lower
/// bound $v_P(c) \ge \text{min\_valuation}$ (Breuer 2016, DOI: `10.1007/s40993-016-0056-5`).
///
/// # Stability
/// Exact integer valuation comparison on normalized [`Qp`] digits ($v_P(0) = +\infty$ always satisfies any finite bound).
///
/// # Domain of Validity
/// Valid for all prime bases $P$, precisions $N \ge 1$, and non-negative integer bounds `min_valuation`.
#[must_use]
#[verified_engine::verified]
pub fn satisfies_padic_valuation_bound<const P: u64, const N: usize>(
    coeff: Qp<P, N>,
    min_valuation: u64,
) -> bool {
    match coeff.valuation() {
        None => true,
        Some(v) => v >= 0 && (v as u64) >= min_valuation,
    }
}
