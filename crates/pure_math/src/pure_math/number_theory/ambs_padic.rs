// [cite:math_scalars]
//! Stack-allocated $p$-adic ([`Qp<P, N>`]) square-root and AMBS congruence Newton-Hensel lifters.

use math_scalars::modular::tonelli_shanks_i64;
pub use math_scalars::{Qp, ScalarError};

/// Lifts modular square roots of `residue` ($X^2 - \text{residue} \equiv 0 \pmod{P^N}$) into
/// fixed-precision stack $p$-adic numbers [`Qp<P, N>`] without heap allocation.
///
/// # Errors
/// - Returns [`ScalarError::ArithmeticOverflow`] when `residue == i64::MIN` or `P > i64::MAX as u64`.
/// - Returns [`ScalarError::SingularDerivative`] when $2 r_0 \equiv 0 \pmod P$ (ramified / $P = 2$ or $r_0 = 0$).
#[verified_engine::verified]
pub fn hensel_lift_sqrt_padic<const P: u64, const N: usize>(
    residue: i64,
) -> Result<([Qp<P, N>; 2], usize), ScalarError> {
    let p_i64 = i64::try_from(P).map_err(|_| ScalarError::ArithmeticOverflow)?;
    let neg_res = residue
        .checked_neg()
        .ok_or(ScalarError::ArithmeticOverflow)?;
    let Some([r0, r1]) = tonelli_shanks_i64(residue, p_i64) else {
        return Ok(([Qp::<P, N>::ZERO, Qp::<P, N>::ZERO], 0));
    };
    let lift0 = Qp::<P, N>::hensel_lift_root(&[neg_res, 0, 1], r0 as u64)?;
    if r0 == r1 {
        return Ok(([lift0, Qp::<P, N>::ZERO], 1));
    }
    let lift1 = Qp::<P, N>::hensel_lift_root(&[neg_res, 0, 1], r1 as u64)?;
    Ok(([lift0, lift1], 2))
}

/// Solves the AMBS quasiperfect modularity congruence $2 n_L z^2 + 1 \equiv 0 \pmod{P^N}$
/// directly in $\mathbb{Q}_P$ via stack-allocated Newton-Hensel lifting of $f(z) = 1 + 2 n_L z^2$.
///
/// # Errors
/// - Returns [`ScalarError::ArithmeticOverflow`] if $2 n_L$ or $P$ overflows `i64`.
/// - Returns [`ScalarError::SingularDerivative`] if $P = 2$ or $P \mid n_L$.
#[verified_engine::verified]
pub fn ambs_prefix_padic_roots<const P: u64, const N: usize>(
    n_l: u64,
) -> Result<([Qp<P, N>; 2], usize), ScalarError> {
    let p_i64 = i64::try_from(P).map_err(|_| ScalarError::ArithmeticOverflow)?;
    let two_nl = i64::try_from(n_l)
        .ok()
        .and_then(|v| v.checked_mul(2))
        .ok_or(ScalarError::ArithmeticOverflow)?;
    let Some(inv_neg_two_nl) = math_scalars::modular::mod_inv_i64(-two_nl, p_i64) else {
        return Err(ScalarError::SingularDerivative);
    };
    let Some([r0, r1]) = tonelli_shanks_i64(inv_neg_two_nl, p_i64) else {
        return Ok(([Qp::<P, N>::ZERO, Qp::<P, N>::ZERO], 0));
    };
    let coeffs = [1_i64, 0, two_nl];
    let z0 = Qp::<P, N>::hensel_lift_root(&coeffs, r0 as u64)?;
    if r0 == r1 {
        return Ok(([z0, Qp::<P, N>::ZERO], 1));
    }
    let z1 = Qp::<P, N>::hensel_lift_root(&coeffs, r1 as u64)?;
    Ok(([z0, z1], 2))
}
