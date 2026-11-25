// [cite:modular_polynomials_review]
//! Zero-allocation stack `Qp<const P: u64, const N: usize>` $p$-adic root lifting and
//! arbitrary-precision Hensel square-root lifting for `AmbsDsp`.

use super::{AmbsDsp, AmbsError, modular_inverse};
use crate::math_types::{Integer, ops::Pow, ops::RemRounding};
pub use math_scalars::{Qp, ScalarError, eval_int_poly_qp, hensel_lift_root, tonelli_shanks_u64};

/// Lifts all simple modular square roots of $x^2 - a \equiv 0 \pmod P$ to $\mathbb{Q}_P$ up to
/// precision $O(P^N)$ on the stack with zero heap allocations (Hensel 1908, DOI: `10.1515/9783112342404`).
///
/// # Stability
/// Uses Tonelli-Shanks over $\mathbb{F}_P$ followed by quadratically convergent Newton-Hensel
/// steps $x_{m+1} = x_m - (x_m^2 - a)/(2x_m)$ in the ultrametric norm $|\cdot|_P$.
///
/// # Domain of Validity
/// Requires odd prime $P > 2$, $N \ge 1$, and $a \not\equiv 0 \pmod P$ for two distinct simple lifts
/// (returns `(roots, 0)` for quadratic non-residues or ramified $a \equiv 0 \pmod P$).
#[must_use]
#[verified_engine::verified]
pub fn lift_quadratic_residue_qp<const P: u64, const N: usize>(a: i64) -> ([Qp<P, N>; 2], usize) {
    let () = Qp::<P, N>::VALIDATE_PADIC;
    if P == 2 {
        return ([Qp::zero(), Qp::zero()], 0);
    }
    let a_mod = math_scalars::mod_reduce_i64(a, P);
    if a_mod == 0 {
        return ([Qp::zero(), Qp::zero()], 0);
    }
    let (mod_roots, count) = tonelli_shanks_u64(a_mod, P);
    if count == 0 {
        return ([Qp::zero(), Qp::zero()], 0);
    }
    let poly = [-a, 0, 1];
    let mut out = [Qp::<P, N>::zero(); 2];
    let mut lifted = 0;
    for &r0 in mod_roots.iter().take(count) {
        if let Ok(qp_root) = hensel_lift_root::<P, N, 3>(&poly, r0) {
            out[lifted] = qp_root;
            lifted += 1;
        }
    }
    (out, lifted)
}

impl AmbsDsp {
    #[verified_engine::verified]
    pub(super) fn hensels_lifting(
        a: &Integer,
        p: &Integer,
        k: u32,
    ) -> Result<Vec<Integer>, AmbsError> {
        let roots_mod_p = Self::tonelli_shanks(a, p)?;
        let mut final_roots = Vec::new();
        let p_k = p.clone().pow(k);

        for r1 in roots_mod_p {
            let mut x_n = r1.clone();
            let mut p_n = p.clone();
            for _ in 1..k {
                let f_x_n = (&x_n * &x_n) - a.clone();
                let f_prime_x_n = &x_n * Integer::from(2);
                if let Some(f_prime_inv) = modular_inverse(&f_prime_x_n, p) {
                    let term = (f_x_n / &p_n) * f_prime_inv;
                    let term_mod = term.rem_euc(p);
                    x_n -= term_mod * &p_n;
                    p_n *= p;
                    x_n = x_n.rem_euc(&p_n);
                } else {
                    break;
                }
            }
            if Integer::from(&x_n * &x_n).rem_euc(&p_k) == a.clone().rem_euc(&p_k) {
                final_roots.push(x_n);
            }
        }

        final_roots.sort();
        final_roots.dedup();
        Ok(final_roots)
    }
}
