#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:math_scalars]

use crate::error::ScalarError;
use crate::fp::Fp;
use crate::modular::{inv_mod_u64, mul_mod_u64, sub_mod_u64};

pub(super) fn eval_mod_p_and_deriv<const P: u64>(coeffs: &[i64], r0: u64) -> (Fp<P>, Fp<P>) {
    let x = Fp::<P>::new(r0);
    let mut fx = Fp::<P>::ZERO;
    let mut dfx = Fp::<P>::ZERO;
    for (deg, &c) in coeffs.iter().enumerate().rev() {
        let cf = Fp::<P>::from_i64(c);
        fx = fx * x + cf;
        if deg > 0 {
            dfx = dfx * x + Fp::<P>::new(deg as u64) * cf;
        }
    }
    (fx, dfx)
}

pub(super) fn zpn_from_u64<const P: u64, const N: usize>(mut mag: u64) -> [u64; N] {
    let mut d = [0u64; N];
    for slot in &mut d {
        *slot = mag % P;
        mag /= P;
    }
    d
}

pub(super) fn zpn_from_i64<const P: u64, const N: usize>(c: i64) -> [u64; N] {
    if c == 0 {
        return [0u64; N];
    }
    let d = zpn_from_u64::<P, N>(c.unsigned_abs());
    if c < 0 { zpn_neg::<P, N>(&d) } else { d }
}

pub(super) fn zpn_neg<const P: u64, const N: usize>(a: &[u64; N]) -> [u64; N] {
    let mut out = [0u64; N];
    let mut borrow = 0u64;
    for i in 0..N {
        if a[i] == 0 && borrow == 0 {
            out[i] = 0;
        } else {
            out[i] = P - borrow - a[i];
            borrow = 1;
        }
    }
    out
}

pub(super) fn zpn_add<const P: u64, const N: usize>(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
    let mut out = [0u64; N];
    let mut carry = 0u64;
    for i in 0..N {
        let sum1 = (a[i] as u128) + (b[i] as u128) + (carry as u128);
        if sum1 >= (P as u128) {
            out[i] = (sum1 - (P as u128)) as u64;
            carry = 1;
        } else {
            out[i] = sum1 as u64;
            carry = 0;
        }
    }
    out
}

pub(super) fn zpn_sub<const P: u64, const N: usize>(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
    zpn_add::<P, N>(a, &zpn_neg::<P, N>(b))
}

pub(super) fn zpn_add_scaled_shift<const P: u64, const N: usize>(
    acc: &mut [u64; N],
    u: &[u64; N],
    factor: u64,
    shift: usize,
) {
    if factor == 0 || shift >= N {
        return;
    }
    let p128 = P as u128;
    let mut carry = 0u128;
    for j in 0..(N - shift) {
        let total = (acc[shift + j] as u128) + (u[j] as u128) * (factor as u128) + carry;
        acc[shift + j] = (total % p128) as u64;
        carry = total / p128;
    }
}

pub(super) fn zpn_mul<const P: u64, const N: usize>(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
    let mut acc = [0u64; N];
    for (i, &ai) in a.iter().enumerate() {
        zpn_add_scaled_shift::<P, N>(&mut acc, b, ai, i);
    }
    acc
}

pub(super) fn zpn_invert_unit<const P: u64, const N: usize>(
    u: &[u64; N],
) -> Result<[u64; N], ScalarError> {
    let b0 = inv_mod_u64(u[0], P).ok_or(ScalarError::NonInvertible)?;
    let mut b = [0u64; N];
    let mut c = [0u64; N];
    b[0] = b0;
    zpn_add_scaled_shift::<P, N>(&mut c, u, b0, 0);
    for k in 1..N {
        if c[k] != 0 {
            let bk = mul_mod_u64(sub_mod_u64(0, c[k], P), b0, P);
            b[k] = bk;
            zpn_add_scaled_shift::<P, N>(&mut c, u, bk, k);
        }
    }
    Ok(b)
}

pub(super) fn eval_zpn_poly_and_deriv<const P: u64, const N: usize>(
    coeffs: &[i64],
    x: &[u64; N],
) -> ([u64; N], [u64; N]) {
    let mut fx = [0u64; N];
    let mut dfx = [0u64; N];
    for (deg, &c) in coeffs.iter().enumerate().rev() {
        let cz = zpn_from_i64::<P, N>(c);
        fx = zpn_add::<P, N>(&zpn_mul::<P, N>(&fx, x), &cz);
        if deg > 0 {
            let dz = zpn_from_u64::<P, N>(deg as u64);
            let dterm = zpn_mul::<P, N>(&dz, &cz);
            dfx = zpn_add::<P, N>(&zpn_mul::<P, N>(&dfx, x), &dterm);
        }
    }
    (fx, dfx)
}
