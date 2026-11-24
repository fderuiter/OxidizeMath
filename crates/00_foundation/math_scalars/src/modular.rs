#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:math_scalars]

use crate::error::ScalarError;

/// Overflow-safe modular addition $(a + b) \bmod m$ for any modulus $m \ge 1$ up to `u64::MAX`.
#[inline]
#[must_use]
pub const fn add_mod_u64(a: u64, b: u64, m: u64) -> u64 {
    let ar = a % m;
    let br = b % m;
    let gap = m - br;
    if ar >= gap { ar - gap } else { ar + br }
}

/// Overflow-safe modular subtraction $(a - b) \bmod m$ for any modulus $m \ge 1$ up to `u64::MAX`.
#[inline]
#[must_use]
pub const fn sub_mod_u64(a: u64, b: u64, m: u64) -> u64 {
    let ar = a % m;
    let br = b % m;
    if ar >= br { ar - br } else { m - (br - ar) }
}

/// Widened 128-bit modular multiplication $(a \cdot b) \bmod m$ that never overflows for any `u64` modulus.
#[inline]
#[must_use]
pub const fn mul_mod_u64(a: u64, b: u64, m: u64) -> u64 {
    let ar = (a % m) as u128;
    let br = (b % m) as u128;
    ((ar * br) % (m as u128)) as u64
}

/// Binary modular exponentiation $a^e \bmod m$ using widened 128-bit multiplication.
#[must_use]
pub const fn pow_mod_u64(base: u64, mut exp: u64, m: u64) -> u64 {
    if m <= 1 {
        return 0;
    }
    let mut acc = 1u64 % m;
    let mut cur = base % m;
    while exp > 0 {
        if (exp & 1) == 1 {
            acc = mul_mod_u64(acc, cur, m);
        }
        cur = mul_mod_u64(cur, cur, m);
        exp >>= 1;
    }
    acc
}

/// Unsigned modular extended Euclidean inverse returning $x \in [0, m)$ with $a x \equiv 1 \pmod m$,
/// or `None` when $\gcd(a, m) \ne 1$. Safe for all `u64` moduli including $m > 2^{63}$.
#[must_use]
pub const fn inv_mod_u64(a: u64, m: u64) -> Option<u64> {
    if m <= 1 {
        return None;
    }
    let mut old_r = m;
    let mut r = a % m;
    if r == 0 {
        return None;
    }
    let mut old_t = 0u64;
    let mut t = 1u64;
    while r != 0 {
        let q = old_r / r;
        let next_r = old_r % r;
        let qt = mul_mod_u64(q, t, m);
        let next_t = sub_mod_u64(old_t, qt, m);
        old_r = r;
        r = next_r;
        old_t = t;
        t = next_t;
    }
    if old_r == 1 { Some(old_t) } else { None }
}

/// Checked modular inverse returning [`ScalarError::NonInvertible`] when $\gcd(a, m) \ne 1$.
///
/// # Errors
/// Returns [`ScalarError::InvalidModulus`] when `m < 2` and [`ScalarError::NonInvertible`] when not coprime.
pub const fn checked_inv_mod_u64(a: u64, m: u64) -> Result<u64, ScalarError> {
    if m < 2 {
        return Err(ScalarError::InvalidModulus);
    }
    match inv_mod_u64(a, m) {
        Some(inv) => Ok(inv),
        None => Err(ScalarError::NonInvertible),
    }
}

/// Deterministic 64-bit Miller-Rabin primality test exact across the entire `0..=u64::MAX` domain.
#[must_use]
pub const fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    const BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    let mut idx = 0;
    while idx < BASES.len() {
        let p = BASES[idx];
        if n == p {
            return true;
        }
        if n.is_multiple_of(p) {
            return false;
        }
        idx += 1;
    }
    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    let mut b_idx = 0;
    while b_idx < BASES.len() {
        if !miller_rabin_witness(BASES[b_idx], d, s, n) {
            return false;
        }
        b_idx += 1;
    }
    true
}

const fn miller_rabin_witness(a: u64, d: u64, s: u32, n: u64) -> bool {
    let mut x = pow_mod_u64(a % n, d, n);
    if x == 1 || x == n - 1 {
        return true;
    }
    let mut r = 1;
    while r < s {
        x = mul_mod_u64(x, x, n);
        if x == n - 1 {
            return true;
        }
        r += 1;
    }
    false
}

/// Tonelli-Shanks square-root solver returning canonical sorted roots `[r0, r1]` ($r_0 \le r_1$)
/// of $y^2 \equiv n \pmod p$ for prime $p \ge 2$, or `None` when $n$ is a quadratic non-residue.
#[must_use]
pub const fn tonelli_shanks_u64(n: u64, p: u64) -> Option<[u64; 2]> {
    if p < 2 {
        return None;
    }
    let nr = n % p;
    if nr == 0 {
        return Some([0, 0]);
    }
    if p == 2 {
        return Some([1, 1]);
    }
    if pow_mod_u64(nr, (p - 1) / 2, p) != 1 {
        return None;
    }
    if (p & 3) == 3 {
        let r = pow_mod_u64(nr, (p >> 2) + 1, p);
        return Some(sort_pair(r, p - r));
    }
    tonelli_loop_u64(nr, p)
}

const fn tonelli_loop_u64(nr: u64, p: u64) -> Option<[u64; 2]> {
    let s = (p - 1).trailing_zeros();
    let q = (p - 1) >> s;
    let mut z = 2u64;
    while pow_mod_u64(z, (p - 1) / 2, p) != p - 1 {
        z += 1;
    }
    let mut m = s;
    let mut c = pow_mod_u64(z, q, p);
    let mut t = pow_mod_u64(nr, q, p);
    let mut r = pow_mod_u64(nr, (q >> 1) + 1, p);

    while t != 1 {
        let mut i = 1u32;
        let mut temp = mul_mod_u64(t, t, p);
        while temp != 1 {
            temp = mul_mod_u64(temp, temp, p);
            i += 1;
            if i >= m {
                return None;
            }
        }
        let b = pow_mod_u64(c, 1u64 << (m - i - 1), p);
        m = i;
        c = mul_mod_u64(b, b, p);
        t = mul_mod_u64(t, c, p);
        r = mul_mod_u64(r, b, p);
    }
    Some(sort_pair(r, p - r))
}

const fn sort_pair(a: u64, b: u64) -> [u64; 2] {
    if a <= b { [a, b] } else { [b, a] }
}

/// Reduces a signed `i64` into `[0, p)` for positive modulus `p > 0`.
#[inline]
#[must_use]
pub const fn mod_reduce_i64(val: i64, p: i64) -> i64 {
    val.rem_euclid(p)
}

/// Widened signed modular product $(a \cdot b) \bmod p$ in `[0, p)` for `p > 0`.
#[inline]
#[must_use]
pub const fn mod_mul_i64(a: i64, b: i64, p: i64) -> i64 {
    let pu = p as u64;
    let au = a.rem_euclid(p) as u64;
    let bu = b.rem_euclid(p) as u64;
    mul_mod_u64(au, bu, pu) as i64
}

/// Signed modular exponentiation $a^e \bmod p$ in `[0, p)` for `p > 1`.
#[inline]
#[must_use]
pub const fn mod_pow_i64(base: i64, exp: u64, p: i64) -> i64 {
    if p <= 1 {
        return 0;
    }
    pow_mod_u64(base.rem_euclid(p) as u64, exp, p as u64) as i64
}

/// Signed modular inverse $a^{-1} \bmod p$ in `(0, p)` for `p > 1`, or `None` if non-invertible.
#[inline]
#[must_use]
pub const fn mod_inv_i64(a: i64, p: i64) -> Option<i64> {
    if p <= 1 {
        return None;
    }
    match inv_mod_u64(a.rem_euclid(p) as u64, p as u64) {
        Some(inv) => Some(inv as i64),
        None => None,
    }
}

/// Signed Tonelli-Shanks wrapper returning sorted `[r0, r1]` in `[0, p)` for odd prime `p >= 3` (or `p == 2`).
#[inline]
#[must_use]
pub const fn tonelli_shanks_i64(n: i64, p: i64) -> Option<[i64; 2]> {
    if p < 2 {
        return None;
    }
    match tonelli_shanks_u64(n.rem_euclid(p) as u64, p as u64) {
        Some([r0, r1]) => Some([r0 as i64, r1 as i64]),
        None => None,
    }
}
