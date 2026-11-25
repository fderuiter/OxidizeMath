#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:modular_polynomials_review]
//! Carry-safe 64-bit modular arithmetic, deterministic 64-bit Miller-Rabin primality
//! validation, Tonelli-Shanks square roots, and 128-bit Montgomery REDC kernels.

/// Largest 64-bit prime $2^{64} - 59 = 18{,}446{,}744{,}073{,}709{,}551{,}557$.
pub const MAX_PRIME_U64: u64 = 18_446_744_073_709_551_557;

/// Deterministic Miller-Rabin witness set exact for every 64-bit unsigned integer (`n < 2^64`).
const MR_BASES_64: [u64; 7] = [2, 325, 9_375, 28_178, 450_775, 9_780_504, 1_795_265_022];

/// Reduces signed `a` into canonical `[0, m - 1]` (`0` if `m == 0`).
#[inline]
#[must_use]
pub const fn mod_reduce_i64(a: i64, m: u64) -> u64 {
    if m == 0 {
        return 0;
    }
    let m128 = m as i128;
    let r = (a as i128) % m128;
    if r < 0 { (r + m128) as u64 } else { r as u64 }
}

/// Reduces signed 128-bit `a` into canonical `[0, m - 1]` (`0` if `m == 0`).
#[inline]
#[must_use]
pub const fn mod_reduce_i128(a: i128, m: u64) -> u64 {
    if m == 0 {
        return 0;
    }
    let mag = a.unsigned_abs();
    let rem = (mag % (m as u128)) as u64;
    if a < 0 && rem != 0 { m - rem } else { rem }
}

/// Carry-safe `(a + b) mod m` for arbitrary `u64` inputs without overflow above $2^{63}$.
#[inline]
#[must_use]
pub const fn mod_add_u64(a: u64, b: u64, m: u64) -> u64 {
    if m == 0 {
        return 0;
    }
    let ar = a % m;
    let br = b % m;
    let diff = m - br;
    if ar >= diff { ar - diff } else { ar + br }
}

/// Carry-safe `(a - b) mod m` for arbitrary `u64` inputs without underflow.
#[inline]
#[must_use]
pub const fn mod_sub_u64(a: u64, b: u64, m: u64) -> u64 {
    if m == 0 {
        return 0;
    }
    let ar = a % m;
    let br = b % m;
    if ar >= br { ar - br } else { m - (br - ar) }
}

/// Carry-safe `(-a) mod m` for arbitrary `u64` inputs.
#[inline]
#[must_use]
pub const fn mod_neg_u64(a: u64, m: u64) -> u64 {
    if m == 0 {
        return 0;
    }
    let r = a % m;
    if r == 0 { 0 } else { m - r }
}

/// Widened `(a * b) mod m` via `u128` without overflow for any `u64` modulus `m >= 1`.
#[inline]
#[must_use]
pub const fn mod_mul_u64(a: u64, b: u64, m: u64) -> u64 {
    if m == 0 {
        return 0;
    }
    (((a % m) as u128 * (b % m) as u128) % (m as u128)) as u64
}

/// Binary modular exponentiation $b^e \bmod m$ (`0` if `m <= 1`).
#[must_use]
pub const fn mod_pow_u64(base: u64, mut exp: u64, m: u64) -> u64 {
    if m <= 1 {
        return 0;
    }
    let mut result = 1_u64;
    let mut cur = base % m;
    while exp > 0 {
        if (exp & 1) == 1 {
            result = mod_mul_u64(result, cur, m);
        }
        exp >>= 1;
        if exp > 0 {
            cur = mod_mul_u64(cur, cur, m);
        }
    }
    result
}

/// Iterative Extended-Euclidean modular inverse $a^{-1} \bmod m$ (`None` iff $\gcd(a, m) \ne 1$ or `m <= 1`).
///
/// Uses `i128` Bézout accumulators bounded in magnitude by `m < 2^64`, preventing overflow
/// for all 64-bit moduli including primes above $2^{63}$.
#[must_use]
pub const fn mod_inv_u64(a: u64, m: u64) -> Option<u64> {
    if m <= 1 {
        return None;
    }
    let m128 = m as i128;
    let mut t: i128 = 0;
    let mut new_t: i128 = 1;
    let mut r: i128 = m128;
    let mut new_r: i128 = (a % m) as i128;
    while new_r != 0 {
        let q = r / new_r;
        let next_t = t - q * new_t;
        t = new_t;
        new_t = next_t;
        let next_r = r - q * new_r;
        r = new_r;
        new_r = next_r;
    }
    if r != 1 {
        return None;
    }
    if t < 0 {
        t += m128;
    }
    Some(t as u64)
}

/// Exact `const fn` primality test for all 64-bit unsigned integers using 7-base Miller-Rabin.
#[must_use]
pub const fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 || n == 5 || n == 7 {
        return true;
    }
    if (n & 1) == 0 || n.is_multiple_of(3) || n.is_multiple_of(5) || n.is_multiple_of(7) {
        return false;
    }
    let d_raw = n - 1;
    let s = d_raw.trailing_zeros();
    let d = d_raw >> s;
    let mut idx = 0;
    while idx < MR_BASES_64.len() {
        let a = MR_BASES_64[idx] % n;
        if a != 0 && !miller_rabin_witness_pass(a, d, s, n) {
            return false;
        }
        idx += 1;
    }
    true
}

const fn miller_rabin_witness_pass(a: u64, d: u64, s: u32, n: u64) -> bool {
    let mut x = mod_pow_u64(a, d, n);
    if x == 1 || x == n - 1 {
        return true;
    }
    let mut r = 1;
    while r < s {
        x = mod_mul_u64(x, x, n);
        if x == n - 1 {
            return true;
        }
        if x == 1 {
            return false;
        }
        r += 1;
    }
    false
}

/// Solves $y^2 \equiv \text{rhs} \pmod p$ for prime `p`, returning sorted roots `([y0, y1], count)`.
#[must_use]
pub const fn tonelli_shanks_u64(rhs: u64, p: u64) -> ([u64; 2], usize) {
    if p < 2 {
        return ([0, 0], 0);
    }
    let n = rhs % p;
    if p == 2 {
        return ([n, 0], 1);
    }
    if n == 0 {
        return ([0, 0], 1);
    }
    if mod_pow_u64(n, (p - 1) / 2, p) != 1 {
        return ([0, 0], 0);
    }
    if (p & 3) == 3 {
        let r = mod_pow_u64(n, (p + 1) / 4, p);
        return sorted_distinct_roots(r, p);
    }
    tonelli_shanks_general(n, p)
}

const fn sorted_distinct_roots(r: u64, p: u64) -> ([u64; 2], usize) {
    let neg = mod_neg_u64(r, p);
    if r == neg {
        ([r, 0], 1)
    } else if r < neg {
        ([r, neg], 2)
    } else {
        ([neg, r], 2)
    }
}

const fn tonelli_shanks_general(n: u64, p: u64) -> ([u64; 2], usize) {
    let pm1 = p - 1;
    let s = pm1.trailing_zeros();
    let q = pm1 >> s;
    let mut z = 2_u64;
    while z < p && mod_pow_u64(z, pm1 / 2, p) != pm1 {
        z += 1;
    }
    let mut m = s;
    let mut c = mod_pow_u64(z, q, p);
    let mut t = mod_pow_u64(n, q, p);
    let mut r = mod_pow_u64(n, q.div_ceil(2), p);
    while t != 1 {
        let mut i = 1_u32;
        let mut temp = mod_mul_u64(t, t, p);
        while temp != 1 && i < m {
            temp = mod_mul_u64(temp, temp, p);
            i += 1;
        }
        if i >= m {
            return ([0, 0], 0);
        }
        let b = mod_pow_u64(c, 1_u64 << (m - i - 1), p);
        m = i;
        c = mod_mul_u64(b, b, p);
        t = mod_mul_u64(t, c, p);
        r = mod_mul_u64(r, b, p);
    }
    sorted_distinct_roots(r, p)
}

/// Precomputed compile-time Montgomery constants for an odd 64-bit prime $P \ge 3$ ($R = 2^{64}$).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct MontgomeryParams {
    pub(crate) p: u64,
    pub(crate) n_prime: u64,
    pub(crate) r_mod_p: u64,
    pub(crate) r2_mod_p: u64,
}

impl MontgomeryParams {
    /// Computes $N' = -P^{-1} \bmod 2^{64}$, $R \bmod P$, and $R^2 \bmod P$ for odd $P \ge 3$.
    #[must_use]
    pub(crate) const fn new_odd(p: u64) -> Self {
        let mut inv = 1_u64;
        let mut step = 0;
        while step < 6 {
            inv = inv.wrapping_mul(2_u64.wrapping_sub(p.wrapping_mul(inv)));
            step += 1;
        }
        let n_prime = inv.wrapping_neg();
        let p128 = p as u128;
        // 2^64 mod p = ((2^64 - p) % p) without needing 1_u128 << 64 if desired, though 1_u128 << 64 fits in u128.
        let r_mod_p = ((1_u128 << 64) % p128) as u64;
        let r2_mod_p = (((r_mod_p as u128) * (r_mod_p as u128)) % p128) as u64;
        Self {
            p,
            n_prime,
            r_mod_p,
            r2_mod_p,
        }
    }

    /// Carry-safe 128-bit Montgomery reduction $T \cdot R^{-1} \bmod P$ for $0 \le T < P \cdot 2^{64}$.
    ///
    /// Splits $T = T_{\mathrm{hi}} 2^{64} + T_{\mathrm{lo}}$ so $T + m P$ never overflows `u128`
    /// even when $P = 18{,}446{,}744{,}073{,}709{,}551{,}557 > 2^{63}$.
    #[inline]
    #[must_use]
    pub(crate) const fn redc(self, t: u128) -> u64 {
        let t_lo = t as u64;
        let t_hi = (t >> 64) as u64;
        let m = t_lo.wrapping_mul(self.n_prime);
        let mp = (m as u128) * (self.p as u128);
        let mp_lo = mp as u64;
        let mp_hi = (mp >> 64) as u64;
        let (_, carry_lo) = t_lo.overflowing_add(mp_lo);
        let u = (t_hi as u128) + (mp_hi as u128) + (carry_lo as u128);
        let p128 = self.p as u128;
        if u >= p128 {
            (u - p128) as u64
        } else {
            u as u64
        }
    }

    /// Encodes canonical $x \in [0, P - 1]$ into Montgomery residue $x R \bmod P$.
    #[inline]
    #[must_use]
    pub(crate) const fn to_mont(self, x: u64) -> u64 {
        self.redc((x as u128) * (self.r2_mod_p as u128))
    }

    /// Decodes Montgomery residue $a R \bmod P$ back into canonical $[0, P - 1]$.
    #[inline]
    #[must_use]
    pub(crate) const fn decode_mont(self, mont: u64) -> u64 {
        self.redc(mont as u128)
    }

    /// Multiplies two Montgomery residues $(a R)(b R) R^{-1} \equiv (a b) R \pmod P$.
    #[inline]
    #[must_use]
    pub(crate) const fn mul_mont(self, a_mont: u64, b_mont: u64) -> u64 {
        self.redc((a_mont as u128) * (b_mont as u128))
    }
}
