#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:modular_polynomials_review]
//! Fixed-precision stack $p$-adic field elements $\mathbb{Q}_P$ ([`Qp`]) with ultrametric
//! valuations, carry-safe base-$P$ arithmetic up to $2^{64} - 59$, unit-series inversion,
//! and zero-allocation Newton-Hensel simple root lifting ([`hensel_lift_root`]).

use crate::error::ScalarError;
use crate::fp::Fp;
use crate::modular::{is_prime_u64, mod_inv_u64, mod_mul_u64, mod_neg_u64};
use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// Fixed-precision $p$-adic number $x = P^v \sum_{k=0}^{N-1} d_k P^k + O(P^{v+N}) \in \mathbb{Q}_P$.
///
/// Canonical non-zero elements satisfy `digits[0] != 0` and `digits[k] < P`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Qp<const P: u64, const N: usize> {
    digits: [u64; N],
    valuation: i32,
    is_zero: bool,
}

impl<const P: u64, const N: usize> Default for Qp<P, N> {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

impl<const P: u64, const N: usize> Qp<P, N> {
    /// Compile-time check enforcing prime `P` and digit precision `N >= 1`.
    pub const VALIDATE_PADIC: () = {
        assert!(is_prime_u64(P), "Qp<P, N> base P must be prime");
        assert!(N >= 1, "Qp<P, N> digit precision N must be >= 1");
    };

    /// Exact $p$-adic additive identity $0$ ($v_P(0) = +\infty$).
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        let () = Self::VALIDATE_PADIC;
        Self {
            digits: [0; N],
            valuation: 0,
            is_zero: true,
        }
    }

    /// Multiplicative identity $1 = P^0 \cdot 1$.
    #[must_use]
    pub const fn one() -> Self {
        let () = Self::VALIDATE_PADIC;
        let mut digits = [0_u64; N];
        digits[0] = 1;
        Self {
            digits,
            valuation: 0,
            is_zero: false,
        }
    }

    /// Uniformizer $\pi = P = P^1 \cdot 1$ with valuation $v_P(P) = 1$.
    #[must_use]
    pub const fn uniformizer() -> Self {
        let () = Self::VALIDATE_PADIC;
        let mut digits = [0_u64; N];
        digits[0] = 1;
        Self {
            digits,
            valuation: 1,
            is_zero: false,
        }
    }

    /// Constructs a canonical `Qp` from unnormalized base-$P$ digits at valuation `base_val`.
    #[must_use]
    pub fn from_raw_digits(raw: [u64; N], base_val: i32) -> Self {
        let () = Self::VALIDATE_PADIC;
        let mut reduced = [0_u64; N];
        let mut carry = 0_u128;
        for (slot, &d) in reduced.iter_mut().zip(&raw) {
            let sum = (d as u128) + carry;
            *slot = (sum % (P as u128)) as u64;
            carry = sum / (P as u128);
        }
        Self::normalize_digits(reduced, base_val)
    }

    fn normalize_digits(raw: [u64; N], base_val: i32) -> Self {
        let Some(shift) = raw.iter().position(|&d| d != 0) else {
            return Self::zero();
        };
        let mut digits = [0_u64; N];
        digits[..(N - shift)].copy_from_slice(&raw[shift..]);
        Self {
            digits,
            valuation: base_val.saturating_add(shift as i32),
            is_zero: false,
        }
    }

    /// Embeds unsigned integer `val` into $\mathbb{Q}_P$.
    #[must_use]
    pub fn from_u64(mut val: u64) -> Self {
        let () = Self::VALIDATE_PADIC;
        if val == 0 {
            return Self::zero();
        }
        let mut v = 0_i32;
        while val.is_multiple_of(P) {
            val /= P;
            v += 1;
        }
        let mut digits = [0_u64; N];
        for slot in &mut digits {
            if val == 0 {
                break;
            }
            *slot = val % P;
            val /= P;
        }
        Self {
            digits,
            valuation: v,
            is_zero: false,
        }
    }

    /// Embeds signed integer `val` into $\mathbb{Q}_P$.
    #[must_use]
    pub fn from_i64(val: i64) -> Self {
        let () = Self::VALIDATE_PADIC;
        if val == 0 {
            return Self::zero();
        }
        let mag = Self::from_u64(val.unsigned_abs());
        if val < 0 { -mag } else { mag }
    }

    /// Embeds exact rational `numer / denom` into $\mathbb{Q}_P$.
    pub fn from_rational(numer: i64, denom: i64) -> Result<Self, ScalarError> {
        if denom == 0 {
            return Err(ScalarError::ZeroDenominator);
        }
        Self::from_i64(numer).checked_div(Self::from_i64(denom))
    }

    /// Returns `true` iff `self == 0` up to active precision.
    #[inline]
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.is_zero
    }

    /// $p$-adic valuation $v_P(x) \in \mathbb{Z}$, or `None` ($+\infty$) for zero.
    #[inline]
    #[must_use]
    pub const fn valuation(&self) -> Option<i32> {
        if self.is_zero {
            None
        } else {
            Some(self.valuation)
        }
    }

    /// $p$-adic absolute value $|x|_P = P^{-v_P(x)}$ (`0.0` for zero).
    #[must_use]
    pub fn padic_abs_f64(&self) -> f64 {
        if self.is_zero {
            0.0
        } else {
            libm::pow(P as f64, -f64::from(self.valuation))
        }
    }

    /// Unit-part base-$P$ digits `[d_0, d_1, ..., d_{N-1}]`.
    #[inline]
    #[must_use]
    pub const fn digits(&self) -> &[u64; N] {
        &self.digits
    }

    /// Reconstructs the integer residue $\sum_{k=0}^{m-1} d_k P^{v + k} \bmod P^m$ for $v \ge 0$ when $P^m$ fits in `u128`.
    #[must_use]
    pub fn to_u128_mod_pn(&self) -> Option<u128> {
        if self.is_zero {
            return Some(0);
        }
        if self.valuation < 0 {
            return None;
        }
        let v = self.valuation as usize;
        if v >= N {
            return Some(0);
        }
        let p128 = P as u128;
        let mut p_pow = 1_u128;
        let mut acc = 0_u128;
        for k in 0..N {
            if k >= v {
                let d = self.digits[k - v] as u128;
                acc = acc.checked_add(d.checked_mul(p_pow)?)?;
            }
            if k + 1 < N {
                p_pow = p_pow.checked_mul(p128)?;
            }
        }
        Some(acc)
    }

    /// Verifies the non-Archimedean ultrametric inequality $|a + b|_P \le \max(|a|_P, |b|_P)$.
    #[must_use]
    pub fn satisfies_ultrametric(a: Self, b: Self) -> bool {
        let sum = a + b;
        match (sum.valuation(), a.valuation(), b.valuation()) {
            (None, _, _) => true,
            (Some(_), None, None) => false,
            (Some(vs), Some(va), None) => vs >= va,
            (Some(vs), None, Some(vb)) => vs >= vb,
            (Some(vs), Some(va), Some(vb)) => vs >= va.min(vb),
        }
    }

    /// Fallible multiplicative inverse $x^{-1} = P^{-v} u^{-1} \in \mathbb{Q}_P$.
    #[must_use]
    pub fn checked_inv(self) -> Option<Self> {
        if self.is_zero {
            return None;
        }
        let w0 = mod_inv_u64(self.digits[0], P)?;
        let mut w = [0_u64; N];
        w[0] = w0;
        let (q0, _) = div_rem_p((self.digits[0] as u128) * (w0 as u128), P);
        let mut carry_p = q0 as u128;

        for k in 1..N {
            let mut rem = (carry_p % (P as u128)) as u64;
            carry_p /= P as u128;
            for i in 1..=k {
                accum_prod_base_p(self.digits[i], w[k - i], P, &mut rem, &mut carry_p);
            }
            let target = mod_neg_u64(rem, P);
            w[k] = mod_mul_u64(target, w0, P);
            accum_prod_base_p(self.digits[0], w[k], P, &mut rem, &mut carry_p);
        }
        Some(Self {
            digits: w,
            valuation: -self.valuation,
            is_zero: false,
        })
    }

    /// Fallible multiplicative inverse returning [`ScalarError::DivisionByZero`] when zero.
    pub fn try_inverse(self) -> Result<Self, ScalarError> {
        self.checked_inv().ok_or(ScalarError::DivisionByZero)
    }

    /// Fallible division `self * rhs^-1` in $\mathbb{Q}_P$.
    pub fn checked_div(self, rhs: Self) -> Result<Self, ScalarError> {
        Ok(self * rhs.try_inverse()?)
    }
}

#[inline]
const fn div_rem_p(prod: u128, p: u64) -> (u64, u64) {
    let p128 = p as u128;
    ((prod / p128) as u64, (prod % p128) as u64)
}

#[inline]
fn accum_prod_base_p(a: u64, b: u64, p: u64, rem: &mut u64, carry_p: &mut u128) {
    if a == 0 || b == 0 {
        return;
    }
    let (q, r) = div_rem_p((a as u128) * (b as u128), p);
    let diff = p - r;
    if *rem >= diff {
        *rem -= diff;
        *carry_p += 1;
    } else {
        *rem += r;
    }
    *carry_p += q as u128;
}

impl<const P: u64, const N: usize> fmt::Display for Qp<P, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero {
            return write!(f, "0 + O({P}^{N})");
        }
        for k in 0..N {
            if k > 0 {
                f.write_str(" + ")?;
            }
            let exp = self.valuation + (k as i32);
            write!(f, "{}*{P}^{exp}", self.digits[k])?;
        }
        write!(f, " + O({P}^{})", self.valuation + (N as i32))
    }
}

impl<const P: u64, const N: usize> Neg for Qp<P, N> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        if self.is_zero {
            return Self::zero();
        }
        let mut digits = [0_u64; N];
        digits[0] = P - self.digits[0];
        for (slot, &d) in digits.iter_mut().zip(&self.digits).skip(1) {
            *slot = (P - 1) - d;
        }
        Self {
            digits,
            valuation: self.valuation,
            is_zero: false,
        }
    }
}

impl<const P: u64, const N: usize> Add for Qp<P, N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        if self.is_zero {
            return rhs;
        }
        if rhs.is_zero {
            return self;
        }
        let (lo, hi) = if self.valuation <= rhs.valuation {
            (self, rhs)
        } else {
            (rhs, self)
        };
        let delta = (hi.valuation - lo.valuation) as usize;
        if delta >= N {
            return lo;
        }
        let mut sum = [0_u64; N];
        let mut carry = 0_u64;
        for (k, slot) in sum.iter_mut().enumerate() {
            let b_digit = if k >= delta { hi.digits[k - delta] } else { 0 };
            let mut acc = lo.digits[k];
            let mut next_carry = 0_u64;
            let diff_b = P - b_digit;
            if acc >= diff_b {
                acc -= diff_b;
                next_carry += 1;
            } else {
                acc += b_digit;
            }
            if carry > 0 {
                let diff_c = P - carry;
                if acc >= diff_c {
                    acc -= diff_c;
                    next_carry += 1;
                } else {
                    acc += carry;
                }
            }
            *slot = acc;
            carry = next_carry;
        }
        Self::normalize_digits(sum, lo.valuation)
    }
}

impl<const P: u64, const N: usize> AddAssign for Qp<P, N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u64, const N: usize> Sub for Qp<P, N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<const P: u64, const N: usize> SubAssign for Qp<P, N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u64, const N: usize> Mul for Qp<P, N> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        if self.is_zero || rhs.is_zero {
            return Self::zero();
        }
        let mut prod = [0_u64; N];
        let mut carry_p = 0_u128;
        for (k, slot) in prod.iter_mut().enumerate() {
            let mut rem = (carry_p % (P as u128)) as u64;
            carry_p /= P as u128;
            for i in 0..=k {
                accum_prod_base_p(self.digits[i], rhs.digits[k - i], P, &mut rem, &mut carry_p);
            }
            *slot = rem;
        }
        Self::normalize_digits(prod, self.valuation.saturating_add(rhs.valuation))
    }
}

impl<const P: u64, const N: usize> MulAssign for Qp<P, N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u64, const N: usize> Div for Qp<P, N> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self.checked_div(rhs)
            .expect("Qp division by zero; use checked_div or try_inverse")
    }
}

/// Evaluates an integer-coefficient polynomial $f(x) = \sum_{k=0}^{d} c_k x^k$ at `x` in $\mathbb{Q}_P$.
#[must_use]
pub fn eval_int_poly_qp<const P: u64, const N: usize>(coeffs: &[i64], x: Qp<P, N>) -> Qp<P, N> {
    let mut acc = Qp::<P, N>::zero();
    for &c in coeffs.iter().rev() {
        acc = acc * x + Qp::<P, N>::from_i64(c);
    }
    acc
}

/// Zero-allocation Newton-Hensel simple root lifting from $f(r_0) \equiv 0 \pmod P$ to $\mathbb{Q}_P$
/// up to precision $O(P^N)$ (`DEG_CAP` retained as maximum degree check for bounded stack evaluation).
///
/// Requires $f(r_0) \equiv 0 \pmod P$ ([`ScalarError::RootNotFound`]) and $f'(r_0) \not\equiv 0 \pmod P$
/// ([`ScalarError::SingularDerivative`]), converging quadratically in $\lceil \log_2 N \rceil + 1$ steps.
pub fn hensel_lift_root<const P: u64, const N: usize, const DEG_CAP: usize>(
    poly_coeffs: &[i64],
    mod_p_root: u64,
) -> Result<Qp<P, N>, ScalarError> {
    let () = Qp::<P, N>::VALIDATE_PADIC;
    if poly_coeffs.len() > DEG_CAP {
        return Err(ScalarError::DegreeOverflow {
            capacity: DEG_CAP,
            required: poly_coeffs.len(),
        });
    }
    let r0_fp = Fp::<P>::from_u64(mod_p_root);
    let mut f_fp = Fp::<P>::ZERO;
    let mut df_fp = Fp::<P>::ZERO;
    for (idx, &c) in poly_coeffs.iter().enumerate().rev() {
        f_fp = f_fp * r0_fp + Fp::<P>::new(c);
        if idx > 0 {
            df_fp = df_fp * r0_fp + Fp::<P>::new(c) * Fp::<P>::from_u64(idx as u64);
        }
    }
    if !f_fp.is_zero() {
        return Err(ScalarError::RootNotFound);
    }
    if df_fp.is_zero() {
        return Err(ScalarError::SingularDerivative);
    }

    let mut x = Qp::<P, N>::from_u64(r0_fp.value());
    let max_steps = (usize::BITS - N.leading_zeros()) as usize + 2;
    for _ in 0..max_steps {
        let fx = eval_int_poly_qp(poly_coeffs, x);
        if fx.is_zero() || fx.valuation().is_some_and(|v| v >= (N as i32)) {
            break;
        }
        let dfx = eval_int_deriv_qp(poly_coeffs, x);
        let step = fx.checked_div(dfx)?;
        x -= step;
    }
    Ok(x)
}

fn eval_int_deriv_qp<const P: u64, const N: usize>(coeffs: &[i64], x: Qp<P, N>) -> Qp<P, N> {
    if coeffs.len() <= 1 {
        return Qp::<P, N>::zero();
    }
    let mut acc = Qp::<P, N>::zero();
    for (idx, &c) in coeffs.iter().enumerate().skip(1).rev() {
        let term = Qp::<P, N>::from_i64(c) * Qp::<P, N>::from_u64(idx as u64);
        acc = acc * x + term;
    }
    acc
}
