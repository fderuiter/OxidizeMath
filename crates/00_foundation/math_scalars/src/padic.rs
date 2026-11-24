#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:math_scalars]

#[path = "padic_zpn.rs"]
mod padic_zpn;

use crate::error::ScalarError;
use crate::fp::Fp;
use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use padic_zpn::{
    eval_mod_p_and_deriv, eval_zpn_poly_and_deriv, zpn_add, zpn_invert_unit, zpn_mul, zpn_neg,
    zpn_sub,
};

/// Fixed-precision stack-allocated $p$-adic number in $\mathbb{Q}_P$ retaining `N` base-$P$ digits:
/// $$x = P^v \sum_{i=0}^{N-1} d_i P^i + O(P^{v + N}), \qquad d_0 \in \{1, \dots, P-1\}.$$
///
/// Composite moduli `P` and `N == 0` are rejected at compile time. All ring operations,
/// unit inversions, and Newton-Hensel polynomial root lifts execute on the stack with zero
/// dynamic heap allocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Qp<const P: u64, const N: usize> {
    is_zero: bool,
    val: i32,
    digits: [u64; N],
}

impl<const P: u64, const N: usize> Default for Qp<P, N> {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const P: u64, const N: usize> Qp<P, N> {
    const VALID_PADIC: () = {
        let () = Fp::<P>::VALID_PRIME;
        assert!(N >= 1, "Qp digit precision N must be >= 1");
    };

    /// Exact zero element $0 \in \mathbb{Q}_P$ ($v_P(0) = +\infty$, represented as `None`).
    pub const ZERO: Self = {
        let () = Self::VALID_PADIC;
        Self {
            is_zero: true,
            val: 0,
            digits: [0u64; N],
        }
    };

    /// Multiplicative identity $1 = P^0 (1 + 0 P + \dots) \in \mathbb{Q}_P$.
    pub const ONE: Self = {
        let () = Self::VALID_PADIC;
        let mut d = [0u64; N];
        d[0] = 1;
        Self {
            is_zero: false,
            val: 0,
            digits: d,
        }
    };

    /// Uniformizer $\pi = P = P^1 (1 + 0 P + \dots)$ with $v_P(P) = 1$.
    pub const UNIFORMIZER: Self = {
        let () = Self::VALID_PADIC;
        let mut d = [0u64; N];
        d[0] = 1;
        Self {
            is_zero: false,
            val: 1,
            digits: d,
        }
    };

    /// Constructs a canonical `Qp<P, N>` from valuation `val` and raw base-$P$ digits (`digits[i] < P`).
    ///
    /// # Errors
    /// Returns [`ScalarError::InvalidDigit`] if any entry in `digits` is $\ge P$.
    pub fn from_valuation_and_digits(val: i32, digits: [u64; N]) -> Result<Self, ScalarError> {
        let () = Self::VALID_PADIC;
        for &d in &digits {
            if d >= P {
                return Err(ScalarError::InvalidDigit);
            }
        }
        Ok(Self::normalize_raw(val, digits))
    }

    /// Embeds an unsigned integer $n \in \mathbb{N}$ into $\mathbb{Q}_P$.
    #[must_use]
    pub fn from_u64(mut n: u64) -> Self {
        let () = Self::VALID_PADIC;
        if n == 0 {
            return Self::ZERO;
        }
        let mut val = 0i32;
        while n.is_multiple_of(P) {
            n /= P;
            val = val.saturating_add(1);
        }
        let mut digits = [0u64; N];
        let mut idx = 0;
        while n > 0 && idx < N {
            digits[idx] = n % P;
            n /= P;
            idx += 1;
        }
        Self {
            is_zero: false,
            val,
            digits,
        }
    }

    /// Embeds a signed integer $z \in \mathbb{Z}$ into $\mathbb{Q}_P$.
    #[must_use]
    pub fn from_i64(z: i64) -> Self {
        let () = Self::VALID_PADIC;
        if z == 0 {
            Self::ZERO
        } else if z > 0 {
            Self::from_u64(z as u64)
        } else {
            -Self::from_u64(z.unsigned_abs())
        }
    }

    /// Embeds a rational number $\text{num} / \text{den} \in \mathbb{Q}$ into $\mathbb{Q}_P$.
    ///
    /// # Errors
    /// Returns [`ScalarError::DivisionByZero`] when `den == 0`.
    pub fn from_ratio(num: i64, den: i64) -> Result<Self, ScalarError> {
        if den == 0 {
            return Err(ScalarError::DivisionByZero);
        }
        Self::from_i64(num).checked_div(Self::from_i64(den))
    }

    /// Returns `true` when `self == 0` (to working precision).
    #[inline]
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.is_zero
    }

    /// Discrete $p$-adic valuation $v_P(x) \in \mathbb{Z}$, or `None` ($+\infty$) for `ZERO`.
    #[inline]
    #[must_use]
    pub const fn valuation(&self) -> Option<i32> {
        if self.is_zero { None } else { Some(self.val) }
    }

    /// Non-Archimedean $p$-adic absolute value $|x|_P = P^{-v_P(x)}$ (`0.0` for `ZERO`).
    #[must_use]
    pub fn norm_f64(&self) -> f64 {
        if self.is_zero {
            0.0
        } else {
            libm::pow(P as f64, -f64::from(self.val))
        }
    }

    /// Canonical unit base-$P$ digits `[d_0, d_1, ..., d_{N-1}]` (`[0; N]` for `ZERO`).
    #[inline]
    #[must_use]
    pub const fn digits(&self) -> &[u64; N] {
        &self.digits
    }

    /// Evaluates the truncated integer residue $\sum_{i=0}^{k-1} d_i P^{v + i} \bmod P^k$
    /// for $v \ge 0$ when $P^k$ fits in `u128`.
    #[must_use]
    pub fn to_mod_pk_u128(&self, k: usize) -> Option<u128> {
        if self.is_zero {
            return Some(0);
        }
        if self.val < 0 {
            return None;
        }
        let mut p_pow = 1u128;
        for _ in 0..k {
            p_pow = p_pow.checked_mul(P as u128)?;
        }
        let shift = self.val as usize;
        if shift >= k {
            return Some(0);
        }
        let mut acc = 0u128;
        let mut cur_p = 1u128;
        for _ in 0..shift {
            cur_p = cur_p.checked_mul(P as u128)?;
        }
        let limit = N.min(k - shift);
        for i in 0..limit {
            let term = (self.digits[i] as u128).checked_mul(cur_p)? % p_pow;
            acc = (acc + term) % p_pow;
            if i + 1 < limit {
                cur_p = cur_p.checked_mul(P as u128)?;
            }
        }
        Some(acc)
    }

    /// Verifies the strong triangle (ultrametric) inequality
    /// $v_P(x + y) \ge \min(v_P(x), v_P(y))$ (with `None` treated as $+\infty$).
    #[must_use]
    pub fn satisfies_ultrametric(self, rhs: Self) -> bool {
        let sum = self + rhs;
        match (self.valuation(), rhs.valuation(), sum.valuation()) {
            (_, _, None) => true,
            (None, None, Some(_)) => false,
            (None, Some(vy), Some(vs)) => vs >= vy,
            (Some(vx), None, Some(vs)) => vs >= vx,
            (Some(vx), Some(vy), Some(vs)) => vs >= vx.min(vy),
        }
    }

    /// Checked multiplicative inverse $x^{-1} \in \mathbb{Q}_P^\times$.
    ///
    /// # Errors
    /// Returns [`ScalarError::NonInvertible`] when `self == 0`.
    pub fn checked_inverse(self) -> Result<Self, ScalarError> {
        let () = Self::VALID_PADIC;
        if self.is_zero {
            return Err(ScalarError::NonInvertible);
        }
        let inv_digits = zpn_invert_unit::<P, N>(&self.digits)?;
        Ok(Self {
            is_zero: false,
            val: self.val.saturating_neg(),
            digits: inv_digits,
        })
    }

    /// Checked $p$-adic division `self * rhs.checked_inverse()?`.
    ///
    /// # Errors
    /// Returns [`ScalarError::DivisionByZero`] when `rhs == 0`.
    pub fn checked_div(self, rhs: Self) -> Result<Self, ScalarError> {
        assert!(P >= 2);
        assert!(N >= 1);
        let inv = rhs
            .checked_inverse()
            .map_err(|_| ScalarError::DivisionByZero)?;
        Ok(self * inv)
    }

    /// Lifts a simple mod-$P$ polynomial root $r_0$ of $f(X) = \sum_{j=0}^m c_j X^j \in \mathbb{Z}[X]$
    /// to mod-$P^N$ precision in $\mathbb{Z}_P \subset \mathbb{Q}_P$ via quadratic Newton-Hensel iteration:
    /// $$x_{k+1} \equiv x_k - f(x_k)\,f'(x_k)^{-1} \pmod{P^{2^{k+1}}}.$$
    ///
    /// # Errors
    /// - Returns [`ScalarError::NotARootModP`] when $f(r_0) \not\equiv 0 \pmod P$ or `coeffs.is_empty()`.
    /// - Returns [`ScalarError::SingularDerivative`] when $f'(r_0) \equiv 0 \pmod P$.
    pub fn hensel_lift_root(coeffs: &[i64], r0: u64) -> Result<Self, ScalarError> {
        let () = Self::VALID_PADIC;
        if coeffs.is_empty() {
            return Err(ScalarError::NotARootModP);
        }
        let r0_mod = r0 % P;
        let (f0, df0) = eval_mod_p_and_deriv::<P>(coeffs, r0_mod);
        if !f0.is_zero() {
            return Err(ScalarError::NotARootModP);
        }
        if df0.is_zero() {
            return Err(ScalarError::SingularDerivative);
        }

        let mut x = [0u64; N];
        x[0] = r0_mod;
        let mut prec = 1usize;
        while prec < N {
            let (fx, dfx) = eval_zpn_poly_and_deriv::<P, N>(coeffs, &x);
            let inv_dfx =
                zpn_invert_unit::<P, N>(&dfx).map_err(|_| ScalarError::SingularDerivative)?;
            let step = zpn_mul::<P, N>(&fx, &inv_dfx);
            x = zpn_sub::<P, N>(&x, &step);
            prec = prec.saturating_mul(2);
        }
        Ok(Self::normalize_raw(0, x))
    }

    fn normalize_raw(mut val: i32, raw: [u64; N]) -> Self {
        let mut first_nz = N;
        for (i, &d) in raw.iter().enumerate() {
            if d != 0 {
                first_nz = i;
                break;
            }
        }
        if first_nz == N {
            return Self::ZERO;
        }
        let mut digits = [0u64; N];
        let count = N - first_nz;
        digits[..count].copy_from_slice(&raw[first_nz..N]);
        val = val.saturating_add(first_nz as i32);
        Self {
            is_zero: false,
            val,
            digits,
        }
    }
}

impl<const P: u64, const N: usize> Neg for Qp<P, N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        if self.is_zero {
            self
        } else {
            Self::normalize_raw(self.val, zpn_neg::<P, N>(&self.digits))
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
        let min_v = self.val.min(rhs.val);
        let sa = (self.val as i64 - min_v as i64) as usize;
        let sb = (rhs.val as i64 - min_v as i64) as usize;
        let mut a_shifted = [0u64; N];
        let mut b_shifted = [0u64; N];
        if sa < N {
            a_shifted[sa..N].copy_from_slice(&self.digits[..(N - sa)]);
        }
        if sb < N {
            b_shifted[sb..N].copy_from_slice(&rhs.digits[..(N - sb)]);
        }
        Self::normalize_raw(min_v, zpn_add::<P, N>(&a_shifted, &b_shifted))
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
            return Self::ZERO;
        }
        Self::normalize_raw(
            self.val.saturating_add(rhs.val),
            zpn_mul::<P, N>(&self.digits, &rhs.digits),
        )
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
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.checked_div(rhs).expect("Qp division by zero")
    }
}

impl<const P: u64, const N: usize> fmt::Display for Qp<P, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero {
            return write!(f, "0 + O({P}^{N})");
        }
        for (idx, &d) in self.digits.iter().enumerate() {
            if idx > 0 {
                write!(f, " + ")?;
            }
            let exp = self.val.saturating_add(idx as i32);
            write!(f, "{d}*{P}^{exp}")?;
        }
        write!(f, " + O({P}^{})", self.val.saturating_add(N as i32))
    }
}
