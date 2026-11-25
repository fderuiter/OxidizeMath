#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:modular_polynomials_review]
//! Stack-allocated Galois extension fields $\mathbb{F}_{P^{\text{DEG}}} \cong \mathbb{F}_P\[x\]/(m(x))$
//! ([`FpExt`]) with Horner shift-reduce multiplication, Frobenius endomorphism, and
//! Extended-Euclidean inversion using zero heap allocations.

use crate::error::ScalarError;
use crate::fp::Fp;
use crate::modular::is_prime_u64;
use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Compile-time specification of a monic degree-`DEG` modulus polynomial
/// $m(x) = x^{\text{DEG}} + \sum_{k=0}^{\text{DEG}-1} c_k x^k \in \mathbb{F}_P\[x\]$.
pub trait IrreducibleModulus<const P: u64, const DEG: usize>:
    Copy + Clone + PartialEq + Eq + fmt::Debug
{
    /// Ascending lower-degree coefficients `[c_0, c_1, ..., c_{DEG - 1}]` of monic $m(x)$.
    const COEFFS: [u64; DEG];
}

/// Element of the Galois quotient field $\mathbb{F}_{P^{\text{DEG}}} \cong \mathbb{F}_P\[x\]/(m(x))$.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct FpExt<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> {
    pub(crate) coeffs: [Fp<P>; DEG],
    _modulus: PhantomData<M>,
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> Default for FpExt<P, DEG, M> {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> FpExt<P, DEG, M> {
    /// Compile-time check enforcing prime characteristic `P` and extension degree `DEG >= 1`.
    pub const VALIDATE_EXTENSION: () = {
        assert!(
            is_prime_u64(P),
            "FpExt<P, DEG, M> characteristic P must be prime"
        );
        assert!(
            DEG >= 1,
            "FpExt<P, DEG, M> extension degree DEG must be >= 1"
        );
    };

    /// Additive identity $0 \in \mathbb{F}_{P^{\text{DEG}}}$.
    pub const ZERO: Self = {
        let () = Self::VALIDATE_EXTENSION;
        Self {
            coeffs: [Fp::<P>::ZERO; DEG],
            _modulus: PhantomData,
        }
    };

    /// Multiplicative identity $1 \in \mathbb{F}_{P^{\text{DEG}}}$.
    pub const ONE: Self = {
        let () = Self::VALIDATE_EXTENSION;
        let mut coeffs = [Fp::<P>::ZERO; DEG];
        coeffs[0] = Fp::<P>::ONE;
        Self {
            coeffs,
            _modulus: PhantomData,
        }
    };

    /// Constructs $\sum_{k=0}^{\text{DEG}-1} c_k \alpha^k$ from base-field coefficients.
    #[inline]
    #[must_use]
    pub const fn from_coeffs(coeffs: [Fp<P>; DEG]) -> Self {
        let () = Self::VALIDATE_EXTENSION;
        Self {
            coeffs,
            _modulus: PhantomData,
        }
    }

    /// Constructs from raw `u64` residues reduced modulo `P`.
    #[must_use]
    pub const fn from_u64_coeffs(raw: [u64; DEG]) -> Self {
        let () = Self::VALIDATE_EXTENSION;
        let mut coeffs = [Fp::<P>::ZERO; DEG];
        let mut i = 0;
        while i < DEG {
            coeffs[i] = Fp::<P>::from_u64(raw[i]);
            i += 1;
        }
        Self {
            coeffs,
            _modulus: PhantomData,
        }
    }

    /// Embeds base-field element `c` into the constant term $c \cdot \alpha^0$.
    #[must_use]
    pub const fn from_base(c: Fp<P>) -> Self {
        let () = Self::VALIDATE_EXTENSION;
        let mut coeffs = [Fp::<P>::ZERO; DEG];
        coeffs[0] = c;
        Self {
            coeffs,
            _modulus: PhantomData,
        }
    }

    /// Returns the generator $\alpha = x \bmod m(x)$ (or $-c_0$ when `DEG == 1`).
    #[must_use]
    pub fn generator() -> Self {
        let () = Self::VALIDATE_EXTENSION;
        if DEG == 1 {
            Self::from_base(-Fp::<P>::from_u64(M::COEFFS[0]))
        } else {
            let mut coeffs = [Fp::<P>::ZERO; DEG];
            coeffs[1] = Fp::<P>::ONE;
            Self::from_coeffs(coeffs)
        }
    }

    /// Borrows the canonical polynomial basis representation `[c_0, ..., c_{DEG - 1}]`.
    #[inline]
    #[must_use]
    pub const fn coeffs(&self) -> &[Fp<P>; DEG] {
        &self.coeffs
    }

    /// Returns `true` iff all coordinates equal zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| c.is_zero())
    }

    /// Multiplies `self` by $\alpha \equiv x \pmod{m(x)}$ via linear-feedback shift reduction.
    #[must_use]
    pub fn mul_by_alpha(self) -> Self {
        let top = self.coeffs[DEG - 1];
        let mut next = [Fp::<P>::ZERO; DEG];
        for k in (1..DEG).rev() {
            next[k] = self.coeffs[k - 1];
        }
        if !top.is_zero() {
            for (k, slot) in next.iter_mut().enumerate() {
                *slot -= top * Fp::<P>::from_u64(M::COEFFS[k]);
            }
        }
        Self::from_coeffs(next)
    }

    /// Scales `self` by base-field scalar `s`.
    #[must_use]
    pub fn scale(self, s: Fp<P>) -> Self {
        let mut out = [Fp::<P>::ZERO; DEG];
        for (slot, &c) in out.iter_mut().zip(&self.coeffs) {
            *slot = c * s;
        }
        Self::from_coeffs(out)
    }

    /// Binary exponentiation $a^e \in \mathbb{F}_{P^{\text{DEG}}}$.
    #[must_use]
    pub fn pow(self, mut exp: u64) -> Self {
        let mut acc = Self::ONE;
        let mut cur = self;
        while exp > 0 {
            if (exp & 1) == 1 {
                acc *= cur;
            }
            exp >>= 1;
            if exp > 0 {
                cur *= cur;
            }
        }
        acc
    }

    /// Frobenius automorphism $\varphi(a) = a^P \in \mathbb{F}_{P^{\text{DEG}}}$.
    #[inline]
    #[must_use]
    pub fn frobenius(self) -> Self {
        self.pow(P)
    }

    /// Fallible Extended-Euclidean inverse $a^{-1} \bmod m(x)$ using only `[Fp<P>; DEG]` stack arrays.
    ///
    /// Returns `None` if `self == 0` or if $\gcd(a(x), m(x)) \ne 1$ (e.g., reducible modulus).
    #[must_use]
    pub fn checked_inv(self) -> Option<Self> {
        let d_b = poly_deg(&self.coeffs)?;
        if d_b == 0 {
            return self.coeffs[0].checked_inv().map(Self::from_base);
        }
        // Step 1: divide monic m(x) = x^DEG + M::COEFFS by b(x) = self.coeffs.
        let mut r_prev = [Fp::<P>::ZERO; DEG];
        for (slot, &raw) in r_prev.iter_mut().zip(&M::COEFFS) {
            *slot = Fp::<P>::from_u64(raw);
        }
        let mut r_curr = self.coeffs;
        let b_lc_inv = r_curr[d_b].checked_inv()?;
        // Eliminate the implicit monic x^DEG term using factor = 1 * b_lc_inv at shift = DEG - d_b.
        let shift_top = DEG - d_b;
        for j in 0..d_b {
            r_prev[shift_top + j] -= b_lc_inv * r_curr[j];
        }
        let mut t_prev = alpha_monomial::<P, DEG, M>(b_lc_inv, shift_top);
        t_prev = -t_prev;
        reduce_in_place_deg(&mut r_prev, &r_curr, d_b, b_lc_inv, &Self::ONE, &mut t_prev);
        let mut t_curr = Self::ONE;
        core::mem::swap(&mut r_prev, &mut r_curr);
        core::mem::swap(&mut t_prev, &mut t_curr);

        // Remaining Euclidean steps stay strictly below degree DEG.
        while let Some(deg_c) = poly_deg(&r_curr) {
            let lc_inv = r_curr[deg_c].checked_inv()?;
            let mut next_t = t_prev;
            reduce_in_place_deg(&mut r_prev, &r_curr, deg_c, lc_inv, &t_curr, &mut next_t);
            t_prev = t_curr;
            t_curr = next_t;
            core::mem::swap(&mut r_prev, &mut r_curr);
        }
        if poly_deg(&r_prev) != Some(0) {
            return None;
        }
        r_prev[0].checked_inv().map(|inv0| t_prev.scale(inv0))
    }

    /// Fallible multiplicative inverse returning [`ScalarError::DivisionByZero`] (if zero) or
    /// [`ScalarError::NonInvertible`] (if $\gcd(a, m) \ne 1$).
    pub fn try_inverse(self) -> Result<Self, ScalarError> {
        if self.is_zero() {
            return Err(ScalarError::DivisionByZero);
        }
        self.checked_inv().ok_or(ScalarError::NonInvertible)
    }
}

fn poly_deg<const P: u64, const DEG: usize>(coeffs: &[Fp<P>; DEG]) -> Option<usize> {
    (0..DEG).rev().find(|&k| !coeffs[k].is_zero())
}

fn alpha_monomial<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>>(
    coeff: Fp<P>,
    power: usize,
) -> FpExt<P, DEG, M> {
    let mut acc = FpExt::<P, DEG, M>::from_base(coeff);
    for _ in 0..power {
        acc = acc.mul_by_alpha();
    }
    acc
}

fn reduce_in_place_deg<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>>(
    rem: &mut [Fp<P>; DEG],
    div: &[Fp<P>; DEG],
    div_deg: usize,
    div_lc_inv: Fp<P>,
    t_curr: &FpExt<P, DEG, M>,
    t_acc: &mut FpExt<P, DEG, M>,
) {
    while let Some(rem_deg) = poly_deg(rem) {
        if rem_deg < div_deg {
            break;
        }
        let shift = rem_deg - div_deg;
        let factor = rem[rem_deg] * div_lc_inv;
        rem[rem_deg] = Fp::<P>::ZERO;
        for j in 0..div_deg {
            rem[shift + j] -= factor * div[j];
        }
        let q_term = alpha_monomial::<P, DEG, M>(factor, shift);
        *t_acc -= q_term * *t_curr;
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> fmt::Debug
    for FpExt<P, DEG, M>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FpExt::<{P}, {DEG}>({:?})", self.coeffs)
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> fmt::Display
    for FpExt<P, DEG, M>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (k, c) in self.coeffs.iter().enumerate() {
            if k > 0 {
                f.write_str(" + ")?;
            }
            write!(f, "{}*a^{k}", c.value())?;
        }
        Ok(())
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> Add for FpExt<P, DEG, M> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = [Fp::<P>::ZERO; DEG];
        for ((slot, &a), &b) in out.iter_mut().zip(&self.coeffs).zip(&rhs.coeffs) {
            *slot = a + b;
        }
        Self::from_coeffs(out)
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> AddAssign for FpExt<P, DEG, M> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> Sub for FpExt<P, DEG, M> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = [Fp::<P>::ZERO; DEG];
        for ((slot, &a), &b) in out.iter_mut().zip(&self.coeffs).zip(&rhs.coeffs) {
            *slot = a - b;
        }
        Self::from_coeffs(out)
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> SubAssign for FpExt<P, DEG, M> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> Neg for FpExt<P, DEG, M> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let mut out = [Fp::<P>::ZERO; DEG];
        for (slot, &a) in out.iter_mut().zip(&self.coeffs) {
            *slot = -a;
        }
        Self::from_coeffs(out)
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> FpExt<P, DEG, M> {
    fn mul_ext(self, rhs: Self) -> Self {
        let mut acc = Self::ZERO;
        for &b_j in rhs.coeffs.iter().rev() {
            acc = acc.mul_by_alpha() + self.scale(b_j);
        }
        acc
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> Mul for FpExt<P, DEG, M> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_ext(rhs)
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> MulAssign for FpExt<P, DEG, M> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_ext(rhs);
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> Div for FpExt<P, DEG, M> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let inv = rhs
            .checked_inv()
            .expect("FpExt division by zero or non-invertible divisor");
        self.mul_ext(inv)
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> DivAssign for FpExt<P, DEG, M> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
