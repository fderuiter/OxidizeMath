#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:math_scalars]

use crate::error::ScalarError;
use crate::fp::Fp;
use crate::poly::{EuclideanDomain, StackPoly};
use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Defines a monic degree-`DEG` irreducible polynomial
/// $M(x) = x^{\text{DEG}} + \sum_{i=0}^{\text{DEG}-1} c_i x^i \in \mathbb{F}_P[x]$
/// parameterizing the Galois extension field $\mathbb{F}_{P^{\text{DEG}}}$.
pub trait IrreducibleModulus<const P: u64, const DEG: usize>:
    Copy + Clone + fmt::Debug + PartialEq + Eq
{
    /// Lower coefficients `[c_0, c_1, ..., c_{DEG-1}]` of $M(x) = x^{\text{DEG}} + \sum c_i x^i$.
    const MODULUS_LOW: [u64; DEG];
}

/// Element of the Galois extension field $\mathbb{F}_{P^{\text{DEG}}} \cong \mathbb{F}_P[x] / (M(x))$
/// stored on the stack as `DEG` coordinates over [`Fp<P>`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FpExt<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> {
    coeffs: [Fp<P>; DEG],
    _marker: PhantomData<M>,
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> Default for FpExt<P, DEG, M> {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> FpExt<P, DEG, M> {
    const VALID_EXT: () = {
        let () = Fp::<P>::VALID_PRIME;
        assert!(DEG >= 1, "FpExt degree DEG must be >= 1");
    };

    /// Additive identity $0 \in \mathbb{F}_{P^{\text{DEG}}}$.
    pub const ZERO: Self = {
        let () = Self::VALID_EXT;
        Self {
            coeffs: [Fp::<P>::ZERO; DEG],
            _marker: PhantomData,
        }
    };

    /// Multiplicative identity $1 \in \mathbb{F}_{P^{\text{DEG}}}$.
    pub const ONE: Self = {
        let () = Self::VALID_EXT;
        let mut c = [Fp::<P>::ZERO; DEG];
        c[0] = Fp::<P>::ONE;
        Self {
            coeffs: c,
            _marker: PhantomData,
        }
    };

    /// Constructs $\sum_{i=0}^{\text{DEG}-1} c_i \alpha^i$ from [`Fp<P>`] coordinates.
    #[inline]
    #[must_use]
    pub const fn from_coeffs(coeffs: [Fp<P>; DEG]) -> Self {
        let () = Self::VALID_EXT;
        Self {
            coeffs,
            _marker: PhantomData,
        }
    }

    /// Constructs $\sum_{i=0}^{\text{DEG}-1} (c_i \bmod P) \alpha^i$ from `u64` coordinates.
    #[must_use]
    pub const fn from_u64_coeffs(raw: [u64; DEG]) -> Self {
        let () = Self::VALID_EXT;
        let mut coeffs = [Fp::<P>::ZERO; DEG];
        let mut i = 0;
        while i < DEG {
            coeffs[i] = Fp::<P>::new(raw[i]);
            i += 1;
        }
        Self {
            coeffs,
            _marker: PhantomData,
        }
    }

    /// Embeds a base-field scalar $c \in \mathbb{F}_P$ into $\mathbb{F}_{P^{\text{DEG}}}$.
    #[inline]
    #[must_use]
    pub const fn from_base(c: Fp<P>) -> Self {
        let () = Self::VALID_EXT;
        let mut coeffs = [Fp::<P>::ZERO; DEG];
        coeffs[0] = c;
        Self {
            coeffs,
            _marker: PhantomData,
        }
    }

    /// Adjoined root generator $\alpha = x \bmod M(x)$ (`1` when `DEG == 1` and $M(x) = x - c_0$).
    #[must_use]
    pub const fn generator() -> Self {
        let () = Self::VALID_EXT;
        if DEG == 1 {
            Self::from_base(Fp::<P>::new(M::MODULUS_LOW[0]).const_neg())
        } else {
            let mut coeffs = [Fp::<P>::ZERO; DEG];
            coeffs[1] = Fp::<P>::ONE;
            Self {
                coeffs,
                _marker: PhantomData,
            }
        }
    }

    /// Reduces an arbitrary [`StackPoly<Fp<P>, CAP>`] modulo $M(x)$ into $\mathbb{F}_{P^{\text{DEG}}}$.
    ///
    /// # Errors
    /// Returns [`ScalarError::CapacityExceeded`] if `CAP < DEG + 1`.
    pub fn from_stack_poly<const CAP: usize>(
        poly: &StackPoly<Fp<P>, CAP>,
    ) -> Result<Self, ScalarError> {
        let () = Self::VALID_EXT;
        if CAP < DEG + 1 {
            return Err(ScalarError::CapacityExceeded);
        }
        let mut mod_slice = [Fp::<P>::ZERO; CAP];
        for (i, &m) in M::MODULUS_LOW.iter().enumerate() {
            mod_slice[i] = Fp::<P>::new(m);
        }
        mod_slice[DEG] = Fp::<P>::ONE;
        let modulus = StackPoly::<Fp<P>, CAP>::from_slice(&mod_slice[..=DEG])?;
        let rem = poly.checked_rem(&modulus)?;
        let mut coeffs = [Fp::<P>::ZERO; DEG];
        for (i, &c) in rem.coeffs().iter().enumerate() {
            coeffs[i] = c;
        }
        Ok(Self::from_coeffs(coeffs))
    }

    /// Exports `self` as a [`StackPoly<Fp<P>, CAP>`] of degree $< \text{DEG}$.
    ///
    /// # Errors
    /// Returns [`ScalarError::CapacityExceeded`] if `CAP < DEG`.
    pub fn to_stack_poly<const CAP: usize>(&self) -> Result<StackPoly<Fp<P>, CAP>, ScalarError> {
        StackPoly::from_slice(&self.coeffs)
    }

    /// Borrows the canonical polynomial basis coordinates `[c_0, ..., c_{DEG-1}]`.
    #[inline]
    #[must_use]
    pub const fn coeffs(&self) -> &[Fp<P>; DEG] {
        &self.coeffs
    }

    /// Returns `true` when all coordinates are zero.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        let mut i = 0;
        while i < DEG {
            if !self.coeffs[i].is_zero() {
                return false;
            }
            i += 1;
        }
        true
    }

    /// Binary exponentiation `self^exp` in $\mathbb{F}_{P^{\text{DEG}}}$.
    #[must_use]
    pub fn pow(self, mut exp: u128) -> Self {
        let mut acc = Self::ONE;
        let mut base = self;
        while exp > 0 {
            if (exp & 1) == 1 {
                acc *= base;
            }
            base *= base;
            exp >>= 1;
        }
        acc
    }

    /// Evaluates the Frobenius automorphism $\phi(z) = z^P$.
    #[inline]
    #[must_use]
    pub fn frobenius(self) -> Self {
        self.pow(P as u128)
    }

    /// Fallible multiplicative inverse in $\mathbb{F}_{P^{\text{DEG}}}^\times$ computed via
    /// zero-allocation polynomial extended Euclidean reduction modulo $M(x)$.
    ///
    /// # Errors
    /// Returns [`ScalarError::NonInvertible`] when `self == 0` or $\gcd(\text{self}, M) \ne 1$.
    pub fn checked_inverse(self) -> Result<Self, ScalarError> {
        let () = Self::VALID_EXT;
        if self.is_zero() {
            return Err(ScalarError::NonInvertible);
        }
        let Some(a_deg) = poly_deg(&self.coeffs) else {
            return Err(ScalarError::NonInvertible);
        };
        if a_deg == 0 {
            return Ok(Self::from_base(self.coeffs[0].checked_inverse()?));
        }

        // First Euclidean step: divide monic M(x) = x^DEG + low by a(x) (degree a_deg < DEG)
        let (q0, r0) = div_monic_modulus_by(&M::MODULUS_LOW, &self.coeffs, a_deg)?;
        let mut old_r = self.coeffs;
        let mut r = r0;
        let mut old_t = Self::ONE.coeffs;
        let mut t = neg_coeffs(mul_mod_m::<P, DEG, M>(&q0, &old_t));

        while let Some(r_deg) = poly_deg(&r) {
            let old_deg = poly_deg(&old_r).unwrap_or(0);
            let (q, rem) = div_bounded(&old_r, old_deg, &r, r_deg)?;
            let qt = mul_mod_m::<P, DEG, M>(&q, &t);
            let next_t = sub_coeffs(&old_t, &qt);
            old_r = r;
            r = rem;
            old_t = t;
            t = next_t;
        }
        if poly_deg(&old_r) != Some(0) {
            return Err(ScalarError::NonInvertible);
        }
        let inv_lc = old_r[0].checked_inverse()?;
        let mut out = [Fp::<P>::ZERO; DEG];
        for i in 0..DEG {
            out[i] = old_t[i] * inv_lc;
        }
        Ok(Self::from_coeffs(out))
    }

    /// Fallible field division `self * rhs.checked_inverse()?`.
    ///
    /// # Errors
    /// Returns [`ScalarError::DivisionByZero`] when `rhs` is not invertible.
    pub fn checked_div(self, rhs: Self) -> Result<Self, ScalarError> {
        let inv = rhs
            .checked_inverse()
            .map_err(|_| ScalarError::DivisionByZero)?;
        Ok(self * inv)
    }
}

fn poly_deg<const P: u64, const DEG: usize>(c: &[Fp<P>; DEG]) -> Option<usize> {
    let mut i = DEG;
    while i > 0 {
        i -= 1;
        if !c[i].is_zero() {
            return Some(i);
        }
    }
    None
}

fn neg_coeffs<const P: u64, const DEG: usize>(mut a: [Fp<P>; DEG]) -> [Fp<P>; DEG] {
    for x in &mut a {
        *x = -*x;
    }
    a
}

fn sub_coeffs<const P: u64, const DEG: usize>(a: &[Fp<P>; DEG], b: &[Fp<P>; DEG]) -> [Fp<P>; DEG] {
    let mut out = [Fp::<P>::ZERO; DEG];
    for i in 0..DEG {
        out[i] = a[i] - b[i];
    }
    out
}

fn shift_x_mod_m<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>>(
    v: &[Fp<P>; DEG],
) -> [Fp<P>; DEG] {
    let top = v[DEG - 1];
    let mut out = [Fp::<P>::ZERO; DEG];
    for i in (1..DEG).rev() {
        out[i] = v[i - 1];
    }
    if !top.is_zero() {
        for (slot, &m) in out.iter_mut().zip(M::MODULUS_LOW.iter()) {
            *slot -= top * Fp::<P>::new(m);
        }
    }
    out
}

fn mul_mod_m<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>>(
    a: &[Fp<P>; DEG],
    b: &[Fp<P>; DEG],
) -> [Fp<P>; DEG] {
    let mut acc = [Fp::<P>::ZERO; DEG];
    for i in (0..DEG).rev() {
        acc = shift_x_mod_m::<P, DEG, M>(&acc);
        let ai = a[i];
        if !ai.is_zero() {
            for j in 0..DEG {
                acc[j] += ai * b[j];
            }
        }
    }
    acc
}

fn div_monic_modulus_by<const P: u64, const DEG: usize>(
    mod_low: &[u64; DEG],
    a: &[Fp<P>; DEG],
    a_deg: usize,
) -> Result<([Fp<P>; DEG], [Fp<P>; DEG]), ScalarError> {
    let inv_lead = a[a_deg].checked_inverse()?;
    let mut rem = [Fp::<P>::ZERO; DEG];
    for i in 0..DEG {
        rem[i] = Fp::<P>::new(mod_low[i]);
    }
    let mut quot = [Fp::<P>::ZERO; DEG];
    // Eliminate the leading 1 * x^DEG term using shift = DEG - a_deg (>= 1 because a_deg >= 1)
    let first_shift = DEG - a_deg;
    quot[first_shift] = inv_lead;
    for i in 0..a_deg {
        rem[i + first_shift] -= inv_lead * a[i];
    }
    if let Some(rem_deg) = poly_deg(&rem)
        && rem_deg >= a_deg
    {
        let (q_sub, r_sub) = div_bounded(&rem, rem_deg, a, a_deg)?;
        for i in 0..DEG {
            quot[i] += q_sub[i];
        }
        rem = r_sub;
    }
    Ok((quot, rem))
}

fn div_bounded<const P: u64, const DEG: usize>(
    num: &[Fp<P>; DEG],
    num_deg: usize,
    den: &[Fp<P>; DEG],
    den_deg: usize,
) -> Result<([Fp<P>; DEG], [Fp<P>; DEG]), ScalarError> {
    if num_deg < den_deg {
        return Ok(([Fp::<P>::ZERO; DEG], *num));
    }
    let inv_lead = den[den_deg].checked_inverse()?;
    let mut rem = *num;
    let mut quot = [Fp::<P>::ZERO; DEG];
    while let Some(cur_deg) = poly_deg(&rem) {
        if cur_deg < den_deg {
            break;
        }
        let shift = cur_deg - den_deg;
        let factor = rem[cur_deg] * inv_lead;
        quot[shift] += factor;
        for i in 0..=den_deg {
            rem[i + shift] -= factor * den[i];
        }
    }
    Ok((quot, rem))
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> fmt::Debug
    for FpExt<P, DEG, M>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("FpExt").field(&self.coeffs).finish()
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> fmt::Display
    for FpExt<P, DEG, M>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, c) in self.coeffs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", c.value())?;
        }
        write!(f, "]")
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> Neg for FpExt<P, DEG, M> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self::from_coeffs(neg_coeffs(self.coeffs))
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> Add for FpExt<P, DEG, M> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = [Fp::<P>::ZERO; DEG];
        for (slot, (&a, &b)) in out
            .iter_mut()
            .zip(self.coeffs.iter().zip(rhs.coeffs.iter()))
        {
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
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_coeffs(sub_coeffs(&self.coeffs, &rhs.coeffs))
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> SubAssign for FpExt<P, DEG, M> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> Mul for FpExt<P, DEG, M> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_coeffs(mul_mod_m::<P, DEG, M>(&self.coeffs, &rhs.coeffs))
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> MulAssign for FpExt<P, DEG, M> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> Div for FpExt<P, DEG, M> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.checked_div(rhs).expect("FpExt division by zero")
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> DivAssign for FpExt<P, DEG, M> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
