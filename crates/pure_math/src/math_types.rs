use crate::error::NumberTheoryError;
use dashu::base::{Abs, Sign};
use dashu::float::FBig;
pub use dashu::integer::{IBig, UBig};
use dashu::rational::RBig;
pub use math_scalars::{BigFloat, BigInt, BigRational, I256, ScalarError, U256};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, RemAssign, Sub, SubAssign};
/// Arbitrary-precision signed integer wrapper over [`dashu::integer::IBig`].
#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
pub struct Integer(pub IBig);
impl From<Integer> for IBig {
    fn from(i: Integer) -> Self {
        i.0
    }
}
impl Default for Integer {
    fn default() -> Self {
        Self::new()
    }
}
impl Integer {
    /// Constructs a new instance from the supplied parameters.
    pub fn new() -> Self {
        Integer(IBig::from(0u8))
    }
    /// Computes `from` (from).
    pub fn from<T>(v: T) -> Self
    where
        IBig: From<T>,
    {
        Integer(IBig::from(v))
    }
    /// Computes `to_f64` (to f64).
    pub fn to_f64(&self) -> f64 {
        use dashu::float::round::mode::HalfAway;
        let f: FBig<HalfAway, 2> = FBig::from(self.0.clone());
        f.to_f64().value()
    }
    /// Computes `to_i32` (to i32).
    pub fn to_i32(&self) -> Option<i32> {
        i32::try_from(&self.0).ok()
    }
    /// Computes `to_usize` (to usize).
    pub fn to_usize(&self) -> Option<usize> {
        usize::try_from(&self.0).ok()
    }
    /// Computes `to_u64` (to u64).
    pub fn to_u64(&self) -> Option<u64> {
        u64::try_from(&self.0).ok()
    }
    /// Converts `self` to `i64` if within bounds.
    pub fn to_i64(&self) -> Option<i64> {
        i64::try_from(&self.0).ok()
    }
    /// Returns `true` if `self == 0`.
    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
    /// Returns `true` if `self < 0`.
    pub fn is_negative(&self) -> bool {
        self.0.sign() == Sign::Negative
    }
    /// Returns the unsigned magnitude `|self|` as [`UBig`].
    pub fn unsigned_abs_ubig(&self) -> UBig {
        self.0.clone().into_parts().1
    }
    /// Shifts `self` left by `bits` ($n \cdot 2^{\text{bits}}$).
    pub fn shl_bits(&self, bits: usize) -> Self {
        Integer(&self.0 << bits)
    }
    /// Exact floor division $\lfloor a / b \rfloor$ for positive divisor `rhs`.
    pub fn div_floor(&self, rhs: &Self) -> Self {
        let mut q = &self.0 / &rhs.0;
        let r = &self.0 % &rhs.0;
        if r < IBig::ZERO {
            q -= IBig::ONE;
        }
        Integer(q)
    }
    /// Computes `assign` (assign).
    pub fn assign(&mut self, v: i128) {
        self.0 = IBig::from(v);
    }
    /// Computes `parse` (parse).
    pub fn parse(s: &str) -> Result<Self, NumberTheoryError> {
        Self::from_str_radix(s, 10)
    }
    /// Computes `from_str_radix` (from str radix).
    pub fn from_str_radix(s: &str, radix: u32) -> Result<Self, NumberTheoryError> {
        if !(2..=36).contains(&radix) {
            return Err(NumberTheoryError::InvalidRadix { radix });
        }
        IBig::from_str_radix(s, radix)
            .map(Integer)
            .map_err(|_| NumberTheoryError::InvalidIntegerLiteral { radix })
    }
    /// Returns `true` if `self` is an even integer ($n \equiv 0 \pmod{2}$).
    #[verified_engine::verified]
    pub fn is_even(&self) -> bool {
        &self.0 % IBig::from(2u8) == IBig::from(0u8)
    }
    /// Returns `true` if `other` divides `self` with zero remainder.
    #[verified_engine::verified]
    pub fn is_divisible(&self, other: &Integer) -> bool {
        (&self.0 % &other.0) == IBig::from(0u8)
    }
    /// Computes the integer square root $\lfloor \sqrt{n} \rfloor$.
    #[verified_engine::verified]
    pub fn sqrt(&self) -> Self {
        Integer(self.0.nth_root(2))
    }
    #[doc = include_str!("../../../docs/math_specs/snippets/int_pow_mod.md")]
    #[verified_engine::verified]
    pub fn pow_mod(&self, exp: &Integer, m: &Integer) -> Result<Self, NumberTheoryError> {
        if m.0 == IBig::from(0u8) {
            return Err(NumberTheoryError::ZeroModulus);
        }
        let mut base = &self.0 % &m.0;
        let mut exp_val = exp.0.clone();
        if exp_val < IBig::from(0u8) {
            let inv = self.invert(m)?;
            base = inv.0;
            exp_val = -exp_val;
        }
        let mut result = IBig::from(1u8);
        while exp_val > IBig::from(0u8) {
            if &exp_val % IBig::from(2u8) == IBig::from(1u8) {
                result = (result * &base) % &m.0;
            }
            exp_val /= IBig::from(2u8);
            base = (&base * &base) % &m.0;
        }
        Ok(Integer(result))
    }
    #[doc = include_str!("../../../docs/math_specs/snippets/int_invert.md")]
    #[verified_engine::verified]
    pub fn invert(&self, m: &Integer) -> Result<Self, NumberTheoryError> {
        if m.0 == IBig::from(0u8) {
            return Err(NumberTheoryError::ZeroModulus);
        }
        let (gcd, x, _y) = self.extended_gcd(m);
        if gcd.0 == IBig::from(1u8) {
            let mut r = x.0 % &m.0;
            if r < IBig::from(0u8) {
                r += m.0.clone().abs();
            }
            Ok(Integer(r))
        } else {
            Err(NumberTheoryError::NoModularInverse)
        }
    }
    /// Returns the non-negative magnitude $|n|$.
    #[verified_engine::verified]
    pub fn abs(self) -> Self {
        Integer(self.0.abs())
    }
    #[verified_engine::verified]
    fn extended_gcd(&self, other: &Integer) -> (Integer, Integer, Integer) {
        let mut old_r = self.0.clone();
        let mut r = other.0.clone();
        let mut old_s = IBig::from(1u8);
        let mut s = IBig::from(0u8);
        let mut old_t = IBig::from(0u8);
        let mut t = IBig::from(1u8);
        while r != IBig::from(0u8) {
            let quotient = &old_r / &r;
            let temp = r.clone();
            r = &old_r - &quotient * &r;
            old_r = temp;
            let temp = s.clone();
            s = &old_s - &quotient * &s;
            old_s = temp;
            let temp = t.clone();
            t = &old_t - &quotient * &t;
            old_t = temp;
        }
        (Integer(old_r), Integer(old_s), Integer(old_t))
    }
}

impl PartialEq<i32> for Integer {
    fn eq(&self, other: &i32) -> bool {
        self.0 == IBig::from(*other)
    }
}
impl PartialOrd<i32> for Integer {
    fn partial_cmp(&self, other: &i32) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&IBig::from(*other))
    }
}

math_commons::impl_binop!(Integer, Add, add);
math_commons::impl_binop!(Integer, Sub, sub);
math_commons::impl_binop!(Integer, Mul, mul);
math_commons::impl_binop!(Integer, Div, div);
math_commons::impl_binop!(Integer, Rem, rem);

impl AddAssign for Integer {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}
impl SubAssign for Integer {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}
impl MulAssign for Integer {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}
impl DivAssign for Integer {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}
impl RemAssign for Integer {
    fn rem_assign(&mut self, rhs: Self) {
        self.0 %= rhs.0;
    }
}
impl MulAssign<&Integer> for Integer {
    fn mul_assign(&mut self, rhs: &Self) {
        self.0 *= &rhs.0;
    }
}
impl AddAssign<&Integer> for Integer {
    fn add_assign(&mut self, rhs: &Self) {
        self.0 += &rhs.0;
    }
}
impl SubAssign<&Integer> for Integer {
    fn sub_assign(&mut self, rhs: &Self) {
        self.0 -= &rhs.0;
    }
}

/// Arbitrary-precision binary floating-point wrapper over [`BigFloat`] preserving mantissa precision.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Float(pub BigFloat);
impl Float {
    /// Fallibly constructs a `Float` with `prec` mantissa bits from finite `v`,
    /// rejecting `prec == 0` ([`ScalarError::ZeroPrecision`]) and `NaN` / `±Infinity` ([`ScalarError::NonFiniteFloat`]).
    #[verified_engine::verified]
    pub fn try_with_prec(prec: u32, v: f64) -> Result<Self, ScalarError> {
        BigFloat::try_with_prec(prec as usize, v).map(Float)
    }
    /// Constructs a `Float` with `prec` mantissa bits from finite `v`.
    ///
    /// # Panics
    /// Panics if `prec == 0` or `v` is non-finite (`NaN` or `±Infinity`); use [`Self::try_with_prec`] for fallible construction.
    pub fn with_val(prec: u32, v: f64) -> Self {
        Self::try_with_prec(prec, v)
            .expect("Float::with_val requires prec > 0 and a finite f64 value")
    }
    /// Active binary mantissa precision in bits.
    #[must_use]
    pub fn precision(&self) -> usize {
        self.0.precision()
    }
    /// Converts `self` to its closest `f64` value.
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        self.0.to_f64()
    }
    /// Computes `abs` (abs) preserving mantissa precision.
    pub fn abs(self) -> Self {
        Float(self.0.abs())
    }
    /// Rounds the arbitrary-precision float to the nearest `f64` integer value.
    #[verified_engine::verified]
    pub fn round(self) -> f64 {
        self.0.round_f64()
    }
}
impl Add for Float {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Float(self.0 + rhs.0)
    }
}
impl Sub for Float {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Float(self.0 - rhs.0)
    }
}
impl Mul for Float {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Float(self.0 * rhs.0)
    }
}
impl Div for Float {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Float(self.0 / rhs.0)
    }
}
impl<'a> Add<&'a Float> for Float {
    type Output = Float;
    fn add(self, rhs: &'a Float) -> Float {
        Float(self.0 + rhs.0.clone())
    }
}
impl<'a> Sub<&'a Float> for Float {
    type Output = Float;
    fn sub(self, rhs: &'a Float) -> Float {
        Float(self.0 - rhs.0.clone())
    }
}
impl<'a> Mul<&'a Float> for Float {
    type Output = Float;
    fn mul(self, rhs: &'a Float) -> Float {
        Float(self.0 * rhs.0.clone())
    }
}
impl<'b> Add<&'b Float> for &Float {
    type Output = Float;
    fn add(self, rhs: &'b Float) -> Float {
        Float(self.0.clone() + rhs.0.clone())
    }
}
impl<'b> Sub<&'b Float> for &Float {
    type Output = Float;
    fn sub(self, rhs: &'b Float) -> Float {
        Float(self.0.clone() - rhs.0.clone())
    }
}
impl<'b> Mul<&'b Float> for &Float {
    type Output = Float;
    fn mul(self, rhs: &'b Float) -> Float {
        Float(self.0.clone() * rhs.0.clone())
    }
}
impl<'b> Div<&'b Float> for &Float {
    type Output = Float;
    fn div(self, rhs: &'b Float) -> Float {
        Float(self.0.clone() / rhs.0.clone())
    }
}
impl AddAssign for Float {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0.clone() + rhs.0;
    }
}
impl SubAssign for Float {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0.clone() - rhs.0;
    }
}
impl PartialEq<f64> for Float {
    fn eq(&self, other: &f64) -> bool {
        BigFloat::try_with_prec(self.precision().max(53), *other).is_ok_and(|rhs| self.0 == rhs)
    }
}
impl PartialOrd<f64> for Float {
    fn partial_cmp(&self, other: &f64) -> Option<std::cmp::Ordering> {
        BigFloat::try_with_prec(self.precision().max(53), *other)
            .ok()
            .and_then(|rhs| self.0.partial_cmp(&rhs))
    }
}

/// Data structure `Rational` representing Rational state.
#[derive(Clone, Debug, PartialEq)]
pub struct Rational(pub RBig);
impl Default for Rational {
    fn default() -> Self {
        Self::new()
    }
}

impl Rational {
    /// Constructs a new instance from the supplied parameters.
    pub fn new() -> Self {
        Rational(RBig::from(0u8))
    }
    /// Computes `from` (from).
    pub fn from(v: (i128, i128)) -> Self {
        let n = dashu::integer::IBig::from(v.0);
        let d = dashu::integer::IBig::from(v.1);
        let (sign_d, mag_d) = d.into_parts();
        let mut real_n = n;
        if sign_d == dashu::base::Sign::Negative {
            real_n = -real_n;
        }
        Rational(RBig::from_parts(real_n, mag_d))
    }
    /// Constructs a reduced `Rational` from signed [`IBig`] numerator and positive [`UBig`] denominator.
    #[must_use]
    pub fn from_parts(numer: IBig, denom: UBig) -> Self {
        Rational(RBig::from_parts(numer, denom))
    }
    /// Fallibly constructs a reduced `Rational` from exact [`Integer`] numerator and denominator.
    pub fn try_from_integers(numer: Integer, denom: Integer) -> Result<Self, ScalarError> {
        BigRational::try_new(BigInt::from_ibig(numer.0), BigInt::from_ibig(denom.0))
            .map(|r| Rational(r.as_rbig().clone()))
    }
    /// Exact reduced signed [`Integer`] numerator without `f64` narrowing.
    #[must_use]
    #[verified_engine::verified]
    pub fn numerator(&self) -> Integer {
        Integer(self.0.numerator().clone())
    }
    /// Exact reduced positive [`Integer`] denominator without `f64` narrowing.
    #[must_use]
    #[verified_engine::verified]
    pub fn denominator(&self) -> Integer {
        Integer(IBig::from(self.0.denominator().clone()))
    }
    /// Converts `self` into a canonical [`BigRational`].
    #[must_use]
    pub fn to_big_rational(&self) -> BigRational {
        BigRational::from_rbig(self.0.clone())
    }
    /// Converts the numerator of the reduced fraction to `f64`.
    #[verified_engine::verified]
    pub fn numer(&self) -> f64 {
        use dashu::float::round::mode::HalfAway;
        let n_f: FBig<HalfAway, 2> = FBig::from(self.0.numerator().clone());
        n_f.to_f64().value()
    }
    /// Converts the positive denominator of the reduced fraction to `f64`.
    #[verified_engine::verified]
    pub fn denom(&self) -> f64 {
        use dashu::float::round::mode::HalfAway;
        let d_f: FBig<HalfAway, 2> = FBig::from(self.0.denominator().clone());
        d_f.to_f64().value()
    }
}
math_commons::impl_binop!(Rational, Add, add);
math_commons::impl_binop!(Rational, Sub, sub);
math_commons::impl_binop!(Rational, Mul, mul);
math_commons::impl_binop!(Rational, Div, div);

/// Submodule providing `ops` types, traits, and algorithms.
pub mod ops {
    /// Trait `Pow` defining Pow operations.
    pub trait Pow<Rhs> {
        /// Variant `type` (type).
        type Output;
        /// Computes `pow` (pow).
        fn pow(self, rhs: Rhs) -> Self::Output;
    }
    impl Pow<u32> for super::Integer {
        type Output = super::Integer;
        fn pow(self, rhs: u32) -> Self::Output {
            super::Integer(self.0.pow(rhs as usize))
        }
    }
    /// Trait `RemRounding` defining RemRounding operations.
    pub trait RemRounding<Rhs> {
        /// Variant `type` (type).
        type Output;
        /// Computes `rem_euc` (rem euc).
        fn rem_euc(self, rhs: Rhs) -> Self::Output;
    }
    impl RemRounding<super::Integer> for super::Integer {
        type Output = super::Integer;
        fn rem_euc(self, rhs: super::Integer) -> Self::Output {
            let mut r = &self.0 % &rhs.0;
            if r < dashu::integer::IBig::from(0u8) {
                use dashu::base::Abs;
                r += rhs.0.abs();
            }
            super::Integer(r)
        }
    }
    impl RemRounding<&super::Integer> for super::Integer {
        type Output = super::Integer;
        fn rem_euc(self, rhs: &super::Integer) -> Self::Output {
            let mut r = &self.0 % &rhs.0;
            if r < dashu::integer::IBig::from(0u8) {
                use dashu::base::Abs;
                r += rhs.0.clone().abs();
            }
            super::Integer(r)
        }
    }
}
