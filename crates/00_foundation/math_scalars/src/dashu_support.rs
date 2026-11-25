// [cite:math_commons]
use crate::context::{DomainField, DomainRing};
use crate::error::ScalarDomainError;
use crate::traits::{Field, Ring, Scalar};
use dashu::base::Abs;
use dashu::float::round::mode::HalfEven;
use dashu::float::{Context, FBig};
use dashu::integer::IBig;
use dashu::rational::RBig;

impl Scalar for IBig {}

impl Ring for IBig {
    #[inline]
    fn zero() -> Self {
        Self::from(0u8)
    }

    #[inline]
    fn one() -> Self {
        Self::from(1u8)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::from(0u8)
    }
}

impl Scalar for RBig {}

impl Ring for RBig {
    #[inline]
    fn zero() -> Self {
        Self::from(0u8)
    }

    #[inline]
    fn one() -> Self {
        Self::from(1u8)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::from(0u8)
    }
}

impl Field for RBig {
    #[inline]
    fn checked_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(Self::from(1u8) / self.clone())
        }
    }

    #[inline]
    fn multiplicative_inverse(&self) -> Self {
        Self::from(1u8) / self.clone()
    }
}

impl Scalar for FBig {}

/// Explicit `#![no_std]` (`alloc`-backed) dynamic-precision binary floating-point context over [`dashu::float::FBig`].
///
/// Stores `precision_bits` in an explicit value rather than `std` thread-local state so polynomial
/// and numerical kernels remain deterministic across bare-metal and WebAssembly targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DashuFloatContext {
    precision_bits: usize,
}

impl DashuFloatContext {
    /// Fallibly constructs a [`DashuFloatContext`] after validating `precision_bits >= 1`.
    #[inline]
    pub const fn try_new(precision_bits: usize) -> Result<Self, ScalarDomainError> {
        if precision_bits == 0 {
            Err(ScalarDomainError::InvalidPrecision { precision_bits })
        } else {
            Ok(Self { precision_bits })
        }
    }

    /// Constructs a [`DashuFloatContext`], panicking if `precision_bits == 0`; prefer [`Self::try_new`].
    #[inline]
    #[must_use]
    pub const fn new(precision_bits: usize) -> Self {
        match Self::try_new(precision_bits) {
            Ok(ctx) => ctx,
            Err(_) => panic!("DashuFloatContext::new requires precision_bits >= 1"),
        }
    }

    /// Returns the configured binary significand precision in bits.
    #[inline(always)]
    #[must_use]
    pub const fn precision_bits(&self) -> usize {
        self.precision_bits
    }

    #[inline(always)]
    fn inner_ctx(&self) -> Context<HalfEven> {
        Context::<HalfEven>::new(self.precision_bits)
    }

    /// Converts an `i64` integer into an [`FBig`] rounded to `self.precision_bits()`.
    #[inline]
    #[must_use]
    pub fn from_i64(&self, val: i64) -> FBig {
        FBig::from(val).with_precision(self.precision_bits).value()
    }

    /// Fallibly converts a finite `f64` into an [`FBig`] rounded to `self.precision_bits()`.
    #[inline]
    pub fn try_from_f64(&self, val: f64) -> Option<FBig> {
        FBig::try_from(val)
            .ok()
            .map(|f| f.with_precision(self.precision_bits).value())
    }

    /// Computes $\sqrt{x}$ rounded to `self.precision_bits()` for $x \ge 0$.
    #[inline]
    pub fn try_sqrt(&self, x: &FBig) -> Option<FBig> {
        if *x < FBig::<dashu::float::round::mode::Zero>::ZERO {
            None
        } else {
            Some(self.inner_ctx().sqrt(x.repr()).value().with_rounding())
        }
    }
}

impl DomainRing for DashuFloatContext {
    type Element = FBig;

    #[inline]
    fn zero(&self) -> FBig {
        FBig::ZERO
    }

    #[inline]
    fn one(&self) -> FBig {
        FBig::ONE.with_precision(self.precision_bits).value()
    }

    #[inline]
    fn is_zero(&self, elem: &FBig) -> bool {
        *elem == FBig::<dashu::float::round::mode::Zero>::ZERO
    }

    #[inline]
    fn add(&self, a: &FBig, b: &FBig) -> FBig {
        self.inner_ctx()
            .add(a.repr(), b.repr())
            .value()
            .with_rounding()
    }

    #[inline]
    fn sub(&self, a: &FBig, b: &FBig) -> FBig {
        self.inner_ctx()
            .sub(a.repr(), b.repr())
            .value()
            .with_rounding()
    }

    #[inline]
    fn mul(&self, a: &FBig, b: &FBig) -> FBig {
        self.inner_ctx()
            .mul(a.repr(), b.repr())
            .value()
            .with_rounding()
    }

    #[inline]
    fn neg(&self, a: &FBig) -> FBig {
        -a.clone()
    }
}

impl DomainField for DashuFloatContext {
    #[inline]
    fn try_inv(&self, a: &FBig) -> Result<FBig, ScalarDomainError> {
        if self.is_zero(a) {
            Err(ScalarDomainError::DivisionByZero)
        } else {
            let one = self.one();
            Ok(self
                .inner_ctx()
                .div(one.repr(), a.repr())
                .value()
                .with_rounding())
        }
    }

    #[inline]
    fn try_div(&self, a: &FBig, b: &FBig) -> Result<FBig, ScalarDomainError> {
        if self.is_zero(b) {
            Err(ScalarDomainError::DivisionByZero)
        } else {
            Ok(self
                .inner_ctx()
                .div(a.repr(), b.repr())
                .value()
                .with_rounding())
        }
    }
}

/// Explicit `alloc`-backed arbitrary-precision prime-modulus context over [`dashu::integer::IBig`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DashuModContext {
    modulus: IBig,
}

impl DashuModContext {
    /// Fallibly constructs a [`DashuModContext`] after validating `modulus >= 2`.
    pub fn try_new(modulus: IBig) -> Result<Self, ScalarDomainError> {
        if modulus < IBig::from(2u8) {
            let small = u64::try_from(&modulus).unwrap_or(0);
            Err(ScalarDomainError::InvalidModulus { modulus: small })
        } else {
            Ok(Self { modulus })
        }
    }

    /// Constructs a [`DashuModContext`], panicking if `modulus < 2`; prefer [`Self::try_new`].
    #[must_use]
    pub fn new(modulus: IBig) -> Self {
        Self::try_new(modulus).expect("DashuModContext::new requires modulus >= 2")
    }

    /// Returns a reference to the runtime modulus.
    #[inline]
    #[must_use]
    pub const fn modulus(&self) -> &IBig {
        &self.modulus
    }

    /// Reduces an arbitrary integer into `0..modulus`.
    #[must_use]
    pub fn reduce(&self, value: &IBig) -> IBig {
        let mut r = value % &self.modulus;
        if r < IBig::from(0u8) {
            r += self.modulus.clone().abs();
        }
        r
    }
}

fn extended_gcd_ibig(a: &IBig, m: &IBig) -> (IBig, IBig) {
    let mut old_r = a.clone();
    let mut r = m.clone();
    let mut old_s = IBig::from(1u8);
    let mut s = IBig::from(0u8);
    let zero = IBig::from(0u8);
    while r != zero {
        let q = &old_r / &r;
        let next_r = &old_r - &q * &r;
        old_r = r;
        r = next_r;
        let next_s = &old_s - &q * &s;
        old_s = s;
        s = next_s;
    }
    let mut inv = old_s % m;
    if inv < zero {
        inv += m.clone().abs();
    }
    (old_r, inv)
}

impl DomainRing for DashuModContext {
    type Element = IBig;

    #[inline]
    fn zero(&self) -> IBig {
        IBig::from(0u8)
    }

    #[inline]
    fn one(&self) -> IBig {
        IBig::from(1u8) % &self.modulus
    }

    #[inline]
    fn is_zero(&self, elem: &IBig) -> bool {
        self.reduce(elem) == IBig::from(0u8)
    }

    #[inline]
    fn add(&self, a: &IBig, b: &IBig) -> IBig {
        self.reduce(&(a + b))
    }

    #[inline]
    fn sub(&self, a: &IBig, b: &IBig) -> IBig {
        self.reduce(&(a - b))
    }

    #[inline]
    fn mul(&self, a: &IBig, b: &IBig) -> IBig {
        self.reduce(&(a * b))
    }

    #[inline]
    fn neg(&self, a: &IBig) -> IBig {
        let r = self.reduce(a);
        if r == IBig::from(0u8) {
            r
        } else {
            &self.modulus - r
        }
    }
}

impl DomainField for DashuModContext {
    fn try_inv(&self, a: &IBig) -> Result<IBig, ScalarDomainError> {
        let residue = self.reduce(a);
        if residue == IBig::from(0u8) {
            return Err(ScalarDomainError::DivisionByZero);
        }
        let (gcd, inv) = extended_gcd_ibig(&residue, &self.modulus);
        if gcd != IBig::from(1u8) {
            return Err(ScalarDomainError::NonInvertibleResidue {
                residue: u64::try_from(&residue).unwrap_or(u64::MAX),
                modulus: u64::try_from(&self.modulus).unwrap_or(u64::MAX),
            });
        }
        Ok(inv)
    }
}
