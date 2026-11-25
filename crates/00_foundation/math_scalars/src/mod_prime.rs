// [cite:math_commons]
use crate::context::{DomainField, DomainRing};
use crate::error::ScalarDomainError;

/// Bare `#![no_std]` runtime prime-modulus residue context $\mathrm{GF}(p)$ ($\mathbb{Z}/p\mathbb{Z}$).
///
/// Represents field/ring elements as canonical `u64` residues in `0..modulus` and widens intermediate
/// products to `u128` so all modular additions, subtractions, multiplications, and Horner evaluations
/// are exact without overflow or heap allocation for any `2 <= modulus <= u64::MAX`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModPrimeContext {
    modulus: u64,
}

impl ModPrimeContext {
    /// Fallibly constructs a modular domain context after validating `modulus >= 2`.
    #[inline]
    pub const fn try_new(modulus: u64) -> Result<Self, ScalarDomainError> {
        if modulus < 2 {
            Err(ScalarDomainError::InvalidModulus { modulus })
        } else {
            Ok(Self { modulus })
        }
    }

    /// Constructs a modular domain context, panicking if `modulus < 2`; prefer [`Self::try_new`].
    #[inline]
    #[must_use]
    pub const fn new(modulus: u64) -> Self {
        match Self::try_new(modulus) {
            Ok(ctx) => ctx,
            Err(_) => panic!("ModPrimeContext::new requires modulus >= 2"),
        }
    }

    /// Returns the active runtime modulus $p \ge 2$.
    #[inline(always)]
    #[must_use]
    pub const fn modulus(&self) -> u64 {
        self.modulus
    }

    /// Reduces an unsigned `u64` integer into its canonical representative in `0..modulus`.
    #[inline(always)]
    #[must_use]
    pub const fn reduce(&self, value: u64) -> u64 {
        value % self.modulus
    }

    /// Reduces a signed `i128` integer into its canonical Euclidean residue in `0..modulus`.
    #[inline]
    #[must_use]
    pub const fn reduce_i128(&self, value: i128) -> u64 {
        value.rem_euclid(self.modulus as i128) as u64
    }
}

#[inline]
fn extended_gcd_u64(a: u64, modulus: u64) -> (u64, u64) {
    let m = modulus as i128;
    let mut old_r = a as i128;
    let mut r = m;
    let mut old_s: i128 = 1;
    let mut s: i128 = 0;

    while r != 0 {
        let q = old_r / r;
        let next_r = old_r - q * r;
        old_r = r;
        r = next_r;

        let next_s = old_s - q * s;
        old_s = s;
        s = next_s;
    }

    (old_r as u64, old_s.rem_euclid(m) as u64)
}

impl DomainRing for ModPrimeContext {
    type Element = u64;

    #[inline(always)]
    fn zero(&self) -> u64 {
        0
    }

    #[inline(always)]
    fn one(&self) -> u64 {
        1 % self.modulus
    }

    #[inline(always)]
    fn is_zero(&self, elem: &u64) -> bool {
        (*elem).is_multiple_of(self.modulus)
    }

    #[inline(always)]
    fn add(&self, a: &u64, b: &u64) -> u64 {
        let ra = (*a % self.modulus) as u128;
        let rb = (*b % self.modulus) as u128;
        ((ra + rb) % (self.modulus as u128)) as u64
    }

    #[inline(always)]
    fn sub(&self, a: &u64, b: &u64) -> u64 {
        let ra = *a % self.modulus;
        let rb = *b % self.modulus;
        if ra >= rb {
            ra - rb
        } else {
            self.modulus - (rb - ra)
        }
    }

    #[inline(always)]
    fn mul(&self, a: &u64, b: &u64) -> u64 {
        let prod = ((*a % self.modulus) as u128) * ((*b % self.modulus) as u128);
        (prod % (self.modulus as u128)) as u64
    }

    #[inline(always)]
    fn neg(&self, a: &u64) -> u64 {
        let ra = *a % self.modulus;
        if ra == 0 { 0 } else { self.modulus - ra }
    }
}

impl DomainField for ModPrimeContext {
    #[inline]
    fn try_inv(&self, a: &u64) -> Result<u64, ScalarDomainError> {
        let residue = *a % self.modulus;
        if residue == 0 {
            return Err(ScalarDomainError::DivisionByZero);
        }
        let (gcd, inv) = extended_gcd_u64(residue, self.modulus);
        if gcd != 1 {
            return Err(ScalarDomainError::NonInvertibleResidue {
                residue,
                modulus: self.modulus,
            });
        }
        Ok(inv)
    }
}
