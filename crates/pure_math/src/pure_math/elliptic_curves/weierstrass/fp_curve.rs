// [cite:hurwitz_kronecker]
// [cite:modular_polynomials_review]
//! # Const-Generic Finite Field Short-Weierstrass Curves
//!
//! Group-law operations on $y^2 = x^3 + ax + b$ over compile-time prime fields [`Fp<P>`].

use super::{CurvePoint, binary_scalar_mul, populate_orbit};
use crate::pure_math::algebra::fields::Fp;
use crate::pure_math::algebra::traits::{Field, Ring};
use crate::pure_math::elliptic_curves::EllipticCurveError;
use crate::pure_math::number_theory::primes::is_prime;

/// Non-singular short-Weierstrass curve $y^2 = x^3 + ax + b$ over compile-time prime field [`Fp<P>`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FpWeierstrassCurve<const P: i64> {
    /// Linear coefficient $a \in \mathbb{F}_P$.
    pub a: Fp<P>,
    /// Constant coefficient $b \in \mathbb{F}_P$.
    pub b: Fp<P>,
}

impl<const P: i64> FpWeierstrassCurve<P> {
    /// Constructs a validated curve over [`Fp<P>`] for prime $P > 3$.
    ///
    /// # Errors
    /// Returns [`EllipticCurveError::InvalidModulus`] when $P \le 3$ or composite, and
    /// [`EllipticCurveError::SingularCurve`] when $4a^3 + 27b^2 \equiv 0 \pmod P$.
    #[verified_engine::verified]
    pub fn new(a: i64, b: i64) -> Result<Self, EllipticCurveError> {
        if P <= 3 || !is_prime(P as u64) {
            return Err(EllipticCurveError::InvalidModulus { modulus: P });
        }
        let (a_fp, b_fp) = (Fp::<P>::new(a), Fp::<P>::new(b));
        let core = Fp::<P>::new(4) * a_fp * a_fp * a_fp + Fp::<P>::new(27) * b_fp * b_fp;
        if core.is_zero() {
            return Err(EllipticCurveError::SingularCurve);
        }
        Ok(Self { a: a_fp, b: b_fp })
    }

    /// Computes $\Delta \equiv -16(4a^3 + 27b^2) \pmod P$.
    #[must_use]
    #[verified_engine::verified]
    pub fn discriminant(&self) -> Fp<P> {
        let core = Fp::<P>::new(4) * self.a * self.a * self.a + Fp::<P>::new(27) * self.b * self.b;
        Fp::<P>::new(-16) * core
    }

    /// Returns `true` if `point` lies on $y^2 = x^3 + ax + b$ in $\mathbb{F}_P$.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_on_curve(&self, point: CurvePoint<Fp<P>>) -> bool {
        match point {
            CurvePoint::Infinity => true,
            CurvePoint::Affine { x, y } => y * y == x * x * x + self.a * x + self.b,
        }
    }

    /// Negates a point $-P$ in $E(\mathbb{F}_P)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn negate(&self, point: CurvePoint<Fp<P>>) -> CurvePoint<Fp<P>> {
        match point {
            CurvePoint::Infinity => CurvePoint::Infinity,
            CurvePoint::Affine { x, y } => CurvePoint::Affine { x, y: -y },
        }
    }

    /// Adds two points $P + Q$ in $E(\mathbb{F}_P)$ using [`Fp<P>`] field inversions.
    #[must_use]
    #[verified_engine::verified]
    pub fn add(&self, p: CurvePoint<Fp<P>>, q: CurvePoint<Fp<P>>) -> CurvePoint<Fp<P>> {
        match (p, q) {
            (CurvePoint::Infinity, other) | (other, CurvePoint::Infinity) => other,
            (CurvePoint::Affine { x: x1, y: y1 }, CurvePoint::Affine { x: x2, y: y2 }) => {
                let slope = if x1 == x2 {
                    if y1 != y2 || y1.is_zero() {
                        return CurvePoint::Infinity;
                    }
                    (Fp::<P>::new(3) * x1 * x1 + self.a)
                        * (Fp::<P>::new(2) * y1).multiplicative_inverse()
                } else {
                    (y2 - y1) * (x2 - x1).multiplicative_inverse()
                };
                let x3 = slope * slope - x1 - x2;
                CurvePoint::Affine {
                    x: x3,
                    y: slope * (x1 - x3) - y1,
                }
            }
        }
    }

    /// Doubles a point $2P$ in $E(\mathbb{F}_P)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn double(&self, point: CurvePoint<Fp<P>>) -> CurvePoint<Fp<P>> {
        self.add(point, point)
    }

    /// Computes scalar multiple $\[k\]P$ in $E(\mathbb{F}_P)$ without heap allocation.
    #[must_use]
    #[verified_engine::verified]
    pub fn scalar_mul(&self, scalar: i64, point: CurvePoint<Fp<P>>) -> CurvePoint<Fp<P>> {
        binary_scalar_mul(scalar, point, |pt| self.negate(pt), |u, v| self.add(u, v))
    }

    /// Writes cyclic subgroup orbit $\[1\]P, \[2\]P, \dots, \mathcal{O}$ into `out`, returning count written.
    #[verified_engine::verified]
    pub fn write_orbit(
        &self,
        generator: CurvePoint<Fp<P>>,
        out: &mut [CurvePoint<Fp<P>>],
    ) -> usize {
        populate_orbit(generator, out, |u, v| self.add(u, v))
    }
}

#[verified_engine::verified]
pub(crate) fn mod_mul_i64(lhs: i64, rhs: i64, modulus: i64) -> i64 {
    ((i128::from(lhs) * i128::from(rhs)).rem_euclid(i128::from(modulus))) as i64
}

#[verified_engine::verified]
pub(crate) fn mod_inv_i64(value: i64, modulus: i64) -> Option<i64> {
    let a = value.rem_euclid(modulus);
    if a == 0 {
        return None;
    }
    let (mut old_r, mut r, mut old_s, mut s) = (a, modulus, 1_i64, 0_i64);
    while r != 0 {
        let q = old_r / r;
        let next_r = old_r - q * r;
        old_r = r;
        r = next_r;
        let next_s = old_s - q * s;
        old_s = s;
        s = next_s;
    }
    (old_r == 1).then(|| old_s.rem_euclid(modulus))
}

#[verified_engine::verified]
pub(super) fn mod_pow_i64(mut base: i64, mut exp: u64, modulus: i64) -> i64 {
    let mut acc = 1_i64.rem_euclid(modulus);
    base = base.rem_euclid(modulus);
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mod_mul_i64(acc, base, modulus);
        }
        exp >>= 1;
        if exp > 0 {
            base = mod_mul_i64(base, base, modulus);
        }
    }
    acc
}
