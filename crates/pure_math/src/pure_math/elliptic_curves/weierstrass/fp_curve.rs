// [cite:hurwitz_kronecker]
// [cite:modular_polynomials_review]
//! # Const-Generic & Lattice-Capped Prime-Field Short-Weierstrass Curves
//!
//! Group-law operations on $y^2 = x^3 + ax + b$ over compile-time prime fields [`Fp<P>`]
//! and lattice-capped runtime prime fields [`PrimeWeierstrassCurve`].

use super::real_curve::{CurvePoint, binary_scalar_mul, populate_orbit};
use crate::pure_math::algebra::fields::Fp;
use crate::pure_math::algebra::traits::{Field, Ring};
use crate::pure_math::elliptic_curves::EllipticCurveError;
use crate::pure_math::elliptic_curves::modular_fp::{
    mod_inv_fp, mod_mul, mod_pow_fp, tonelli_shanks_fp,
};
use crate::pure_math::number_theory::primes::is_prime;

/// Maximum prime modulus permitted for $\mathbb{F}_p \times \mathbb{F}_p$ lattice enumeration.
pub const MAX_LATTICE_PRIME: i64 = 997;

/// Non-singular short-Weierstrass curve $y^2 = x^3 + ax + b$ over compile-time prime field [`Fp<P>`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct FpWeierstrassCurve<const P: i64> {
    /// Linear coefficient $a \in \mathbb{F}_P$.
    pub a: Fp<P>,
    /// Constant coefficient $b \in \mathbb{F}_P$.
    pub b: Fp<P>,
}

#[cfg(feature = "serde")]
impl<'de, const P: i64> serde::Deserialize<'de> for FpWeierstrassCurve<P> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Wire<const P: i64> {
            a: Fp<P>,
            b: Fp<P>,
        }
        let wire = Wire::<P>::deserialize(deserializer)?;
        Self::new(wire.a.value, wire.b.value).map_err(serde::de::Error::custom)
    }
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
            return Err(EllipticCurveError::InvalidModulus { p: P });
        }
        let (a_fp, b_fp) = (Fp::<P>::new(a), Fp::<P>::new(b));
        let core = Fp::<P>::new(4) * a_fp * a_fp * a_fp + Fp::<P>::new(27) * b_fp * b_fp;
        if core.is_zero() {
            return Err(EllipticCurveError::SingularCurve {
                p: P,
                a: a_fp.value,
                b: b_fp.value,
            });
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

/// Non-singular short-Weierstrass curve $y^2 \equiv x^3 + ax + b \pmod p$ for runtime prime $p > 3$.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct PrimeWeierstrassCurve {
    /// Reduced linear coefficient $a \in [0, p)$.
    pub a: i64,
    /// Reduced constant coefficient $b \in [0, p)$.
    pub b: i64,
    /// Prime field modulus $p > 3$.
    pub p: i64,
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for PrimeWeierstrassCurve {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Wire {
            a: i64,
            b: i64,
            p: i64,
        }
        let wire = Wire::deserialize(deserializer)?;
        Self::new(wire.a, wire.b, wire.p).map_err(serde::de::Error::custom)
    }
}

impl PrimeWeierstrassCurve {
    /// Constructs a validated curve over $\mathbb{F}_p$ for runtime prime $p > 3$.
    ///
    /// # Errors
    /// Returns [`EllipticCurveError::InvalidModulus`] if $p \le 3$ or composite, and
    /// [`EllipticCurveError::SingularCurve`] if $4a^3 + 27b^2 \equiv 0 \pmod p$.
    #[verified_engine::verified]
    pub fn new(a: i64, b: i64, p: i64) -> Result<Self, EllipticCurveError> {
        if p <= 3 || !is_prime(p as u64) {
            return Err(EllipticCurveError::InvalidModulus { p });
        }
        let (a_red, b_red) = (a.rem_euclid(p), b.rem_euclid(p));
        let a3 = mod_mul(mod_mul(a_red, a_red, p), a_red, p);
        let b2 = mod_mul(b_red, b_red, p);
        if (mod_mul(4, a3, p) + mod_mul(27, b2, p)).rem_euclid(p) == 0 {
            return Err(EllipticCurveError::SingularCurve {
                p,
                a: a_red,
                b: b_red,
            });
        }
        Ok(Self {
            a: a_red,
            b: b_red,
            p,
        })
    }

    /// Computes discriminant $\Delta \equiv -16(4a^3 + 27b^2) \pmod p$ in $[0, p)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn discriminant_mod_p(&self) -> i64 {
        let a3 = mod_mul(mod_mul(self.a, self.a, self.p), self.a, self.p);
        let b2 = mod_mul(self.b, self.b, self.p);
        let core = (mod_mul(4, a3, self.p) + mod_mul(27, b2, self.p)).rem_euclid(self.p);
        mod_mul(-16, core, self.p)
    }

    /// Evaluates $x^3 + ax + b \pmod p$ in $[0, p)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn rhs_mod_p(&self, x: i64) -> i64 {
        let xr = x.rem_euclid(self.p);
        let x3 = mod_mul(mod_mul(xr, xr, self.p), xr, self.p);
        (x3 + mod_mul(self.a, xr, self.p) + self.b).rem_euclid(self.p)
    }

    /// Checks whether `point` lies on $y^2 \equiv x^3 + ax + b \pmod p$.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_on_curve(&self, point: CurvePoint<i64>) -> bool {
        match point {
            CurvePoint::Infinity => true,
            CurvePoint::Affine { x, y } => mod_mul(y, y, self.p) == self.rhs_mod_p(x),
        }
    }

    /// Negates a point $-P \equiv (x, -y \bmod p)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn negate(&self, point: CurvePoint<i64>) -> CurvePoint<i64> {
        match point {
            CurvePoint::Infinity => CurvePoint::Infinity,
            CurvePoint::Affine { x, y } => CurvePoint::Affine {
                x: x.rem_euclid(self.p),
                y: (-y).rem_euclid(self.p),
            },
        }
    }

    /// Returns the modular secant or tangent slope in $[0, p)$, or `None` for vertical lines / $\mathcal{O}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn slope_mod_p(&self, p: CurvePoint<i64>, q: CurvePoint<i64>) -> Option<i64> {
        let (CurvePoint::Affine { x: x1, y: y1 }, CurvePoint::Affine { x: x2, y: y2 }) = (p, q)
        else {
            return None;
        };
        let (x1, y1) = (x1.rem_euclid(self.p), y1.rem_euclid(self.p));
        let (x2, y2) = (x2.rem_euclid(self.p), y2.rem_euclid(self.p));
        if x1 == x2 {
            if y1 != y2 || y1 == 0 {
                return None;
            }
            let num = (mod_mul(3, mod_mul(x1, x1, self.p), self.p) + self.a).rem_euclid(self.p);
            Some(mod_mul(
                num,
                mod_inv_fp(mod_mul(2, y1, self.p), self.p)?,
                self.p,
            ))
        } else {
            let inv = mod_inv_fp((x2 - x1).rem_euclid(self.p), self.p)?;
            Some(mod_mul((y2 - y1).rem_euclid(self.p), inv, self.p))
        }
    }

    /// Adds two points $P + Q$ in $E(\mathbb{F}_p)$ without heap allocation.
    #[must_use]
    #[verified_engine::verified]
    pub fn add(&self, p: CurvePoint<i64>, q: CurvePoint<i64>) -> CurvePoint<i64> {
        match (p, q) {
            (CurvePoint::Infinity, CurvePoint::Infinity) => CurvePoint::Infinity,
            (CurvePoint::Infinity, CurvePoint::Affine { x, y })
            | (CurvePoint::Affine { x, y }, CurvePoint::Infinity) => CurvePoint::Affine {
                x: x.rem_euclid(self.p),
                y: y.rem_euclid(self.p),
            },
            (CurvePoint::Affine { x: x1, y: y1 }, CurvePoint::Affine { x: x2, y: _ }) => {
                let Some(slope) = self.slope_mod_p(p, q) else {
                    return CurvePoint::Infinity;
                };
                let (x1, y1, x2) = (
                    x1.rem_euclid(self.p),
                    y1.rem_euclid(self.p),
                    x2.rem_euclid(self.p),
                );
                let x3 = (mod_mul(slope, slope, self.p) - x1 - x2).rem_euclid(self.p);
                CurvePoint::Affine {
                    x: x3,
                    y: (mod_mul(slope, x1 - x3, self.p) - y1).rem_euclid(self.p),
                }
            }
        }
    }

    /// Doubles a point $2P$ in $E(\mathbb{F}_p)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn double(&self, point: CurvePoint<i64>) -> CurvePoint<i64> {
        self.add(point, point)
    }

    /// Computes scalar multiple $\[k\]P$ in $E(\mathbb{F}_p)$ via iterative binary double-and-add.
    #[must_use]
    #[verified_engine::verified]
    pub fn scalar_mul(&self, scalar: i64, point: CurvePoint<i64>) -> CurvePoint<i64> {
        binary_scalar_mul(scalar, point, |pt| self.negate(pt), |u, v| self.add(u, v))
    }

    /// Writes cyclic subgroup orbit $\[1\]P, \[2\]P, \dots, \mathcal{O}$ into `out` without heap allocation.
    #[verified_engine::verified]
    pub fn write_orbit(&self, generator: CurvePoint<i64>, out: &mut [CurvePoint<i64>]) -> usize {
        populate_orbit(generator, out, |u, v| self.add(u, v))
    }

    /// Enumerates affine curve points over $\mathbb{F}_p \times \mathbb{F}_p$ into `out` (capped at $p \le 997$).
    ///
    /// # Errors
    /// Returns [`EllipticCurveError::ModulusExceedsLatticeCap`] when $p > 997$.
    #[verified_engine::verified]
    pub fn write_affine_points(
        &self,
        out: &mut [CurvePoint<i64>],
    ) -> Result<usize, EllipticCurveError> {
        if self.p > MAX_LATTICE_PRIME {
            return Err(EllipticCurveError::ModulusExceedsLatticeCap {
                modulus: self.p,
                max_modulus: MAX_LATTICE_PRIME,
            });
        }
        let mut written = 0;
        for x in 0..self.p {
            let (roots, count) = tonelli_shanks_fp(self.rhs_mod_p(x), self.p);
            for &y in roots.iter().take(count) {
                if written >= out.len() {
                    return Ok(written);
                }
                out[written] = CurvePoint::Affine { x, y };
                written += 1;
            }
        }
        Ok(written)
    }

    /// Computes total group order $\#E(\mathbb{F}_p)$ (including $\mathcal{O}$) for $p \le 997$ without heap allocation.
    ///
    /// # Errors
    /// Returns [`EllipticCurveError::ModulusExceedsLatticeCap`] when $p > 997$.
    #[verified_engine::verified]
    pub fn point_count(&self) -> Result<u64, EllipticCurveError> {
        if self.p > MAX_LATTICE_PRIME {
            return Err(EllipticCurveError::ModulusExceedsLatticeCap {
                modulus: self.p,
                max_modulus: MAX_LATTICE_PRIME,
            });
        }
        let exp = ((self.p - 1) / 2) as u64;
        let mut count = 1_u64;
        for x in 0..self.p {
            let rhs = self.rhs_mod_p(x);
            if rhs == 0 {
                count += 1;
            } else if mod_pow_fp(rhs, exp, self.p) == 1 {
                count += 2;
            }
        }
        Ok(count)
    }

    /// Computes Frobenius trace $a_p = p + 1 - \#E(\mathbb{F}_p)$ for $p \le 997$.
    ///
    /// # Errors
    /// Returns [`EllipticCurveError::ModulusExceedsLatticeCap`] when $p > 997$.
    #[verified_engine::verified]
    pub fn frobenius_trace(&self) -> Result<i64, EllipticCurveError> {
        Ok(self.p + 1 - (self.point_count()? as i64))
    }

    /// Verifies Hasse's theorem $|a_p| \le 2\sqrt{p}$ via exact integer inequality $a_p^2 \le 4p$.
    ///
    /// # Errors
    /// Returns [`EllipticCurveError::ModulusExceedsLatticeCap`] when $p > 997$.
    #[verified_engine::verified]
    pub fn satisfies_hasse_weil(&self) -> Result<bool, EllipticCurveError> {
        let a_p = self.frobenius_trace()?;
        Ok(i128::from(a_p) * i128::from(a_p) <= 4 * i128::from(self.p))
    }
}

/// Returns the Hasse-Weil envelope amplitude $2\sqrt{p}$ for prime $p$.
#[must_use]
#[verified_engine::verified]
pub fn hasse_weil_bound(p: u64) -> f64 {
    2.0 * (p as f64).sqrt()
}
