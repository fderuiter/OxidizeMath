// [cite:hurwitz_kronecker]
// [cite:modular_polynomials_review]
//! # Short-Weierstrass Elliptic Curve Arithmetic
//!
//! Zero-allocation group-law operations ($P + Q$, $-P$, $\[k\]P$), subgroup orbit writers,
//! lattice point enumeration, and Frobenius trace calculations for non-singular short-Weierstrass
//! curves $y^2 = x^3 + ax + b$ over $\mathbb{R}$ (`f64`) and prime fields $\mathbb{F}_p$.

mod fp_curve;
pub use fp_curve::FpWeierstrassCurve;
use fp_curve::mod_pow_i64;
pub(crate) use fp_curve::{mod_inv_i64, mod_mul_i64};

use super::EllipticCurveError;
use crate::pure_math::number_theory::primes::is_prime;

/// Maximum prime modulus permitted for $\mathbb{F}_p \times \mathbb{F}_p$ lattice enumeration.
pub const MAX_LATTICE_PRIME: i64 = 997;

const REAL_SINGULAR_TOL: f64 = 1e-12;

/// Projective point on an elliptic curve represented in affine coordinates plus the point at infinity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CurvePoint<T> {
    /// Group identity element $\mathcal{O}$ at projective infinity $(0 : 1 : 0)$.
    Infinity,
    /// Finite affine curve point $(x, y)$.
    Affine {
        /// Horizontal coordinate $x$.
        x: T,
        /// Vertical coordinate $y$.
        y: T,
    },
}

impl<T> CurvePoint<T> {
    /// Returns `true` when this point is the group identity $\mathcal{O}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_infinity(&self) -> bool {
        matches!(self, Self::Infinity)
    }

    /// Extracts `(x, y)` coordinates when finite, or `None` for $\mathcal{O}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn coords(&self) -> Option<(T, T)>
    where
        T: Copy,
    {
        match *self {
            Self::Infinity => None,
            Self::Affine { x, y } => Some((x, y)),
        }
    }
}

#[verified_engine::verified]
pub(super) fn binary_scalar_mul<T: Copy>(
    scalar: i64,
    point: CurvePoint<T>,
    negate_fn: impl Fn(CurvePoint<T>) -> CurvePoint<T>,
    add_fn: impl Fn(CurvePoint<T>, CurvePoint<T>) -> CurvePoint<T>,
) -> CurvePoint<T> {
    if scalar == 0 || point.is_infinity() {
        return CurvePoint::Infinity;
    }
    let mut base = if scalar < 0 { negate_fn(point) } else { point };
    let mut rem = scalar.unsigned_abs();
    let mut acc = CurvePoint::Infinity;
    while rem > 0 {
        if rem & 1 == 1 {
            acc = add_fn(acc, base);
        }
        rem >>= 1;
        if rem > 0 {
            base = add_fn(base, base);
        }
    }
    acc
}

#[verified_engine::verified]
pub(super) fn populate_orbit<T: Copy>(
    generator: CurvePoint<T>,
    out: &mut [CurvePoint<T>],
    add_fn: impl Fn(CurvePoint<T>, CurvePoint<T>) -> CurvePoint<T>,
) -> usize {
    if out.is_empty() || generator.is_infinity() {
        return 0;
    }
    let mut current = generator;
    let mut count = 0;
    for slot in out.iter_mut() {
        *slot = current;
        count += 1;
        if current.is_infinity() {
            break;
        }
        current = add_fn(current, generator);
    }
    count
}

/// Non-singular short-Weierstrass elliptic curve $y^2 = x^3 + ax + b$ over $\mathbb{R}$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeierstrassCurve {
    /// Linear Weierstrass coefficient $a$.
    pub a: f64,
    /// Constant Weierstrass coefficient $b$.
    pub b: f64,
}

impl WeierstrassCurve {
    /// Constructs a validated non-singular curve over $\mathbb{R}$ ($\Delta = -16(4a^3 + 27b^2) \neq 0$).
    ///
    /// # Errors
    /// Returns [`EllipticCurveError::SingularCurve`] when inputs are non-finite or singular.
    #[verified_engine::verified]
    pub fn new(a: f64, b: f64) -> Result<Self, EllipticCurveError> {
        if !a.is_finite() || !b.is_finite() {
            return Err(EllipticCurveError::SingularCurve);
        }
        let disc_core = 4.0 * a * a * a + 27.0 * b * b;
        if !disc_core.is_finite() || disc_core.abs() <= REAL_SINGULAR_TOL {
            return Err(EllipticCurveError::SingularCurve);
        }
        Ok(Self { a, b })
    }

    /// Computes the Weierstrass discriminant $\Delta = -16(4a^3 + 27b^2)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn discriminant(&self) -> f64 {
        -16.0 * (4.0 * self.a.powi(3) + 27.0 * self.b.powi(2))
    }

    /// Evaluates the cubic right-hand side $f(x) = x^3 + ax + b$.
    #[must_use]
    #[verified_engine::verified]
    pub fn rhs(&self, x: f64) -> f64 {
        x * x * x + self.a * x + self.b
    }

    /// Checks whether `point` satisfies $y^2 = x^3 + ax + b$ within `tolerance`.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_on_curve(&self, point: CurvePoint<f64>, tolerance: f64) -> bool {
        match point {
            CurvePoint::Infinity => true,
            CurvePoint::Affine { x, y } => {
                x.is_finite() && y.is_finite() && (y * y - self.rhs(x)).abs() <= tolerance.max(0.0)
            }
        }
    }

    /// Computes the group inverse $-P$.
    #[must_use]
    #[verified_engine::verified]
    pub fn negate(&self, point: CurvePoint<f64>) -> CurvePoint<f64> {
        match point {
            CurvePoint::Infinity => CurvePoint::Infinity,
            CurvePoint::Affine { x, y } => CurvePoint::Affine {
                x,
                y: if y == 0.0 { 0.0 } else { -y },
            },
        }
    }

    /// Computes the chord or tangent slope between `p` and `q`, returning `None` for vertical lines or $\mathcal{O}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn secant_or_tangent_slope(&self, p: CurvePoint<f64>, q: CurvePoint<f64>) -> Option<f64> {
        let (CurvePoint::Affine { x: x1, y: y1 }, CurvePoint::Affine { x: x2, y: y2 }) = (p, q)
        else {
            return None;
        };
        let dx = x2 - x1;
        if dx.abs() <= REAL_SINGULAR_TOL {
            if (y1 - y2).abs() <= 1e-9 && y1.abs() > REAL_SINGULAR_TOL {
                Some((3.0 * x1 * x1 + self.a) / (2.0 * y1))
            } else {
                None
            }
        } else {
            Some((y2 - y1) / dx)
        }
    }

    /// Computes the chord-and-tangent group sum $P + Q$ without heap allocation.
    #[must_use]
    #[verified_engine::verified]
    pub fn add(&self, p: CurvePoint<f64>, q: CurvePoint<f64>) -> CurvePoint<f64> {
        match (p, q) {
            (CurvePoint::Infinity, other) | (other, CurvePoint::Infinity) => other,
            (CurvePoint::Affine { x: x1, y: y1 }, CurvePoint::Affine { x: x2, y: _ }) => {
                let Some(slope) = self.secant_or_tangent_slope(p, q) else {
                    return CurvePoint::Infinity;
                };
                let x3 = slope * slope - x1 - x2;
                CurvePoint::Affine {
                    x: x3,
                    y: slope * (x1 - x3) - y1,
                }
            }
        }
    }

    /// Doubles a curve point $2P = P + P$.
    #[must_use]
    #[verified_engine::verified]
    pub fn double(&self, point: CurvePoint<f64>) -> CurvePoint<f64> {
        self.add(point, point)
    }

    /// Computes scalar multiple $\[k\]P$ via iterative binary double-and-add.
    #[must_use]
    #[verified_engine::verified]
    pub fn scalar_mul(&self, scalar: i64, point: CurvePoint<f64>) -> CurvePoint<f64> {
        binary_scalar_mul(scalar, point, |pt| self.negate(pt), |u, v| self.add(u, v))
    }

    /// Writes multiples $\[1\]P, \[2\]P, \dots$ into `out` without heap allocation, returning count written.
    #[verified_engine::verified]
    pub fn write_orbit(&self, generator: CurvePoint<f64>, out: &mut [CurvePoint<f64>]) -> usize {
        populate_orbit(generator, out, |u, v| self.add(u, v))
    }
}

/// Non-singular short-Weierstrass curve $y^2 \equiv x^3 + ax + b \pmod p$ for runtime prime $p > 3$.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrimeWeierstrassCurve {
    /// Reduced linear coefficient $a \in [0, p)$.
    pub a: i64,
    /// Reduced constant coefficient $b \in [0, p)$.
    pub b: i64,
    /// Prime field modulus $p > 3$.
    pub p: i64,
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
            return Err(EllipticCurveError::InvalidModulus { modulus: p });
        }
        let (a_red, b_red) = (a.rem_euclid(p), b.rem_euclid(p));
        let a3 = mod_mul_i64(mod_mul_i64(a_red, a_red, p), a_red, p);
        let b2 = mod_mul_i64(b_red, b_red, p);
        if (mod_mul_i64(4, a3, p) + mod_mul_i64(27, b2, p)).rem_euclid(p) == 0 {
            return Err(EllipticCurveError::SingularCurve);
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
        let a3 = mod_mul_i64(mod_mul_i64(self.a, self.a, self.p), self.a, self.p);
        let b2 = mod_mul_i64(self.b, self.b, self.p);
        let core = (mod_mul_i64(4, a3, self.p) + mod_mul_i64(27, b2, self.p)).rem_euclid(self.p);
        mod_mul_i64(-16, core, self.p)
    }

    /// Evaluates $x^3 + ax + b \pmod p$ in $[0, p)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn rhs_mod_p(&self, x: i64) -> i64 {
        let xr = x.rem_euclid(self.p);
        let x3 = mod_mul_i64(mod_mul_i64(xr, xr, self.p), xr, self.p);
        (x3 + mod_mul_i64(self.a, xr, self.p) + self.b).rem_euclid(self.p)
    }

    /// Checks whether `point` lies on $y^2 \equiv x^3 + ax + b \pmod p$.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_on_curve(&self, point: CurvePoint<i64>) -> bool {
        match point {
            CurvePoint::Infinity => true,
            CurvePoint::Affine { x, y } => mod_mul_i64(y, y, self.p) == self.rhs_mod_p(x),
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
            let num =
                (mod_mul_i64(3, mod_mul_i64(x1, x1, self.p), self.p) + self.a).rem_euclid(self.p);
            Some(mod_mul_i64(
                num,
                mod_inv_i64(mod_mul_i64(2, y1, self.p), self.p)?,
                self.p,
            ))
        } else {
            let inv = mod_inv_i64((x2 - x1).rem_euclid(self.p), self.p)?;
            Some(mod_mul_i64((y2 - y1).rem_euclid(self.p), inv, self.p))
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
                let x3 = (mod_mul_i64(slope, slope, self.p) - x1 - x2).rem_euclid(self.p);
                CurvePoint::Affine {
                    x: x3,
                    y: (mod_mul_i64(slope, x1 - x3, self.p) - y1).rem_euclid(self.p),
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
            let rhs = self.rhs_mod_p(x);
            for y in 0..self.p {
                if mod_mul_i64(y, y, self.p) == rhs {
                    if written >= out.len() {
                        return Ok(written);
                    }
                    out[written] = CurvePoint::Affine { x, y };
                    written += 1;
                }
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
            } else if mod_pow_i64(rhs, exp, self.p) == 1 {
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
