// [cite:hurwitz_kronecker]
// [cite:modular_polynomials_review]
//! # Real Short-Weierstrass Curves & Generic Curve Points
//!
//! Zero-allocation projective identity + affine coordinate representation [`CurvePoint<T>`]
//! and continuous chord-and-tangent arithmetic [`WeierstrassCurve`] over $\mathbb{R}$.

use crate::pure_math::elliptic_curves::EllipticCurveError;

const REAL_SINGULAR_TOL: f64 = 1e-12;

/// Projective point on an elliptic curve represented in affine coordinates plus the point at infinity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct WeierstrassCurve {
    /// Linear Weierstrass coefficient $a$.
    pub a: f64,
    /// Constant Weierstrass coefficient $b$.
    pub b: f64,
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for WeierstrassCurve {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Wire {
            a: f64,
            b: f64,
        }
        let wire = Wire::deserialize(deserializer)?;
        Self::new(wire.a, wire.b).map_err(serde::de::Error::custom)
    }
}

impl WeierstrassCurve {
    /// Constructs a validated non-singular curve over $\mathbb{R}$ ($\Delta = -16(4a^3 + 27b^2) \neq 0$).
    ///
    /// # Errors
    /// Returns [`EllipticCurveError::SingularRealCurve`] when inputs are non-finite or singular.
    #[verified_engine::verified]
    pub fn new(a: f64, b: f64) -> Result<Self, EllipticCurveError> {
        if !a.is_finite() || !b.is_finite() {
            return Err(EllipticCurveError::SingularRealCurve);
        }
        let disc_core = 4.0 * a * a * a + 27.0 * b * b;
        if !disc_core.is_finite() || disc_core.abs() <= REAL_SINGULAR_TOL {
            return Err(EllipticCurveError::SingularRealCurve);
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
