//! Exact rational and dyadic geometric primitives for algorithmic information bounds.
//!
//! Provides arbitrary-precision 2D points, lines, dyadic rationals $m / 2^r$, Euclidean
//! distances, and orthogonal line projections backing the geometric measure theory
//! experiments in Cholak et al.

use crate::math_types::Rational;
use dashu::base::Sign;
use dashu::integer::{IBig, UBig};
use dashu::rational::RBig;

/// Dyadic scale bits ($2^{64}$) used when approximating irrational Euclidean distances.
const DISTANCE_DYADIC_SCALE_BITS: usize = 64;

/// A dyadic rational number of the form $m / 2^r$ with integer numerator $m$ and precision $r$.
///
/// # Examples
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::algorithmic_information::geometry::DyadicRational;
///
/// let d = DyadicRational::new(3, 2);
/// assert!((d.to_f64() - 0.75).abs() < 1e-12);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DyadicRational {
    /// Signed integer numerator $m$.
    pub m: i64,
    /// Non-negative binary precision exponent $r$ representing denominator $2^r$.
    pub r: u32,
}

impl DyadicRational {
    /// Constructs the dyadic rational $m / 2^r$.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(m: i64, r: u32) -> Self {
        Self { m, r }
    }

    /// Converts $m / 2^r$ into an exact reduced [`Rational`].
    #[must_use]
    #[verified_engine::verified]
    pub fn to_rational(&self) -> Rational {
        let numer = IBig::from(self.m);
        let denom = UBig::from(1u8) << (self.r as usize);
        Rational(RBig::from_parts(numer, denom))
    }

    /// Evaluates $m / 2^r$ as an IEEE-754 `f64` approximation.
    #[must_use]
    #[verified_engine::verified]
    pub fn to_f64(&self) -> f64 {
        (self.m as f64) * 2.0_f64.powi(-(self.r as i32))
    }

    /// Computes the dyadic floor $\lfloor 2^r \cdot v \rfloor / 2^r$ of a [`Rational`] at precision $r$.
    #[must_use]
    #[verified_engine::verified]
    pub fn from_rational_floor(value: &Rational, r: u32) -> Self {
        let scale = IBig::from(1u8) << (r as usize);
        let scaled_num = value.0.numerator() * scale;
        let denom = IBig::from(value.0.denominator().clone());
        let mut quotient = &scaled_num / &denom;
        let remainder = &scaled_num % &denom;
        if remainder < IBig::from(0u8) {
            quotient -= IBig::from(1u8);
        }
        let clamped = if let Ok(v) = i64::try_from(&quotient) {
            v
        } else if quotient < IBig::from(0u8) {
            i64::MIN
        } else {
            i64::MAX
        };
        Self { m: clamped, r }
    }

    /// Reduces $m / 2^r$ by stripping shared powers of two until $m$ is odd or $r = 0$.
    #[must_use]
    #[verified_engine::verified]
    pub fn reduced(&self) -> Self {
        if self.m == 0 {
            return Self { m: 0, r: 0 };
        }
        let shift = self.m.trailing_zeros().min(self.r);
        Self {
            m: self.m >> shift,
            r: self.r - shift,
        }
    }
}

/// A point $(x, y) \in \mathbb{Q}^2$ represented with arbitrary-precision rationals.
///
/// # Examples
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::algorithmic_information::geometry::Point2D;
///
/// let p = Point2D::from_ints(3, 4);
/// assert_eq!(p.x.numer(), 3.0);
/// assert_eq!(p.y.numer(), 4.0);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Point2D {
    /// Horizontal rational coordinate.
    pub x: Rational,
    /// Vertical rational coordinate.
    pub y: Rational,
}

impl Point2D {
    /// Constructs a 2D point from exact [`Rational`] coordinates.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(x: Rational, y: Rational) -> Self {
        Self { x, y }
    }

    /// Constructs a 2D point $(x, y)$ from integer coordinates.
    #[must_use]
    #[verified_engine::verified]
    pub fn from_ints(x: i128, y: i128) -> Self {
        Self {
            x: Rational::from((x, 1)),
            y: Rational::from((y, 1)),
        }
    }
}

/// A 2D point with dyadic rational coordinates $(m_x / 2^{r_x}, m_y / 2^{r_y})$.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DyadicPoint2D {
    /// Horizontal dyadic coordinate.
    pub x: DyadicRational,
    /// Vertical dyadic coordinate.
    pub y: DyadicRational,
}

impl DyadicPoint2D {
    /// Constructs a dyadic 2D point from two [`DyadicRational`] coordinates.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(x: DyadicRational, y: DyadicRational) -> Self {
        Self { x, y }
    }

    /// Quantizes a [`Point2D`] via componentwise dyadic floor $\lfloor p \rfloor_r$ at precision $r$.
    #[must_use]
    #[verified_engine::verified]
    pub fn from_point_floor(point: &Point2D, r: u32) -> Self {
        Self {
            x: DyadicRational::from_rational_floor(&point.x, r),
            y: DyadicRational::from_rational_floor(&point.y, r),
        }
    }

    /// Converts this dyadic point into an exact [`Point2D`].
    #[must_use]
    #[verified_engine::verified]
    pub fn to_point(&self) -> Point2D {
        Point2D {
            x: self.x.to_rational(),
            y: self.y.to_rational(),
        }
    }
}

/// An affine line in $\mathbb{R}^2$ parameterized by $L(t) = \text{origin} + t \cdot \text{direction}$.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Line {
    /// Anchor point on the line.
    pub origin: Point2D,
    /// Direction vector spanning the line.
    pub direction: Point2D,
}

impl Line {
    /// Constructs a line through `origin` spanned by `direction`.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(origin: Point2D, direction: Point2D) -> Self {
        Self { origin, direction }
    }
}

/// Computes the exact squared Euclidean distance $\|p_1 - p_2\|_2^2$ between two rational points.
#[must_use]
#[verified_engine::verified]
pub fn distance_squared(p1: &Point2D, p2: &Point2D) -> Rational {
    let dx = p1.x.clone() - p2.x.clone();
    let dy = p1.y.clone() - p2.y.clone();
    (&dx * &dx) + (&dy * &dy)
}

/// Computes the Euclidean distance $\|p_1 - p_2\|_2$ between two 2D points.
///
/// Rational Pythagorean distances simplify to exact reduced rationals, while irrational
/// square roots are approximated to $2^{-64}$ dyadic accuracy via integer square roots.
///
/// # Examples
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::math_types::Rational;
/// use math_explorer::pure_math::algorithmic_information::geometry::{distance, Point2D};
///
/// let p1 = Point2D::from_ints(0, 0);
/// let p2 = Point2D::from_ints(3, 4);
/// assert_eq!(distance(&p1, &p2), Rational::from((5, 1)));
/// ```
#[must_use]
#[verified_engine::verified]
pub fn distance(p1: &Point2D, p2: &Point2D) -> Rational {
    let d2 = distance_squared(p1, p2);
    let (sign, num_mag) = d2.0.numerator().clone().into_parts();
    if sign == Sign::Negative || num_mag == UBig::from(0u8) {
        return Rational::new();
    }
    let den_mag = d2.0.denominator().clone();
    let scaled_radicand = (&num_mag * &den_mag) << (2 * DISTANCE_DYADIC_SCALE_BITS);
    let root = scaled_radicand.nth_root(2);
    let scaled_den = den_mag << DISTANCE_DYADIC_SCALE_BITS;
    Rational(RBig::from_parts(IBig::from(root), scaled_den))
}

/// Computes the exact orthogonal projection of `point` onto `line`.
///
/// If `line.direction` is the zero vector, `line.origin` is returned unchanged.
///
/// # Examples
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::algorithmic_information::geometry::{
///     project_point_onto_line, Line, Point2D,
/// };
///
/// let line = Line::new(Point2D::from_ints(0, 0), Point2D::from_ints(1, 1));
/// let projected = project_point_onto_line(&Point2D::from_ints(2, 0), &line);
/// assert_eq!(projected, Point2D::from_ints(1, 1));
/// ```
#[must_use]
#[verified_engine::verified]
pub fn project_point_onto_line(point: &Point2D, line: &Line) -> Point2D {
    let dx = &line.direction.x;
    let dy = &line.direction.y;
    let norm_sq = (dx * dx) + (dy * dy);
    if norm_sq.0.numerator() == &IBig::from(0u8) {
        return line.origin.clone();
    }
    let vx = point.x.clone() - line.origin.x.clone();
    let vy = point.y.clone() - line.origin.y.clone();
    let dot = (&vx * dx) + (&vy * dy);
    let t = dot / norm_sq;
    Point2D {
        x: &line.origin.x + &(&t * dx),
        y: &line.origin.y + &(&t * dy),
    }
}

// [cite:algorithmic_information_rust]
