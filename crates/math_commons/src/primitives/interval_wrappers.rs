// [cite:math_commons]
//! Domain-constrained interval enclosure wrappers (`[0, 1]`, `(0, \infty)`, `[0, \infty)`)
//! accompanying [`super::interval::Interval`].

use crate::error::MathError;
use crate::primitives::interval::Interval;
use crate::primitives::{NonNegativeFloat, PositiveFloat, UnitInterval};

/// Validated outward-rounded interval enclosure constrained to the unit domain $[0, 1]$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnitIntervalEnclosure(Interval);

impl UnitIntervalEnclosure {
    /// Constructs a unit-domain interval enclosure `[lo, hi]` with `0.0 <= lo <= hi <= 1.0`.
    #[verified_engine::verified]
    pub fn new(lo: f64, hi: f64) -> Result<Self, MathError> {
        let iv = Interval::new(lo, hi)?;
        if iv.lo() < 0.0 || iv.hi() > 1.0 {
            return Err(MathError::OutOfBounds {
                value: if iv.lo() < 0.0 { iv.lo() } else { iv.hi() },
                min: 0.0,
                max: 1.0,
            });
        }
        assert!(iv.lo() >= 0.0 && iv.hi() <= 1.0);
        Ok(Self(iv))
    }

    /// Projects an [`Interval`] onto $[0, 1]$ with outward rounding preserved inside $[0, 1]$.
    #[must_use]
    #[verified_engine::verified]
    pub fn clamp_from(interval: Interval) -> Self {
        let lo = interval.lo().clamp(0.0, 1.0);
        let hi = interval.hi().clamp(0.0, 1.0).max(lo);
        assert!(lo <= hi);
        Self(Interval::from_raw_checked(lo, hi))
    }

    /// Returns the underlying [`Interval`].
    #[inline]
    #[must_use]
    pub const fn interval(&self) -> Interval {
        self.0
    }

    /// Computes the outward-rounded complement $1 - [a, b] \subseteq [0, 1]$.
    #[must_use]
    #[verified_engine::verified]
    pub fn complement(&self) -> Self {
        Self::clamp_from(Interval::from_raw_checked(1.0, 1.0) - self.0)
    }
}

/// Validated outward-rounded interval enclosure strictly bounded on $(0, \infty)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositiveInterval(Interval);

impl PositiveInterval {
    /// Constructs a strictly positive interval enclosure `[lo, hi]` with `0.0 < lo <= hi`.
    #[verified_engine::verified]
    pub fn new(lo: f64, hi: f64) -> Result<Self, MathError> {
        let iv = Interval::new(lo, hi)?;
        if iv.lo() <= 0.0 {
            return Err(MathError::StrictlyOutOfBounds {
                value: iv.lo(),
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        assert!(iv.lo() > 0.0);
        Ok(Self(iv))
    }

    /// Returns the underlying [`Interval`].
    #[inline]
    #[must_use]
    pub const fn interval(&self) -> Interval {
        self.0
    }
}

/// Validated outward-rounded interval enclosure bounded on $[0, \infty)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonNegativeInterval(Interval);

impl NonNegativeInterval {
    /// Constructs a non-negative interval enclosure `[lo, hi]` with `0.0 <= lo <= hi`.
    #[verified_engine::verified]
    pub fn new(lo: f64, hi: f64) -> Result<Self, MathError> {
        let iv = Interval::new(lo, hi)?;
        if iv.lo() < 0.0 {
            return Err(MathError::OutOfBounds {
                value: iv.lo(),
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        assert!(iv.lo() >= 0.0);
        Ok(Self(iv))
    }

    /// Returns the underlying [`Interval`].
    #[inline]
    #[must_use]
    pub const fn interval(&self) -> Interval {
        self.0
    }
}

impl UnitInterval {
    /// Lifts this validated unit-interval scalar into an exact degenerate [`UnitIntervalEnclosure`].
    #[must_use]
    #[verified_engine::verified]
    pub fn to_interval(&self) -> UnitIntervalEnclosure {
        let v = self.value();
        assert!((0.0..=1.0).contains(&v));
        UnitIntervalEnclosure(Interval::from_raw_checked(v, v))
    }
}

impl PositiveFloat {
    /// Lifts this validated positive scalar into an exact degenerate [`PositiveInterval`].
    #[must_use]
    #[verified_engine::verified]
    pub fn to_interval(&self) -> PositiveInterval {
        let v = self.value();
        assert!(v > 0.0);
        PositiveInterval(Interval::from_raw_checked(v, v))
    }
}

impl NonNegativeFloat {
    /// Lifts this validated non-negative scalar into an exact degenerate [`NonNegativeInterval`].
    #[must_use]
    #[verified_engine::verified]
    pub fn to_interval(&self) -> NonNegativeInterval {
        let v = self.value();
        assert!(v >= 0.0);
        NonNegativeInterval(Interval::from_raw_checked(v, v))
    }
}
