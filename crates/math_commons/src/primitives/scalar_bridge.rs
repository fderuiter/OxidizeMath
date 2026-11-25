//! Lossless `From` and checked `TryFrom` conversions between [`crate::primitives::Interval`]
//! and [`math_scalars::Interval<f64>`].
// [cite:math_commons]

use super::Interval;
use crate::error::MathError;

impl From<Interval> for math_scalars::Interval<f64> {
    #[inline]
    fn from(value: Interval) -> Self {
        Self {
            lo: value.lo(),
            hi: value.hi(),
        }
    }
}

impl TryFrom<math_scalars::Interval<f64>> for Interval {
    type Error = MathError;

    #[inline]
    fn try_from(value: math_scalars::Interval<f64>) -> Result<Self, Self::Error> {
        Self::new(value.lo, value.hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitives_interval_and_math_scalars_interval_bridge_roundtrip() {
        let bounded = Interval::new(-1.25, 3.5).expect("valid bounded interval");
        let scalar_iv: math_scalars::Interval<f64> = bounded.into();
        assert_eq!(scalar_iv.lo, -1.25);
        assert_eq!(scalar_iv.hi, 3.5);

        let back = Interval::try_from(scalar_iv).expect("finite ordered interval");
        assert_eq!(back, bounded);

        let unbounded = math_scalars::Interval::new(0.0, f64::INFINITY);
        assert!(Interval::try_from(unbounded).is_err());
    }
}
