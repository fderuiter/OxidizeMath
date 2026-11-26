use crate::error::MarkovError;
use nalgebra::RealField;
use num_traits::ToPrimitive;

/// A time index for non-stationary transition matrices.
///
/// In basketball, this might represent the shot clock time (0-24 seconds).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeIndex<T: RealField + Copy + ToPrimitive> {
    /// The time value.
    time: T,
}

impl<T: RealField + Copy + ToPrimitive> TimeIndex<T> {
    /// Creates a new time index.
    #[verified_engine::verified]
    pub fn new(time: T) -> Result<Self, MarkovError> {
        if !time.is_finite() {
            return Err(MarkovError::InvalidState {
                reason: format!(
                    "Time must be finite, got {}",
                    time.to_f64().unwrap_or(f64::NAN)
                ),
            });
        }
        Ok(TimeIndex { time })
    }

    /// Returns the time value.
    #[verified_engine::verified]
    pub fn value(&self) -> T {
        self.time
    }
}
