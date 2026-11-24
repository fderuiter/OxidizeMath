//! Dimensionally constrained battery degradation domain wrappers (`DepthOfDischarge`,
//! `Capacity`, `Cycles`).
// [cite:battery_degradation]

pub use crate::error::BatteryError;

use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

macro_rules! impl_linear_domain_ops {
    ($name:ident) => {
        impl Mul<f64> for $name {
            type Output = Result<Self, BatteryError>;
            fn mul(self, rhs: f64) -> Self::Output {
                Self::new(self.0 * rhs)
            }
        }

        impl Mul<$name> for f64 {
            type Output = Result<$name, BatteryError>;
            fn mul(self, rhs: $name) -> Self::Output {
                $name::new(self * rhs.0)
            }
        }

        impl Div<f64> for $name {
            type Output = Result<Self, BatteryError>;
            fn div(self, rhs: f64) -> Self::Output {
                Self::new(self.0 / rhs)
            }
        }

        impl Add<$name> for $name {
            type Output = Result<Self, BatteryError>;
            fn add(self, rhs: $name) -> Self::Output {
                Self::new(self.0 + rhs.0)
            }
        }

        impl Sub<$name> for $name {
            type Output = Result<Self, BatteryError>;
            fn sub(self, rhs: $name) -> Self::Output {
                Self::new(self.0 - rhs.0)
            }
        }

        impl Div<$name> for $name {
            type Output = f64;
            fn div(self, rhs: $name) -> f64 {
                self.0 / rhs.0
            }
        }

        impl From<$name> for f64 {
            fn from(val: $name) -> Self {
                val.0
            }
        }
    };
}

/// Depth of Discharge (DoD) as a bounded percentage in `[0.0, 100.0]`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct DepthOfDischarge(f64);

impl DepthOfDischarge {
    /// Constructs a validated [`DepthOfDischarge`] in `[0.0, 100.0]`.
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, BatteryError> {
        if !value.is_finite() || !(0.0..=100.0).contains(&value) {
            return Err(BatteryError::InvalidDepthOfDischarge(value));
        }
        Ok(Self(value))
    }

    /// Constructs a [`DepthOfDischarge`] clamped into `[0.0, 100.0]`.
    #[verified_engine::verified]
    pub fn new_clamped(value: f64) -> Self {
        Self(value.clamp(0.0, 100.0))
    }

    /// Returns the underlying percentage scalar by copy.
    #[inline(always)]
    pub const fn get(self) -> f64 {
        self.0
    }

    /// Returns the underlying percentage scalar.
    #[inline(always)]
    pub const fn value(&self) -> f64 {
        self.0
    }

    /// Returns the underlying percentage as `f64`.
    #[verified_engine::verified]
    pub fn as_f64(&self) -> f64 {
        self.0
    }
}

impl fmt::Display for DepthOfDischarge {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}%", self.0)
    }
}

impl_linear_domain_ops!(DepthOfDischarge);

/// Battery state-of-health capacity as a bounded fraction of initial capacity in `[0.0, 1.0]`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Capacity(f64);

impl Capacity {
    /// Constructs a validated [`Capacity`] in `[0.0, 1.0]`.
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, BatteryError> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(BatteryError::InvalidCapacity(value));
        }
        Ok(Self(value))
    }

    /// Constructs a [`Capacity`] clamped into `[0.0, 1.0]`.
    #[verified_engine::verified]
    pub fn new_clamped(value: f64) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Returns the underlying capacity fraction by copy.
    #[inline(always)]
    pub const fn get(self) -> f64 {
        self.0
    }

    /// Returns the underlying capacity fraction.
    #[inline(always)]
    pub const fn value(&self) -> f64 {
        self.0
    }

    /// Returns the underlying capacity fraction as `f64`.
    #[verified_engine::verified]
    pub fn as_f64(&self) -> f64 {
        self.0
    }
}

impl fmt::Display for Capacity {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}", self.0)
    }
}

impl_linear_domain_ops!(Capacity);

/// Non-negative count of Equivalent Full Cycles (EFC).
///
/// Self-multiplication (`Cycles * Cycles`), reciprocal coercion (`f64 / Cycles`), bare scalar
/// addition, and transcendental `.ln()` / `.exp()` are rejected at compile time:
///
/// ```compile_fail
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::battery_degradation::types::Cycles;
///
/// let n = Cycles::new(100.0).unwrap();
/// let _sq = n * n;
/// ```
///
/// ```compile_fail
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::battery_degradation::types::Cycles;
///
/// let n = Cycles::new(100.0).unwrap();
/// let _bad = n.ln();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Cycles(f64);

impl Cycles {
    /// Constructs a validated finite non-negative [`Cycles`] count.
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, BatteryError> {
        if !value.is_finite() || value < 0.0 {
            return Err(BatteryError::NegativeCycles(value));
        }
        Ok(Self(value))
    }

    /// Constructs a [`Cycles`] count clamped to `>= 0.0`.
    #[verified_engine::verified]
    pub fn new_clamped(value: f64) -> Self {
        Self(value.max(0.0))
    }

    /// Returns the underlying cycle count by copy.
    #[inline(always)]
    pub const fn get(self) -> f64 {
        self.0
    }

    /// Returns the underlying cycle count.
    #[inline(always)]
    pub const fn value(&self) -> f64 {
        self.0
    }

    /// Returns the underlying cycle count as `f64`.
    #[verified_engine::verified]
    pub fn as_f64(&self) -> f64 {
        self.0
    }
}

impl fmt::Display for Cycles {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1} cycles", self.0)
    }
}

impl_linear_domain_ops!(Cycles);
