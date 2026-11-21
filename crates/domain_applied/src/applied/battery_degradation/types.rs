pub use crate::error::BatteryError;

use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

macro_rules! impl_math_ops {
    ($name:ident) => {
        impl $name {
            #[allow(missing_docs)]
            pub fn powf(&self, n: f64) -> Result<Self, BatteryError> {
                Self::new(self.0.powf(n))
            }
            #[allow(missing_docs)]
            pub fn sqrt(&self) -> Result<Self, BatteryError> {
                Self::new(self.0.sqrt())
            }
            #[allow(missing_docs)]
            pub fn ln(&self) -> Result<Self, BatteryError> {
                Self::new(self.0.ln())
            }
            #[allow(missing_docs)]
            pub fn exp(&self) -> Result<Self, BatteryError> {
                Self::new(self.0.exp())
            }
            #[allow(missing_docs)]
            pub fn abs(&self) -> Result<Self, BatteryError> {
                Self::new(self.0.abs())
            }
        }

        impl Add<f64> for $name {
            type Output = Result<Self, BatteryError>;
            fn add(self, rhs: f64) -> Self::Output {
                Self::new(self.0 + rhs)
            }
        }
        impl Sub<f64> for $name {
            type Output = Result<Self, BatteryError>;
            fn sub(self, rhs: f64) -> Self::Output {
                Self::new(self.0 - rhs)
            }
        }
        impl Mul<f64> for $name {
            type Output = Result<Self, BatteryError>;
            fn mul(self, rhs: f64) -> Self::Output {
                Self::new(self.0 * rhs)
            }
        }
        impl Div<f64> for $name {
            type Output = Result<Self, BatteryError>;
            fn div(self, rhs: f64) -> Self::Output {
                Self::new(self.0 / rhs)
            }
        }

        impl Add<$name> for f64 {
            type Output = Result<$name, BatteryError>;
            fn add(self, rhs: $name) -> Self::Output {
                $name::new(self + rhs.0)
            }
        }
        impl Sub<$name> for f64 {
            type Output = Result<$name, BatteryError>;
            fn sub(self, rhs: $name) -> Self::Output {
                $name::new(self - rhs.0)
            }
        }
        impl Mul<$name> for f64 {
            type Output = Result<$name, BatteryError>;
            fn mul(self, rhs: $name) -> Self::Output {
                $name::new(self * rhs.0)
            }
        }
        impl Div<$name> for f64 {
            type Output = Result<$name, BatteryError>;
            fn div(self, rhs: $name) -> Self::Output {
                $name::new(self / rhs.0)
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
        impl Mul<$name> for $name {
            type Output = Result<Self, BatteryError>;
            fn mul(self, rhs: $name) -> Self::Output {
                Self::new(self.0 * rhs.0)
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

/// Depth of Discharge (DoD) as a percentage (0.0 to 100.0).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct DepthOfDischarge(f64);

impl DepthOfDischarge {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, BatteryError> {
        if !(0.0..=100.0).contains(&value) {
            return Err(BatteryError::InvalidDepthOfDischarge(value));
        }
        Ok(Self(value))
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new_clamped(value: f64) -> Self {
        Self(value.clamp(0.0, 100.0))
    }
    #[allow(missing_docs)]
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

impl_math_ops!(DepthOfDischarge);

/// Battery Capacity as a fraction of initial capacity (0.0 to 1.0).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Capacity(f64);

impl Capacity {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, BatteryError> {
        if !(0.0..=1.0).contains(&value) {
            return Err(BatteryError::InvalidCapacity(value));
        }
        Ok(Self(value))
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new_clamped(value: f64) -> Self {
        Self(value.clamp(0.0, 1.0))
    }
    #[allow(missing_docs)]
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

impl_math_ops!(Capacity);

/// Number of Equivalent Full Cycles (EFC).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Cycles(f64);

impl Cycles {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, BatteryError> {
        if value < 0.0 {
            return Err(BatteryError::NegativeCycles(value));
        }
        Ok(Self(value))
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new_clamped(value: f64) -> Self {
        Self(value.max(0.0))
    }
    #[allow(missing_docs)]
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

impl_math_ops!(Cycles);
