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

        #[cfg(feature = "serde")]
        impl serde::Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                serializer.serialize_f64(self.0)
            }
        }

        #[cfg(feature = "serde")]
        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                let raw = f64::deserialize(deserializer)?;
                Self::new(raw).map_err(serde::de::Error::custom)
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
        if !value.is_finite() || !(0.0..=100.0).contains(&value) {
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
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
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
        if !value.is_finite() || value < 0.0 {
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

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::*;
    use serde::Deserialize;
    use serde::de::value::{Error as ValueError, F64Deserializer};

    #[test]
    fn test_battery_newtypes_round_trip() {
        let dod = DepthOfDischarge::new(80.0).unwrap();
        let cap = Capacity::new(0.85).unwrap();
        let cyc = Cycles::new(500.0).unwrap();
        assert_eq!(
            serde_json::from_str::<DepthOfDischarge>(&serde_json::to_string(&dod).unwrap())
                .unwrap(),
            dod
        );
        assert_eq!(
            serde_json::from_str::<Capacity>(&serde_json::to_string(&cap).unwrap()).unwrap(),
            cap
        );
        assert_eq!(
            serde_json::from_str::<Cycles>(&serde_json::to_string(&cyc).unwrap()).unwrap(),
            cyc
        );
    }

    #[test]
    fn test_battery_newtypes_reject_invalid_and_non_finite() {
        assert!(serde_json::from_str::<DepthOfDischarge>("-0.1").is_err());
        assert!(serde_json::from_str::<DepthOfDischarge>("100.1").is_err());
        assert!(serde_json::from_str::<Capacity>("-0.01").is_err());
        assert!(serde_json::from_str::<Capacity>("1.01").is_err());
        assert!(serde_json::from_str::<Cycles>("-1.0").is_err());

        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                DepthOfDischarge::deserialize(F64Deserializer::<ValueError>::new(bad)).is_err()
            );
            assert!(Capacity::deserialize(F64Deserializer::<ValueError>::new(bad)).is_err());
            assert!(Cycles::deserialize(F64Deserializer::<ValueError>::new(bad)).is_err());
        }
    }
}
