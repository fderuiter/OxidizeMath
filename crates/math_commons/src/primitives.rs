// [cite:math_commons]
use std::ops::Deref;

#[cfg(feature = "serde")]
#[doc(hidden)]
pub use serde as __serde;

#[macro_export]
#[allow(missing_docs)]
macro_rules! impl_bounded_float_serde {
    ($name:ident) => {
        #[cfg(feature = "serde")]
        impl $crate::primitives::__serde::Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: $crate::primitives::__serde::Serializer,
            {
                serializer.serialize_f64(self.0)
            }
        }

        #[cfg(feature = "serde")]
        impl<'de> $crate::primitives::__serde::Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: $crate::primitives::__serde::Deserializer<'de>,
            {
                let raw = <f64 as $crate::primitives::__serde::Deserialize<'de>>::deserialize(
                    deserializer,
                )?;
                Self::new(raw).map_err($crate::primitives::__serde::de::Error::custom)
            }
        }
    };
}

#[macro_export]
#[allow(missing_docs)]
macro_rules! impl_binop {
    ($name:ident) => {
        impl std::ops::Mul<f64> for $name {
            type Output = Result<Self, String>;
            fn mul(self, rhs: f64) -> Self::Output {
                Self::new(self.0 * rhs)
            }
        }
        impl std::ops::Mul<$name> for f64 {
            type Output = Result<$name, String>;
            fn mul(self, rhs: $name) -> Self::Output {
                $name::new(self * rhs.0)
            }
        }
        impl std::ops::Div<f64> for $name {
            type Output = Result<Self, String>;
            fn div(self, rhs: f64) -> Self::Output {
                Self::new(self.0 / rhs)
            }
        }
    };
    ($type:ident, $trt:ident, $mth:ident) => {
        impl std::ops::$trt for $type {
            type Output = $type;
            fn $mth(self, rhs: $type) -> $type {
                $type(self.0.$mth(rhs.0))
            }
        }
        impl std::ops::$trt<&$type> for $type {
            type Output = $type;
            fn $mth(self, rhs: &$type) -> $type {
                $type(self.0.$mth(&rhs.0))
            }
        }
        impl std::ops::$trt<$type> for &$type {
            type Output = $type;
            fn $mth(self, rhs: $type) -> $type {
                $type(self.0.clone().$mth(rhs.0))
            }
        }
        impl std::ops::$trt<&$type> for &$type {
            type Output = $type;
            fn $mth(self, rhs: &$type) -> $type {
                $type(self.0.clone().$mth(&rhs.0))
            }
        }
    };
}

/// A macro to define a bounded float primitive.
#[macro_export]
macro_rules! define_bounded_float {
    (
        $name:ident,
        $min:expr,
        $max:expr,
        $doc:expr
    ) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
        pub struct $name(f64);

        impl $name {
            /// Creates a new instance, validating bounds.
            pub fn new(value: f64) -> Result<Self, String> {
                if ($min..=$max).contains(&value) && value.is_finite() {
                    Ok(Self(value))
                } else {
                    Err(format!(
                        "Value {} is outside the allowed range [{}, {}]",
                        value, $min, $max
                    ))
                }
            }

            /// Retrieves the raw value.
            pub fn value(&self) -> f64 {
                self.0
            }

            #[allow(missing_docs)]
            pub fn powf(&self, n: f64) -> Result<Self, String> {
                Self::new(self.0.powf(n))
            }

            #[allow(missing_docs)]
            pub fn sqrt(&self) -> Result<Self, String> {
                Self::new(self.0.sqrt())
            }

            #[allow(missing_docs)]
            pub fn ln(&self) -> Result<Self, String> {
                Self::new(self.0.ln())
            }

            #[allow(missing_docs)]
            pub fn exp(&self) -> Result<Self, String> {
                Self::new(self.0.exp())
            }

            #[allow(missing_docs)]
            pub fn abs(&self) -> Result<Self, String> {
                Self::new(self.0.abs())
            }
        }

        impl Deref for $name {
            type Target = f64;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl From<$name> for f64 {
            fn from(val: $name) -> Self {
                val.0
            }
        }

        $crate::impl_binop!($name);
        $crate::impl_bounded_float_serde!($name);
    };
}

/// A macro to define a strictly bounded float primitive (min < value < max).
#[macro_export]
macro_rules! define_strictly_bounded_float {
    (
        $name:ident,
        $min:expr,
        $max:expr,
        $doc:expr
    ) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
        pub struct $name(f64);

        impl $name {
            /// Creates a new instance, validating bounds strictly.
            pub fn new(value: f64) -> Result<Self, String> {
                if value > $min && value < $max && value.is_finite() {
                    Ok(Self(value))
                } else {
                    Err(format!(
                        "Value {} is outside the strictly allowed range ({}, {})",
                        value, $min, $max
                    ))
                }
            }

            /// Retrieves the raw value.
            pub fn value(&self) -> f64 {
                self.0
            }

            #[allow(missing_docs)]
            pub fn powf(&self, n: f64) -> Result<Self, String> {
                Self::new(self.0.powf(n))
            }

            #[allow(missing_docs)]
            pub fn sqrt(&self) -> Result<Self, String> {
                Self::new(self.0.sqrt())
            }

            #[allow(missing_docs)]
            pub fn ln(&self) -> Result<Self, String> {
                Self::new(self.0.ln())
            }

            #[allow(missing_docs)]
            pub fn exp(&self) -> Result<Self, String> {
                Self::new(self.0.exp())
            }

            #[allow(missing_docs)]
            pub fn abs(&self) -> Result<Self, String> {
                Self::new(self.0.abs())
            }
        }

        impl Deref for $name {
            type Target = f64;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl From<$name> for f64 {
            fn from(val: $name) -> Self {
                val.0
            }
        }

        $crate::impl_binop!($name);
        $crate::impl_bounded_float_serde!($name);
    };
}

define_bounded_float!(
    UnitInterval,
    0.0,
    1.0,
    "A floating point value strictly bounded between 0.0 and 1.0 inclusive."
);

impl UnitInterval {
    /// Returns the complementary probability (1.0 - self.value()).
    pub fn complement(&self) -> f64 {
        1.0 - self.0
    }
}

define_strictly_bounded_float!(
    PositiveFloat,
    0.0,
    f64::INFINITY,
    "A floating point value strictly greater than 0.0."
);

define_bounded_float!(
    NonNegativeFloat,
    0.0,
    f64::INFINITY,
    "A floating point value bounded between 0.0 and infinity (inclusive of 0.0)."
);

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::*;
    use serde::Deserialize;
    use serde::de::IntoDeserializer;
    use serde::de::value::{Error as ValError, F64Deserializer};

    fn deser_raw<T: for<'de> Deserialize<'de>>(val: f64) -> Result<T, ValError> {
        let de: F64Deserializer<ValError> = val.into_deserializer();
        T::deserialize(de)
    }

    #[test]
    fn unit_interval_serde_roundtrip_and_rejects_invalid_inputs() {
        for valid in [0.0, 0.25, 0.5, 1.0] {
            let p = UnitInterval::new(valid).unwrap();
            let json = serde_json::to_string(&p).unwrap();
            let restored: UnitInterval = serde_json::from_str(&json).unwrap();
            assert_eq!(restored, p);
        }
        for out_of_range in ["-0.001", "1.001", "-100.0", "42.0"] {
            assert!(serde_json::from_str::<UnitInterval>(out_of_range).is_err());
        }
        for non_finite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.5, 1.5] {
            assert!(deser_raw::<UnitInterval>(non_finite).is_err());
        }
    }

    #[test]
    fn positive_float_serde_roundtrip_and_rejects_invalid_inputs() {
        for valid in [1e-12, 0.5, 42.25] {
            let p = PositiveFloat::new(valid).unwrap();
            let json = serde_json::to_string(&p).unwrap();
            let restored: PositiveFloat = serde_json::from_str(&json).unwrap();
            assert_eq!(restored, p);
        }
        for out_of_range in ["0.0", "-0.0", "-1e-9", "-100.0"] {
            assert!(serde_json::from_str::<PositiveFloat>(out_of_range).is_err());
        }
        for non_finite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
            assert!(deser_raw::<PositiveFloat>(non_finite).is_err());
        }
    }

    #[test]
    fn non_negative_float_serde_roundtrip_and_rejects_invalid_inputs() {
        for valid in [0.0, 1e-12, 100.5] {
            let p = NonNegativeFloat::new(valid).unwrap();
            let json = serde_json::to_string(&p).unwrap();
            let restored: NonNegativeFloat = serde_json::from_str(&json).unwrap();
            assert_eq!(restored, p);
        }
        for out_of_range in ["-1e-9", "-0.5", "-999.0"] {
            assert!(serde_json::from_str::<NonNegativeFloat>(out_of_range).is_err());
        }
        for non_finite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.001] {
            assert!(deser_raw::<NonNegativeFloat>(non_finite).is_err());
        }
    }
}
