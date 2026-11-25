// [cite:math_commons]
use core::ops::Deref;

/// Internal `#![no_std]` float power helper used by bounded float macros.
#[doc(hidden)]
#[inline]
pub fn bounded_powf(x: f64, n: f64) -> f64 {
    libm::pow(x, n)
}

/// Internal `#![no_std]` square-root helper used by bounded float macros.
#[doc(hidden)]
#[inline]
pub fn bounded_sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

/// Internal `#![no_std]` natural-logarithm helper used by bounded float macros.
#[doc(hidden)]
#[inline]
pub fn bounded_ln(x: f64) -> f64 {
    libm::log(x)
}

/// Internal `#![no_std]` exponential helper used by bounded float macros.
#[doc(hidden)]
#[inline]
pub fn bounded_exp(x: f64) -> f64 {
    libm::exp(x)
}

/// Implements checked scalar multiplication/division or delegating binary operators for wrapper types.
#[macro_export]
macro_rules! impl_binop {
    ($name:ident) => {
        impl ::core::ops::Mul<f64> for $name {
            type Output = ::core::result::Result<Self, $crate::error::MathError>;
            fn mul(self, rhs: f64) -> Self::Output {
                Self::new(self.0 * rhs)
            }
        }
        impl ::core::ops::Mul<$name> for f64 {
            type Output = ::core::result::Result<$name, $crate::error::MathError>;
            fn mul(self, rhs: $name) -> Self::Output {
                $name::new(self * rhs.0)
            }
        }
        impl ::core::ops::Div<f64> for $name {
            type Output = ::core::result::Result<Self, $crate::error::MathError>;
            fn div(self, rhs: f64) -> Self::Output {
                Self::new(self.0 / rhs)
            }
        }
    };
    ($type:ident, $trt:ident, $mth:ident) => {
        impl ::core::ops::$trt for $type {
            type Output = $type;
            fn $mth(self, rhs: $type) -> $type {
                $type(self.0.$mth(rhs.0))
            }
        }
        impl ::core::ops::$trt<&$type> for $type {
            type Output = $type;
            fn $mth(self, rhs: &$type) -> $type {
                $type(self.0.$mth(&rhs.0))
            }
        }
        impl ::core::ops::$trt<$type> for &$type {
            type Output = $type;
            fn $mth(self, rhs: $type) -> $type {
                $type(self.0.clone().$mth(rhs.0))
            }
        }
        impl ::core::ops::$trt<&$type> for &$type {
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
        #[cfg_attr(
            feature = "serde",
            derive(serde::Serialize, serde::Deserialize),
            serde(try_from = "f64", into = "f64")
        )]
        pub struct $name(f64);

        impl $name {
            /// Creates a new instance, validating bounds.
            pub fn new(value: f64) -> Result<Self, $crate::error::MathError> {
                if ($min..=$max).contains(&value) && value.is_finite() {
                    Ok(Self(value))
                } else {
                    Err($crate::error::MathError::OutOfBounds {
                        value,
                        min: $min,
                        max: $max,
                    })
                }
            }

            /// Retrieves the raw value.
            pub fn value(&self) -> f64 {
                self.0
            }

            /// Raises the bounded value to the floating-point power `n`, validating bounds.
            pub fn powf(&self, n: f64) -> ::core::result::Result<Self, $crate::error::MathError> {
                Self::new($crate::primitives::bounded_powf(self.0, n))
            }

            /// Computes the principal square root, validating bounds on the result.
            pub fn sqrt(&self) -> ::core::result::Result<Self, $crate::error::MathError> {
                Self::new($crate::primitives::bounded_sqrt(self.0))
            }

            /// Computes the natural logarithm, validating bounds on the result.
            pub fn ln(&self) -> ::core::result::Result<Self, $crate::error::MathError> {
                Self::new($crate::primitives::bounded_ln(self.0))
            }

            /// Computes the base-$e$ exponential, validating bounds on the result.
            pub fn exp(&self) -> ::core::result::Result<Self, $crate::error::MathError> {
                Self::new($crate::primitives::bounded_exp(self.0))
            }

            /// Computes the absolute value, validating bounds on the result.
            pub fn abs(&self) -> ::core::result::Result<Self, $crate::error::MathError> {
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

        impl ::core::convert::TryFrom<f64> for $name {
            type Error = $crate::error::MathError;

            fn try_from(value: f64) -> ::core::result::Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        $crate::impl_binop!($name);
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
        #[cfg_attr(
            feature = "serde",
            derive(serde::Serialize, serde::Deserialize),
            serde(try_from = "f64", into = "f64")
        )]
        pub struct $name(f64);

        impl $name {
            /// Creates a new instance, validating bounds strictly.
            pub fn new(value: f64) -> Result<Self, $crate::error::MathError> {
                if value > $min && value < $max && value.is_finite() {
                    Ok(Self(value))
                } else {
                    Err($crate::error::MathError::StrictlyOutOfBounds {
                        value,
                        min: $min,
                        max: $max,
                    })
                }
            }

            /// Retrieves the raw value.
            pub fn value(&self) -> f64 {
                self.0
            }

            /// Raises the strictly bounded value to the floating-point power `n`, validating bounds.
            pub fn powf(&self, n: f64) -> ::core::result::Result<Self, $crate::error::MathError> {
                Self::new($crate::primitives::bounded_powf(self.0, n))
            }

            /// Computes the principal square root, validating strict bounds on the result.
            pub fn sqrt(&self) -> ::core::result::Result<Self, $crate::error::MathError> {
                Self::new($crate::primitives::bounded_sqrt(self.0))
            }

            /// Computes the natural logarithm, validating strict bounds on the result.
            pub fn ln(&self) -> ::core::result::Result<Self, $crate::error::MathError> {
                Self::new($crate::primitives::bounded_ln(self.0))
            }

            /// Computes the base-$e$ exponential, validating strict bounds on the result.
            pub fn exp(&self) -> ::core::result::Result<Self, $crate::error::MathError> {
                Self::new($crate::primitives::bounded_exp(self.0))
            }

            /// Computes the absolute value, validating strict bounds on the result.
            pub fn abs(&self) -> ::core::result::Result<Self, $crate::error::MathError> {
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

        impl ::core::convert::TryFrom<f64> for $name {
            type Error = $crate::error::MathError;

            fn try_from(value: f64) -> ::core::result::Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        $crate::impl_binop!($name);
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
