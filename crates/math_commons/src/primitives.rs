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

#[macro_export]
#[allow(missing_docs)]
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

            #[allow(missing_docs)]
            pub fn powf(&self, n: f64) -> ::core::result::Result<Self, $crate::error::MathError> {
                Self::new($crate::primitives::bounded_powf(self.0, n))
            }

            #[allow(missing_docs)]
            pub fn sqrt(&self) -> ::core::result::Result<Self, $crate::error::MathError> {
                Self::new($crate::primitives::bounded_sqrt(self.0))
            }

            #[allow(missing_docs)]
            pub fn ln(&self) -> ::core::result::Result<Self, $crate::error::MathError> {
                Self::new($crate::primitives::bounded_ln(self.0))
            }

            #[allow(missing_docs)]
            pub fn exp(&self) -> ::core::result::Result<Self, $crate::error::MathError> {
                Self::new($crate::primitives::bounded_exp(self.0))
            }

            #[allow(missing_docs)]
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

            #[allow(missing_docs)]
            pub fn powf(&self, n: f64) -> ::core::result::Result<Self, $crate::error::MathError> {
                Self::new($crate::primitives::bounded_powf(self.0, n))
            }

            #[allow(missing_docs)]
            pub fn sqrt(&self) -> ::core::result::Result<Self, $crate::error::MathError> {
                Self::new($crate::primitives::bounded_sqrt(self.0))
            }

            #[allow(missing_docs)]
            pub fn ln(&self) -> ::core::result::Result<Self, $crate::error::MathError> {
                Self::new($crate::primitives::bounded_ln(self.0))
            }

            #[allow(missing_docs)]
            pub fn exp(&self) -> ::core::result::Result<Self, $crate::error::MathError> {
                Self::new($crate::primitives::bounded_exp(self.0))
            }

            #[allow(missing_docs)]
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
    /// Creates a [`UnitInterval`] by clamping finite values to `[0.0, 1.0]`
    /// (mapping non-finite values to `0.0`) without heap allocation.
    #[inline]
    pub fn new_clamped(value: f64) -> Self {
        if value.is_finite() {
            Self(value.clamp(0.0, 1.0))
        } else {
            Self(0.0)
        }
    }

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

#[cfg(test)]
mod tests {
    use super::UnitInterval;

    #[test]
    fn test_unit_interval_new_clamped() {
        assert_eq!(UnitInterval::new_clamped(0.42).value(), 0.42);
        assert_eq!(UnitInterval::new_clamped(-0.5).value(), 0.0);
        assert_eq!(UnitInterval::new_clamped(1.5).value(), 1.0);
        assert_eq!(UnitInterval::new_clamped(f64::NAN).value(), 0.0);
        assert_eq!(UnitInterval::new_clamped(f64::INFINITY).value(), 0.0);
        assert!((UnitInterval::new_clamped(0.25).complement() - 0.75).abs() < 1e-12);
    }
}
