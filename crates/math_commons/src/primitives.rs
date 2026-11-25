//! Bounded real scalar newtypes (`UnitInterval`, `PositiveFloat`, `NonNegativeFloat`) with
//! explicit scalar extraction and no `Deref` or domain-escaping transcendental methods.
// [cite:math_commons]

/// Internal `#![no_std]` square-root helper used by bounded float macros.
#[doc(hidden)]
#[inline]
pub fn bounded_sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

/// Internal binary-operator helper macro for bounded float primitives.
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

/// Defines a closed-interval bounded float primitive `[min, max]` without `Deref` or
/// domain-escaping `.ln()`, `.exp()`, or `.powf()` methods.
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
            /// Creates a new instance after validating closed bounds `[min, max]`.
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

            /// Returns the wrapped scalar value by copy.
            #[inline(always)]
            pub const fn get(self) -> f64 {
                self.0
            }

            /// Retrieves the raw scalar value.
            #[inline(always)]
            pub const fn value(&self) -> f64 {
                self.0
            }

            /// Evaluates the principal square root and validates the bounded interval.
            pub fn sqrt(&self) -> ::core::result::Result<Self, $crate::error::MathError> {
                Self::new($crate::primitives::bounded_sqrt(self.0))
            }

            /// Evaluates the absolute value and validates the bounded interval.
            pub fn abs(&self) -> ::core::result::Result<Self, $crate::error::MathError> {
                Self::new(self.0.abs())
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

/// Defines an open-interval strictly bounded float primitive `(min, max)` without `Deref` or
/// domain-escaping `.ln()`, `.exp()`, or `.powf()` methods.
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
            /// Creates a new instance after validating strict open bounds `(min, max)`.
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

            /// Returns the wrapped scalar value by copy.
            #[inline(always)]
            pub const fn get(self) -> f64 {
                self.0
            }

            /// Retrieves the raw scalar value.
            #[inline(always)]
            pub const fn value(&self) -> f64 {
                self.0
            }

            /// Evaluates the principal square root and validates strict bounds.
            pub fn sqrt(&self) -> ::core::result::Result<Self, $crate::error::MathError> {
                Self::new($crate::primitives::bounded_sqrt(self.0))
            }

            /// Evaluates the absolute value and validates strict bounds.
            pub fn abs(&self) -> ::core::result::Result<Self, $crate::error::MathError> {
                Self::new(self.0.abs())
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
    "A floating-point value bounded on `[0.0, 1.0]` requiring explicit `.get()` or `.value()` extraction.\n\n```compile_fail\nuse math_commons::primitives::UnitInterval;\nlet p = UnitInterval::new(0.5).unwrap();\nlet _raw: f64 = *p;\n```\n\n```compile_fail\nuse math_commons::primitives::UnitInterval;\nlet p = UnitInterval::new(0.5).unwrap();\nlet _ = p.ln();\n```"
);

impl UnitInterval {
    /// Returns the complementary probability `1.0 - self.value()`.
    #[inline(always)]
    pub fn complement(&self) -> f64 {
        1.0 - self.0
    }
}

define_strictly_bounded_float!(
    PositiveFloat,
    0.0,
    f64::INFINITY,
    "A floating-point value strictly greater than `0.0` requiring explicit `.get()` or `.value()` extraction."
);

define_bounded_float!(
    NonNegativeFloat,
    0.0,
    f64::INFINITY,
    "A floating-point value in `[0.0, +inf)` requiring explicit `.get()` or `.value()` extraction."
);
