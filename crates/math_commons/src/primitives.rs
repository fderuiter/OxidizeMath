use std::ops::Deref;

#[macro_export]
#[allow(missing_docs)]
macro_rules! impl_binop {
    ($name:ident) => {
        impl std::ops::Mul<f64> for $name {
            type Output = Result<Self, $crate::error::MathError>;
            fn mul(self, rhs: f64) -> Self::Output {
                Self::new(self.0 * rhs)
            }
        }
        impl std::ops::Mul<$name> for f64 {
            type Output = Result<$name, $crate::error::MathError>;
            fn mul(self, rhs: $name) -> Self::Output {
                $name::new(self * rhs.0)
            }
        }
        impl std::ops::Div<f64> for $name {
            type Output = Result<Self, $crate::error::MathError>;
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
            pub fn powf(&self, n: f64) -> Result<Self, $crate::error::MathError> {
                Self::new(self.0.powf(n))
            }

            #[allow(missing_docs)]
            pub fn sqrt(&self) -> Result<Self, $crate::error::MathError> {
                Self::new(self.0.sqrt())
            }

            #[allow(missing_docs)]
            pub fn ln(&self) -> Result<Self, $crate::error::MathError> {
                Self::new(self.0.ln())
            }

            #[allow(missing_docs)]
            pub fn exp(&self) -> Result<Self, $crate::error::MathError> {
                Self::new(self.0.exp())
            }

            #[allow(missing_docs)]
            pub fn abs(&self) -> Result<Self, $crate::error::MathError> {
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
            pub fn powf(&self, n: f64) -> Result<Self, $crate::error::MathError> {
                Self::new(self.0.powf(n))
            }

            #[allow(missing_docs)]
            pub fn sqrt(&self) -> Result<Self, $crate::error::MathError> {
                Self::new(self.0.sqrt())
            }

            #[allow(missing_docs)]
            pub fn ln(&self) -> Result<Self, $crate::error::MathError> {
                Self::new(self.0.ln())
            }

            #[allow(missing_docs)]
            pub fn exp(&self) -> Result<Self, $crate::error::MathError> {
                Self::new(self.0.exp())
            }

            #[allow(missing_docs)]
            pub fn abs(&self) -> Result<Self, $crate::error::MathError> {
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
