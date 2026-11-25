#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:advanced_linear_algebra]
//! Stack-allocated `#![no_std]` error taxonomy for `math_scalars` arithmetic,
//! precision validation, algebraic inversion, and Hensel root lifting.

use core::fmt;

/// Typed, zero-allocation error returned by fallible scalar, rational, finite-field,
/// polynomial, and $p$-adic operations.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ScalarError {
    /// Input floating-point value was `NaN` or `±Infinity` where a finite real was required.
    NonFiniteFloat,
    /// Requested mantissa bit precision was zero (`prec == 0`).
    ZeroPrecision,
    /// Rational denominator was zero.
    ZeroDenominator,
    /// Division or inversion attempted with an exact or numerical zero divisor.
    DivisionByZero,
    /// Algebraic element is non-invertible (zero divisor, singular norm, or reducible modulus).
    NonInvertible,
    /// Fixed-width integer or exponent arithmetic overflowed representable bounds.
    ArithmeticOverflow,
    /// Polynomial operation exceeded fixed stack capacity `CAP`.
    DegreeOverflow {
        /// Maximum coefficient capacity of the target `StackPoly`.
        capacity: usize,
        /// Minimum coefficient slots required by the result.
        required: usize,
    },
    /// Runtime modulus was `< 2` or composite where a prime modulus was required.
    InvalidModulus {
        /// Rejected modulus value.
        modulus: u64,
    },
    /// Formal derivative $f'(r_0) \equiv 0 \pmod p$ prevented simple Newton-Hensel lifting.
    SingularDerivative,
    /// Candidate residue $r_0$ did not satisfy $f(r_0) \equiv 0 \pmod p$.
    RootNotFound,
    /// String parsing radix was outside `[2, 36]`.
    InvalidRadix {
        /// Rejected radix value.
        radix: u32,
    },
    /// String literal failed integer or rational parsing.
    InvalidLiteral,
}

impl fmt::Display for ScalarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::NonFiniteFloat => {
                f.write_str("non-finite floating-point value (NaN or Infinity)")
            }
            Self::ZeroPrecision => f.write_str("precision must be at least 1 bit"),
            Self::ZeroDenominator => f.write_str("rational denominator must be non-zero"),
            Self::DivisionByZero => f.write_str("division by zero"),
            Self::NonInvertible => f.write_str("element has no multiplicative inverse"),
            Self::ArithmeticOverflow => f.write_str("fixed-width arithmetic overflow"),
            Self::DegreeOverflow { capacity, required } => write!(
                f,
                "polynomial requires {required} coefficient slots, exceeding capacity {capacity}"
            ),
            Self::InvalidModulus { modulus } => {
                write!(f, "modulus {modulus} is invalid or composite")
            }
            Self::SingularDerivative => {
                f.write_str("singular derivative mod p prevents simple Hensel lifting")
            }
            Self::RootNotFound => f.write_str("initial residue is not a root modulo p"),
            Self::InvalidRadix { radix } => {
                write!(f, "invalid radix {radix}; expected 2..=36")
            }
            Self::InvalidLiteral => f.write_str("invalid numeric literal"),
        }
    }
}

impl core::error::Error for ScalarError {}

impl ScalarError {
    #[inline]
    const fn diagnostic_code(&self) -> &'static str {
        match self {
            Self::NonFiniteFloat => "SCALAR-001",
            Self::ZeroPrecision => "SCALAR-002",
            Self::ZeroDenominator => "SCALAR-003",
            Self::DivisionByZero => "SCALAR-004",
            Self::NonInvertible => "SCALAR-005",
            Self::ArithmeticOverflow => "SCALAR-006",
            Self::DegreeOverflow { .. } => "SCALAR-007",
            Self::InvalidModulus { .. } => "SCALAR-008",
            Self::SingularDerivative => "SCALAR-009",
            Self::RootNotFound => "SCALAR-010",
            Self::InvalidRadix { .. } => "SCALAR-011",
            Self::InvalidLiteral => "SCALAR-012",
        }
    }
}

impl math_diagnostics::Diagnostic for ScalarError {
    fn severity(&self) -> math_diagnostics::Severity {
        math_diagnostics::Severity::Error
    }

    fn visit_fields(&self, visitor: &mut dyn math_diagnostics::DiagnosticVisitor) {
        visitor.record_str("error_type", "ScalarError");
        visitor.record_display("description", self);
        visitor.record_str("code", self.diagnostic_code());
        match self {
            Self::DegreeOverflow { capacity, required } => {
                visitor.record_usize("capacity", *capacity);
                visitor.record_usize("required", *required);
            }
            Self::InvalidModulus { modulus } => {
                visitor.record_u64("modulus", *modulus);
            }
            Self::InvalidRadix { radix } => {
                visitor.record_u64("radix", u64::from(*radix));
            }
            _ => {}
        }
    }
}
