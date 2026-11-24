#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:math_scalars]

use core::fmt;

/// Typed `#![no_std]` error returned by fallible scalar, algebraic, modular, and $p$-adic routines.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarError {
    /// Input floating-point value was `NaN` or `±Infinity` where a finite value is required.
    NonFiniteFloat,
    /// Caller requested zero bits of floating-point precision (`prec == 0`).
    ZeroPrecision,
    /// Attempted division or remainder by an exact zero divisor.
    DivisionByZero,
    /// Element is zero or a non-unit and therefore has no multiplicative inverse.
    NonInvertible,
    /// Checked fixed-width integer operation overflowed or underflowed 256-bit bounds.
    ArithmeticOverflow,
    /// Runtime modulus parameter was less than 2 or composite where a prime modulus is required.
    InvalidModulus,
    /// Polynomial or expansion exceeded its compile-time stack capacity `CAP`.
    CapacityExceeded,
    /// Initial residue $r_0$ does not satisfy $f(r_0) \equiv 0 \pmod p$ for Hensel lifting.
    NotARootModP,
    /// Derivative $f'(r_0) \equiv 0 \pmod p$ is singular, violating simple-root Hensel hypotheses.
    SingularDerivative,
    /// Supplied radix digit or string character was out of range.
    InvalidDigit,
}

impl fmt::Display for ScalarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteFloat => write!(f, "non-finite floating-point value (NaN or Infinity)"),
            Self::ZeroPrecision => write!(f, "floating-point bit precision must be at least 1"),
            Self::DivisionByZero => write!(f, "division by zero"),
            Self::NonInvertible => write!(f, "element is not multiplicatively invertible"),
            Self::ArithmeticOverflow => write!(f, "fixed-width integer arithmetic overflow"),
            Self::InvalidModulus => write!(f, "modulus must be a prime integer >= 2"),
            Self::CapacityExceeded => write!(f, "stack polynomial capacity exceeded"),
            Self::NotARootModP => write!(f, "initial value is not a root modulo p"),
            Self::SingularDerivative => {
                write!(f, "singular derivative modulo p prevents Hensel lifting")
            }
            Self::InvalidDigit => write!(f, "invalid digit or numeric literal"),
        }
    }
}

impl core::error::Error for ScalarError {}
