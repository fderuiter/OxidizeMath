// [cite:math_commons]
use core::fmt;
use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};

/// Stack-allocated `#![no_std]` error classification for runtime-configured scalar domains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ScalarDomainError {
    /// Runtime modulus was less than `2` and cannot form a nontrivial residue ring or field.
    InvalidModulus {
        /// Rejected modulus value.
        modulus: u64,
    },
    /// Attempted to invert or divide by the additive identity (`0`).
    DivisionByZero,
    /// Residue is nonzero but shares a nontrivial common factor with `modulus`.
    NonInvertibleResidue {
        /// Non-invertible residue representative.
        residue: u64,
        /// Active ring/field modulus.
        modulus: u64,
    },
    /// Requested dynamic floating-point precision bits was zero.
    InvalidPrecision {
        /// Rejected precision bit count.
        precision_bits: usize,
    },
}

impl ScalarDomainError {
    /// Returns the static type identifier `"ScalarDomainError"`.
    #[inline]
    pub const fn error_type(&self) -> &'static str {
        "ScalarDomainError"
    }

    /// Returns the stable diagnostic error code (`SCALAR-001`..`SCALAR-004`).
    #[inline]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InvalidModulus { .. } => "SCALAR-001",
            Self::DivisionByZero => "SCALAR-002",
            Self::NonInvertibleResidue { .. } => "SCALAR-003",
            Self::InvalidPrecision { .. } => "SCALAR-004",
        }
    }
}

impl fmt::Display for ScalarDomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModulus { modulus } => {
                write!(f, "Invalid scalar domain modulus {modulus}: expected >= 2")
            }
            Self::DivisionByZero => write!(f, "Division by zero in algebraic scalar domain"),
            Self::NonInvertibleResidue { residue, modulus } => write!(
                f,
                "Residue {residue} has no multiplicative inverse modulo {modulus}"
            ),
            Self::InvalidPrecision { precision_bits } => write!(
                f,
                "Invalid floating-point precision {precision_bits}: expected >= 1 bit"
            ),
        }
    }
}

impl core::error::Error for ScalarDomainError {}

impl Diagnostic for ScalarDomainError {
    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visitor.record_str("error_type", self.error_type());
        visitor.record_str("code", self.code());
        visitor.record_display("description", self);
        match self {
            Self::InvalidModulus { modulus } => {
                visitor.record_u64("modulus", *modulus);
            }
            Self::DivisionByZero => {}
            Self::NonInvertibleResidue { residue, modulus } => {
                visitor.record_u64("residue", *residue);
                visitor.record_u64("modulus", *modulus);
            }
            Self::InvalidPrecision { precision_bits } => {
                visitor.record_usize("precision_bits", *precision_bits);
            }
        }
    }
}
