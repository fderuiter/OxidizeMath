// [cite:math_commons]
use super::{ArenaError, Diagnostic, DiagnosticVisitor, FitError, MathError, Severity};
#[cfg(feature = "std")]
use super::{EvalError, GlslError, LexError, ParseError, ShaderEmitError};
use thiserror::Error;

/// Extensible crate-level umbrella error for `math_commons` workflows.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MathCommonsError {
    /// Core mathematical validation or dimension failure.
    #[error("{0}")]
    Math(#[from] MathError),
    /// Zero-allocation curve-fitting failure.
    #[error("{0}")]
    Fit(#[from] FitError),
    /// Tier 0 `#![no_std]` byte-arena alignment or capacity failure.
    #[error("{0}")]
    Arena(#[from] ArenaError),
    /// AST lexical analysis failure.
    #[cfg(feature = "std")]
    #[error("{0}")]
    Lex(#[from] LexError),
    /// AST syntax parsing failure.
    #[cfg(feature = "std")]
    #[error("{0}")]
    Parse(#[from] ParseError),
    /// AST runtime evaluation failure.
    #[cfg(feature = "std")]
    #[error("{0}")]
    Eval(#[from] EvalError),
    /// AST-to-GLSL transpilation failure.
    #[cfg(feature = "std")]
    #[error("{0}")]
    Glsl(#[from] GlslError),
    /// AST-to-WGSL/GLSL shader emission failure.
    #[cfg(feature = "std")]
    #[error("{0}")]
    ShaderEmit(#[from] ShaderEmitError),
}

impl MathCommonsError {
    fn as_diagnostic(&self) -> &dyn Diagnostic {
        match self {
            Self::Math(err) => err,
            Self::Fit(err) => err,
            Self::Arena(err) => err,
            #[cfg(feature = "std")]
            Self::Lex(err) => err,
            #[cfg(feature = "std")]
            Self::Parse(err) => err,
            #[cfg(feature = "std")]
            Self::Eval(err) => err,
            #[cfg(feature = "std")]
            Self::Glsl(err) => err,
            #[cfg(feature = "std")]
            Self::ShaderEmit(err) => err,
        }
    }
}

impl Diagnostic for MathCommonsError {
    fn severity(&self) -> Severity {
        self.as_diagnostic().severity()
    }

    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        self.as_diagnostic().visit_fields(visitor);
        visitor.record_str("umbrella_type", "MathCommonsError");
    }
}

impl From<&'static str> for MathCommonsError {
    fn from(reason: &'static str) -> Self {
        Self::Math(MathError::from(reason))
    }
}
