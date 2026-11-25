// [cite:math_commons]
use super::{
    Diagnostic, DiagnosticVisitor, EvalError, FitError, GlslError, LexError, LinalgError,
    MathError, ParseError, QmcError, Severity, ShaderEmitError,
};
use alloc::string::String;
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
    /// `#![no_std]` linear algebra engine failure.
    #[error("{0}")]
    Linalg(#[from] LinalgError),
    /// `#![no_std]` Quasi-Monte Carlo sequence or view failure.
    #[error("{0}")]
    Qmc(#[from] QmcError),
    /// AST lexical analysis failure.
    #[error("{0}")]
    Lex(#[from] LexError),
    /// AST syntax parsing failure.
    #[error("{0}")]
    Parse(#[from] ParseError),
    /// AST runtime evaluation failure.
    #[error("{0}")]
    Eval(#[from] EvalError),
    /// AST-to-GLSL transpilation failure.
    #[error("{0}")]
    Glsl(#[from] GlslError),
    /// AST-to-WGSL/GLSL shader emission failure.
    #[error("{0}")]
    ShaderEmit(#[from] ShaderEmitError),
}

impl MathCommonsError {
    fn as_diagnostic(&self) -> &dyn Diagnostic {
        match self {
            Self::Math(err) => err,
            Self::Fit(err) => err,
            Self::Linalg(err) => err,
            Self::Qmc(err) => err,
            Self::Lex(err) => err,
            Self::Parse(err) => err,
            Self::Eval(err) => err,
            Self::Glsl(err) => err,
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

impl From<String> for MathCommonsError {
    fn from(reason: String) -> Self {
        Self::Math(MathError::from(reason))
    }
}
