// [cite:math_commons]
use super::{
    EvalError, FitError, GlslError, LexError, LinalgError, MathError, ParseError, QmcError,
    ShaderEmitError,
};
use alloc::string::String;
#[cfg(feature = "std")]
use alloc::string::ToString;
use diagnostics::{Diagnostic, Severity};
#[cfg(feature = "std")]
use std::collections::HashMap;
use thiserror::Error;

/// Extensible crate-level umbrella error for `math_commons` workflows.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Error)]
pub enum MathCommonsError {
    /// Core mathematical validation or dimension failure.
    #[error("{0}")]
    Math(#[from] MathError),
    /// Zero-allocation curve-fitting failure.
    #[error("{0}")]
    Fit(#[from] FitError),
    /// `#![no_std]` dense or sparse numerical linear-algebra engine failure.
    #[error("{0}")]
    Linalg(#[from] LinalgError),
    /// `#![no_std]` Quasi-Monte Carlo generator or view failure.
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

    fn error_type(&self) -> &'static str {
        self.as_diagnostic().error_type()
    }

    fn code(&self) -> Option<&'static str> {
        self.as_diagnostic().code()
    }

    #[cfg(feature = "std")]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = self.as_diagnostic().metadata();
        map.insert("umbrella_type".to_string(), "MathCommonsError".to_string());
        map
    }
}

impl From<String> for MathCommonsError {
    fn from(reason: String) -> Self {
        Self::Math(MathError::from(reason))
    }
}
