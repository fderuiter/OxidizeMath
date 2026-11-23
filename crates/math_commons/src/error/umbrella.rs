// [cite:math_commons]
use super::{EvalError, FitError, GlslError, LexError, MathError, ParseError};
use diagnostics::{Diagnostic, Severity};
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
}

impl MathCommonsError {
    fn as_diagnostic(&self) -> &dyn Diagnostic {
        match self {
            Self::Math(err) => err,
            Self::Fit(err) => err,
            Self::Lex(err) => err,
            Self::Parse(err) => err,
            Self::Eval(err) => err,
            Self::Glsl(err) => err,
        }
    }
}

impl Diagnostic for MathCommonsError {
    fn severity(&self) -> Severity {
        self.as_diagnostic().severity()
    }

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
