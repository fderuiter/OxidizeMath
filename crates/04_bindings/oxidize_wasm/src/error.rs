//! Structured diagnostic error conversion from OxidizeMath engines into typed JS `Error` exceptions.
// [cite:engine_linalg]
// [cite:engine_diffeq]
// [cite:engine_pde]
// [cite:math_commons]

use engine_diffeq::DiffeqError;
use engine_linalg::{ArenaError, LinalgError};
use engine_pde::PdeError;
use math_commons::ast::{EvalError, ParseError, ShaderEmitError};
use math_commons::error::MathError;
use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};
use math_scalars::ScalarError;
use std::fmt;

/// Key-value metadata pair harvested from a [`Diagnostic`] implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticField {
    /// Structured field name (for example `"error_type"`, `"required"`, or `"available"`).
    pub key: String,
    /// Formatted value string.
    pub value: String,
}

#[derive(Default)]
struct FieldCollector {
    fields: Vec<DiagnosticField>,
}

impl DiagnosticVisitor for FieldCollector {
    fn record_str(&mut self, key: &str, val: &str) {
        self.fields.push(DiagnosticField {
            key: key.to_string(),
            value: val.to_string(),
        });
    }

    fn record_display(&mut self, key: &str, val: &dyn fmt::Display) {
        self.fields.push(DiagnosticField {
            key: key.to_string(),
            value: val.to_string(),
        });
    }
}

/// Typed SDK error carrying structured [`Diagnostic`] metadata across Rust and `wasm-bindgen` boundaries.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("[{code}] {error_type}: {message}")]
pub struct WasmSdkError {
    /// Stable machine-readable diagnostic code (e.g. `"E_LINALG_SINGULAR"`).
    pub code: &'static str,
    /// Engine error domain name (e.g. `"LinalgError"`, `"PdeError"`, `"WasmArgumentError"`).
    pub error_type: String,
    /// Diagnostic severity level.
    pub severity: Severity,
    /// Human-readable diagnostic description.
    pub message: String,
    /// Structured key-value diagnostic fields harvested via [`Diagnostic::visit_fields`].
    pub fields: Vec<DiagnosticField>,
}

/// Alias for fallible `oxidize_wasm` operations returning [`WasmSdkError`].
pub type WasmSdkResult<T> = Result<T, WasmSdkError>;

impl WasmSdkError {
    /// Constructs a [`WasmSdkError`] from any [`Diagnostic`] + [`fmt::Display`] engine error.
    #[must_use]
    pub fn from_diagnostic<D: Diagnostic + fmt::Display>(
        code: &'static str,
        fallback_type: &'static str,
        diag: &D,
    ) -> Self {
        let mut collector = FieldCollector::default();
        diag.visit_fields(&mut collector);
        let error_type = collector
            .fields
            .iter()
            .find(|f| f.key == "error_type")
            .map_or_else(|| fallback_type.to_string(), |f| f.value.clone());
        Self {
            code,
            error_type,
            severity: diag.severity(),
            message: diag.to_string(),
            fields: collector.fields,
        }
    }

    /// Constructs a dimension or capacity mismatch error at the WASM SDK boundary.
    #[must_use]
    pub fn invalid_argument(
        code: &'static str,
        message: impl Into<String>,
        expected: usize,
        actual: usize,
    ) -> Self {
        let msg = message.into();
        Self {
            code,
            error_type: "WasmArgumentError".to_string(),
            severity: Severity::Error,
            message: msg.clone(),
            fields: vec![
                DiagnosticField {
                    key: "error_type".to_string(),
                    value: "WasmArgumentError".to_string(),
                },
                DiagnosticField {
                    key: "description".to_string(),
                    value: msg,
                },
                DiagnosticField {
                    key: "expected".to_string(),
                    value: expected.to_string(),
                },
                DiagnosticField {
                    key: "actual".to_string(),
                    value: actual.to_string(),
                },
            ],
        }
    }

    /// Looks up a harvested diagnostic field by key.
    #[must_use]
    pub fn field_value(&self, key: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|f| f.key == key)
            .map(|f| f.value.as_str())
    }
}

impl Diagnostic for WasmSdkError {
    fn severity(&self) -> Severity {
        self.severity
    }

    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visitor.record_str("code", self.code);
        visitor.record_str("error_type", &self.error_type);
        visitor.record_str("description", &self.message);
        for field in &self.fields {
            if field.key != "code" && field.key != "error_type" && field.key != "description" {
                visitor.record_str(&field.key, &field.value);
            }
        }
    }
}

impl From<LinalgError> for WasmSdkError {
    fn from(err: LinalgError) -> Self {
        let code = match err {
            LinalgError::SingularMatrix => "E_LINALG_SINGULAR",
            LinalgError::NotPositiveDefinite => "E_LINALG_NOT_SPD",
            LinalgError::DimensionMismatch => "E_LINALG_DIM_MISMATCH",
            LinalgError::WorkspaceTooSmall { .. } => "E_LINALG_WORKSPACE",
            LinalgError::NonConvergence => "E_LINALG_NON_CONVERGENCE",
            LinalgError::InvalidInput => "E_LINALG_INVALID_INPUT",
        };
        Self::from_diagnostic(code, "LinalgError", &err)
    }
}

impl From<ScalarError> for WasmSdkError {
    fn from(err: ScalarError) -> Self {
        Self::from_diagnostic("E_SCALAR_BOUNDS", "ScalarError", &err)
    }
}

impl From<DiffeqError> for WasmSdkError {
    fn from(err: DiffeqError) -> Self {
        Self::from_diagnostic("E_DIFFEQ", "DiffeqError", &err)
    }
}

impl From<PdeError> for WasmSdkError {
    fn from(err: PdeError) -> Self {
        Self::from_diagnostic("E_PDE", "PdeError", &err)
    }
}

impl From<MathError> for WasmSdkError {
    fn from(err: MathError) -> Self {
        Self::from_diagnostic("E_MATH_BOUNDS", "MathError", &err)
    }
}

impl From<ParseError> for WasmSdkError {
    fn from(err: ParseError) -> Self {
        Self::from_diagnostic("E_AST_PARSE", "ParseError", &err)
    }
}

impl From<EvalError> for WasmSdkError {
    fn from(err: EvalError) -> Self {
        Self::from_diagnostic("E_AST_EVAL", "EvalError", &err)
    }
}

impl From<ShaderEmitError> for WasmSdkError {
    fn from(err: ShaderEmitError) -> Self {
        Self::from_diagnostic("E_SHADER_EMIT", "ShaderEmitError", &err)
    }
}

impl From<ArenaError> for WasmSdkError {
    fn from(err: ArenaError) -> Self {
        Self::from_diagnostic("E_ARENA_WORKSPACE", "ArenaError", &err)
    }
}

impl From<WasmSdkError> for wasm_bindgen::JsValue {
    fn from(err: WasmSdkError) -> Self {
        crate::error_to_js_value(&err)
    }
}
