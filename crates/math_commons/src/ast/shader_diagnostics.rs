// [cite:math_commons]
use super::shader::ShaderEmitError;
#[cfg(feature = "std")]
use alloc::string::ToString;

impl diagnostics::Diagnostic for ShaderEmitError {
    fn severity(&self) -> diagnostics::Severity {
        diagnostics::Severity::Error
    }

    fn error_type(&self) -> &'static str {
        "ShaderEmitError"
    }

    #[cfg(feature = "std")]
    fn metadata(&self) -> std::collections::HashMap<alloc::string::String, alloc::string::String> {
        let mut map = std::collections::HashMap::from([
            ("error_type".to_string(), "ShaderEmitError".to_string()),
            ("description".to_string(), self.to_string()),
        ]);
        match self {
            Self::RecursionLimitExceeded(max_depth) => {
                map.insert("max_depth".to_string(), max_depth.to_string());
            }
            Self::NodeLimitExceeded(max_nodes) => {
                map.insert("max_nodes".to_string(), max_nodes.to_string());
            }
            Self::OutputLimitExceeded(max_bytes) => {
                map.insert("max_output_bytes".to_string(), max_bytes.to_string());
            }
            Self::WrongArgumentCount {
                func,
                expected,
                found,
            } => {
                map.insert("function".to_string(), func.as_str().to_string());
                map.insert("expected_args".to_string(), expected.to_string());
                map.insert("found_args".to_string(), found.to_string());
            }
            Self::NonFiniteLiteral => {
                map.insert("reason".to_string(), "non_finite_literal".to_string());
            }
            Self::InvalidIdentifier(ident) => {
                map.insert("identifier".to_string(), ident.clone());
            }
        }
        map
    }
}
