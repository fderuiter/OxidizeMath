// [cite:math_commons]
use super::parser::ParseError;
use alloc::{
    format,
    string::{String, ToString},
};

pub(super) fn format_found_token(found: &str) -> String {
    if found == "<EOF>" || found == "end of expression" {
        "end of expression".to_string()
    } else {
        format!("'{found}'")
    }
}

pub(super) fn format_user_message_for_input(err: &ParseError, input: &str) -> String {
    if matches!(err, ParseError::UnexpectedEof) {
        let trimmed_chars = input.trim().chars().count();
        if trimmed_chars > 0 {
            return format!(
                "Unexpected end of expression at character {}",
                trimmed_chars + 1
            );
        }
    }
    err.user_message()
}

impl diagnostics::Diagnostic for ParseError {
    fn severity(&self) -> diagnostics::Severity {
        diagnostics::Severity::Error
    }

    fn error_type(&self) -> &'static str {
        "ParseError"
    }

    #[cfg(feature = "std")]
    fn metadata(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        if let Self::Lex(lex_err) = self {
            map.extend(lex_err.metadata());
            map.insert("cause_error_type".to_string(), "LexError".to_string());
        }
        map.insert("error_type".to_string(), "ParseError".to_string());
        map.insert("description".to_string(), self.to_string());
        match self {
            Self::UnexpectedToken {
                expected,
                found,
                span,
            } => {
                map.insert("expected".to_string(), expected.clone());
                map.insert("found".to_string(), found.clone());
                map.insert("start".to_string(), span.start.to_string());
            }
            Self::RecursionLimitExceeded(max_depth) => {
                map.insert("max_depth".to_string(), max_depth.to_string());
            }
            Self::NodeLimitExceeded(max_nodes) => {
                map.insert("max_nodes".to_string(), max_nodes.to_string());
            }
            Self::UnknownFunction { name, span } => {
                map.insert("function".to_string(), name.clone());
                map.insert("start".to_string(), span.start.to_string());
            }
            Self::WrongArgumentCount {
                func,
                expected,
                found,
                span,
            } => {
                map.insert("function".to_string(), func.as_str().to_string());
                map.insert("expected".to_string(), expected.to_string());
                map.insert("found".to_string(), found.to_string());
                map.insert("start".to_string(), span.start.to_string());
            }
            Self::Lex(_) | Self::UnexpectedEof | Self::EmptyExpression => {}
        }
        map
    }
}
