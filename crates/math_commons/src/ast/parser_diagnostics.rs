// [cite:math_commons]
use super::parser::ParseError;

impl diagnostics::Diagnostic for ParseError {
    fn severity(&self) -> diagnostics::Severity {
        diagnostics::Severity::Error
    }

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
