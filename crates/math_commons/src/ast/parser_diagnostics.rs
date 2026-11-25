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

impl math_diagnostics::Diagnostic for ParseError {
    fn severity(&self) -> math_diagnostics::Severity {
        math_diagnostics::Severity::Error
    }

    fn visit_fields(&self, visitor: &mut dyn math_diagnostics::DiagnosticVisitor) {
        if let Self::Lex(lex_err) = self {
            lex_err.visit_fields(visitor);
            visitor.record_str("cause_error_type", "LexError");
        }
        visitor.record_str("error_type", "ParseError");
        visitor.record_display("description", self);
        match self {
            Self::UnexpectedToken {
                expected,
                found,
                span,
            } => {
                visitor.record_str("expected", expected);
                visitor.record_str("found", found);
                visitor.record_usize("start", span.start);
            }
            Self::RecursionLimitExceeded(max_depth) => {
                visitor.record_usize("max_depth", *max_depth);
            }
            Self::NodeLimitExceeded(max_nodes) => {
                visitor.record_usize("max_nodes", *max_nodes);
            }
            Self::UnknownFunction { name, span } => {
                visitor.record_str("function", name);
                visitor.record_usize("start", span.start);
            }
            Self::WrongArgumentCount {
                func,
                expected,
                found,
                span,
            } => {
                visitor.record_str("function", func.as_str());
                visitor.record_usize("expected", *expected);
                visitor.record_usize("found", *found);
                visitor.record_usize("start", span.start);
            }
            Self::Lex(_) | Self::UnexpectedEof | Self::EmptyExpression => {}
        }
    }
}
