// [cite:math_commons]
use super::shader::ShaderEmitError;

impl math_diagnostics::Diagnostic for ShaderEmitError {
    fn severity(&self) -> math_diagnostics::Severity {
        math_diagnostics::Severity::Error
    }

    fn visit_fields(&self, visitor: &mut dyn math_diagnostics::DiagnosticVisitor) {
        visitor.record_str("error_type", "ShaderEmitError");
        visitor.record_display("description", self);
        match self {
            Self::RecursionLimitExceeded(max_depth) => {
                visitor.record_usize("max_depth", *max_depth);
            }
            Self::NodeLimitExceeded(max_nodes) => {
                visitor.record_usize("max_nodes", *max_nodes);
            }
            Self::OutputLimitExceeded(max_bytes) => {
                visitor.record_usize("max_output_bytes", *max_bytes);
            }
            Self::WrongArgumentCount {
                func,
                expected,
                found,
            } => {
                visitor.record_str("function", func.as_str());
                visitor.record_usize("expected_args", *expected);
                visitor.record_usize("found_args", *found);
            }
            Self::NonFiniteLiteral => {
                visitor.record_str("reason", "non_finite_literal");
            }
            Self::InvalidIdentifier(ident) => {
                visitor.record_str("identifier", ident);
            }
        }
    }
}
