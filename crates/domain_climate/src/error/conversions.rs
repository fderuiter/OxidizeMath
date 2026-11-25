use super::ClimateError;
use diagnostics::{Diagnostic, DiagnosticVisitor, Severity};
use math_commons::error::{
    EvalError, GlslError, LexError, MathCommonsError, MathError, ParseError, QmcError, RngError,
    ShaderEmitError,
};

macro_rules! impl_error_from {
    ($($err_ty:ident => $ctor:expr),+ $(,)?) => {
        $(
            impl From<$err_ty> for ClimateError {
                #[verified_engine::verified]
                fn from(err: $err_ty) -> Self {
                    ($ctor)(err)
                }
            }
        )+
    };
}

impl_error_from!(
    QmcError => |e| Self::MathCommons(MathCommonsError::from(e)),
    RngError => |e| Self::MathCommons(MathCommonsError::from(e)),
    EvalError => |e| Self::MathCommons(MathCommonsError::from(e)),
    GlslError => |e| Self::MathCommons(MathCommonsError::from(e)),
    LexError => |e| Self::MathCommons(MathCommonsError::from(e)),
    ParseError => |e| Self::MathCommons(MathCommonsError::from(e)),
    ShaderEmitError => |e| Self::MathCommons(MathCommonsError::from(e)),
);

impl From<&'static str> for ClimateError {
    #[verified_engine::verified]
    fn from(reason: &'static str) -> Self {
        Self::Math(MathError::from(reason))
    }
}

impl Diagnostic for ClimateError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        match self {
            Self::Math(err) => err.severity(),
            Self::MathCommons(err) => err.severity(),
            Self::Linalg(err) => err.severity(),
            Self::Fit(err) => err.severity(),
            Self::Optimization(err) => err.severity(),
            Self::LinearAlgebra(err) => err.severity(),
            _ => Severity::Error,
        }
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        match self {
            Self::Math(err) => err.visit_fields(visitor),
            Self::MathCommons(err) => err.visit_fields(visitor),
            Self::Linalg(err) => err.visit_fields(visitor),
            Self::Fit(err) => err.visit_fields(visitor),
            Self::Optimization(err) => err.visit_fields(visitor),
            Self::LinearAlgebra(err) => err.visit_fields(visitor),
            _ => {
                visitor.record_str("error_type", "ClimateError");
                visitor.record_display("description", self);
            }
        }
        visitor.record_str("umbrella_type", "ClimateError");
    }
}
