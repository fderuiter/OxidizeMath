use crate::applied::{
    algorithms::kalman::KalmanError, clinical_trials::types::ClinicalTrialError,
    isosurface::types::VoxelGridError,
};
use diagnostics::{Diagnostic, DiagnosticVisitor, Severity};

impl Diagnostic for ClinicalTrialError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visitor.record_str("error_type", "ClinicalTrialError");
        visitor.record_display("description", self);
    }
}

impl Diagnostic for KalmanError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        if let Self::Math(math_err) = self {
            math_err.visit_fields(visitor);
        }
        visitor.record_str("error_type", "KalmanError");
        visitor.record_display("description", self);
    }
}

impl Diagnostic for VoxelGridError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visitor.record_str("error_type", "VoxelGridError");
        visitor.record_display("description", self);
    }
}
