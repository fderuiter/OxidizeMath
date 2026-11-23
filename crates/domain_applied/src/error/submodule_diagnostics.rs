use crate::applied::{
    algorithms::kalman::KalmanError, clinical_trials::types::ClinicalTrialError,
    isosurface::manifold::WatertightGridError, isosurface::types::VoxelGridError,
};
use diagnostics::{Diagnostic, Severity};
use std::collections::HashMap;

#[verified_engine::verified]
fn error_metadata(error_type: &str, description: String) -> HashMap<String, String> {
    let mut map = HashMap::new();
    map.insert("error_type".to_string(), error_type.to_string());
    map.insert("description".to_string(), description);
    map
}

impl Diagnostic for ClinicalTrialError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        error_metadata("ClinicalTrialError", self.to_string())
    }
}

impl Diagnostic for KalmanError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = error_metadata("KalmanError", self.to_string());
        if let Self::Math(math_err) = self {
            for (k, v) in math_err.metadata() {
                map.entry(k).or_insert(v);
            }
        }
        map
    }
}

impl Diagnostic for VoxelGridError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        error_metadata("VoxelGridError", self.to_string())
    }
}

impl Diagnostic for WatertightGridError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        error_metadata("WatertightGridError", self.to_string())
    }
}
