use super::{
    CopulaError, Glicko2Error, KellyError, MarketPricingError, MarkovError, NumberTheoryError,
    OuError, TdaError, ZipError,
};
use crate::pure_math::{
    algebra::linear_algebra::numerical::LinearAlgebraError,
    analysis::{
        evolution::EvolutionError, optimization::OptimizationError, roots::AnalysisError,
        stochastic::StochasticError, transforms::TransformError,
    },
    elliptic_curves::EllipticCurveError,
    number_theory::ambs::AmbsError,
    tensor::types::TensorError,
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

impl Diagnostic for NumberTheoryError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        error_metadata("NumberTheoryError", self.to_string())
    }
}

impl Diagnostic for KellyError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        error_metadata("KellyError", self.to_string())
    }
}

impl Diagnostic for MarketPricingError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        error_metadata("MarketPricingError", self.to_string())
    }
}

impl Diagnostic for MarkovError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        error_metadata("MarkovError", self.to_string())
    }
}

impl Diagnostic for OuError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        error_metadata("OuError", self.to_string())
    }
}

impl Diagnostic for ZipError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        error_metadata("ZipError", self.to_string())
    }
}

impl Diagnostic for TdaError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        error_metadata("TdaError", self.to_string())
    }
}

impl Diagnostic for CopulaError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        error_metadata("CopulaError", self.to_string())
    }
}

impl Diagnostic for Glicko2Error {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        error_metadata("Glicko2Error", self.to_string())
    }
}

impl Diagnostic for AmbsError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        error_metadata("AmbsError", self.to_string())
    }
}

impl Diagnostic for AnalysisError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        error_metadata("AnalysisError", self.to_string())
    }
}

impl Diagnostic for EllipticCurveError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        error_metadata("EllipticCurveError", self.to_string())
    }
}

impl Diagnostic for EvolutionError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        error_metadata("EvolutionError", self.to_string())
    }
}

impl Diagnostic for LinearAlgebraError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        error_metadata("LinearAlgebraError", self.to_string())
    }
}

impl Diagnostic for OptimizationError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        error_metadata("OptimizationError", self.to_string())
    }
}

impl Diagnostic for StochasticError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        error_metadata("StochasticError", self.to_string())
    }
}

impl Diagnostic for TensorError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        error_metadata("TensorError", self.to_string())
    }
}

impl Diagnostic for TransformError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        error_metadata("TransformError", self.to_string())
    }
}
