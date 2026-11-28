use super::{
    CopulaError, EloError, Glicko2Error, KellyError, MarketPricingError, MarkovError,
    MatchModelError, NumberTheoryError, OuError, TdaError, ZipError,
};
use crate::pure_math::{
    algebra::{BraidError, linear_algebra::numerical::LinearAlgebraError},
    analysis::{
        evolution::EvolutionError, evolutionary::GameTheoryError, kalman::KalmanError,
        optimization::OptimizationError, roots::AnalysisError, stochastic::StochasticError,
        transforms::TransformError,
    },
    elliptic_curves::EllipticCurveError,
    graph_theory::{GraphTheoryError, wardrop::WardropError},
    hyperbolic::HyperbolicError,
    knot_theory::KnotError,
    number_theory::ambs::AmbsError,
    statistics::{
        black_scholes::OptionPricingError, monte_carlo::MonteCarloError, poisson::PoissonError,
        queueing::QueueingError,
    },
    tensor::types::TensorError,
};
use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};

#[verified_engine::verified]
fn visit_named_error<D: core::fmt::Display>(
    visitor: &mut dyn DiagnosticVisitor,
    error_type: &'static str,
    err: &D,
) {
    visitor.record_str("error_type", error_type);
    visitor.record_display("description", err);
}

impl Diagnostic for NumberTheoryError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "NumberTheoryError", self);
    }
}

impl Diagnostic for KellyError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "KellyError", self);
    }
}

impl Diagnostic for MarketPricingError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "MarketPricingError", self);
    }
}

impl Diagnostic for MarkovError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        if let Self::Math(math_err) = self {
            math_err.visit_fields(visitor);
        }
        visit_named_error(visitor, "MarkovError", self);
    }
}

impl Diagnostic for OuError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "OuError", self);
    }
}

impl Diagnostic for ZipError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "ZipError", self);
    }
}

impl Diagnostic for TdaError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "TdaError", self);
    }
}

impl Diagnostic for CopulaError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        if let Self::Math(math_err) = self {
            math_err.visit_fields(visitor);
        }
        visit_named_error(visitor, "CopulaError", self);
    }
}

impl Diagnostic for Glicko2Error {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "Glicko2Error", self);
    }
}

impl Diagnostic for EloError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "EloError", self);
    }
}

impl Diagnostic for MatchModelError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "MatchModelError", self);
    }
}

impl Diagnostic for AmbsError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "AmbsError", self);
    }
}

impl Diagnostic for AnalysisError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "AnalysisError", self);
    }
}

impl Diagnostic for EllipticCurveError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "EllipticCurveError", self);
    }
}

impl Diagnostic for GraphTheoryError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "GraphTheoryError", self);
    }
}

impl Diagnostic for WardropError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "WardropError", self);
    }
}

impl Diagnostic for EvolutionError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "EvolutionError", self);
    }
}

impl Diagnostic for LinearAlgebraError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        if let Self::Linalg(linalg_err) = self {
            linalg_err.visit_fields(visitor);
        }
        visit_named_error(visitor, "LinearAlgebraError", self);
    }
}

impl Diagnostic for OptimizationError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "OptimizationError", self);
    }
}

impl Diagnostic for StochasticError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "StochasticError", self);
    }
}

impl Diagnostic for TensorError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        if let Self::Math(math_err) = self {
            math_err.visit_fields(visitor);
        }
        visit_named_error(visitor, "TensorError", self);
    }
}

impl Diagnostic for TransformError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "TransformError", self);
    }
}

impl Diagnostic for BraidError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "BraidError", self);
    }
}

impl Diagnostic for HyperbolicError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "HyperbolicError", self);
    }
}

impl Diagnostic for KnotError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "KnotError", self);
    }
}

impl Diagnostic for MonteCarloError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "MonteCarloError", self);
    }
}

impl Diagnostic for OptionPricingError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "OptionPricingError", self);
    }
}

impl Diagnostic for PoissonError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "PoissonError", self);
    }
}

impl Diagnostic for QueueingError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visit_named_error(visitor, "QueueingError", self);
    }
}

impl Diagnostic for GameTheoryError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        match self {
            Self::Math(err) => err.severity(),
            Self::MonteCarlo(err) => err.severity(),
            _ => Severity::Error,
        }
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        match self {
            Self::Math(math_err) => math_err.visit_fields(visitor),
            Self::MonteCarlo(mc_err) => mc_err.visit_fields(visitor),
            _ => {}
        }
        visit_named_error(visitor, "GameTheoryError", self);
    }
}

impl Diagnostic for KalmanError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        match self {
            Self::Math(err) => err.severity(),
            _ => Severity::Error,
        }
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        if let Self::Math(math_err) = self {
            math_err.visit_fields(visitor);
        }
        visit_named_error(visitor, "KalmanError", self);
    }
}
