use super::Error;
use diagnostics::{Diagnostic, Severity};
use std::collections::HashMap;

trait FacadeInnerError: std::error::Error + Diagnostic + 'static {
    fn as_std_error(&self) -> &(dyn std::error::Error + 'static);
    fn as_diagnostic(&self) -> &dyn Diagnostic;
}

impl<T: std::error::Error + Diagnostic + 'static> FacadeInnerError for T {
    #[verified_engine::verified]
    fn as_std_error(&self) -> &(dyn std::error::Error + 'static) {
        self
    }

    #[verified_engine::verified]
    fn as_diagnostic(&self) -> &dyn Diagnostic {
        self
    }
}

enum InnerErrorRef<'a> {
    Domain(&'a dyn FacadeInnerError),
    Io(&'a std::io::Error),
}

impl Error {
    #[cfg(feature = "pure_math")]
    #[verified_engine::verified]
    fn pure_math_inner(&self) -> Option<&dyn FacadeInnerError> {
        match self {
            Self::PureMath(err) => Some(err),
            Self::Ambs(err) => Some(err),
            Self::Analysis(err) => Some(err),
            Self::Copula(err) => Some(err),
            Self::EllipticCurve(err) => Some(err),
            Self::Evolution(err) => Some(err),
            Self::Fractran(err) => Some(err),
            Self::Glicko2(err) => Some(err),
            Self::Kelly(err) => Some(err),
            Self::LinearAlgebra(err) => Some(err),
            Self::Markov(err) => Some(err),
            Self::MonteCarlo(err) => Some(err),
            Self::NumberTheory(err) => Some(err),
            Self::Optimization(err) => Some(err),
            Self::Ou(err) => Some(err),
            Self::Stochastic(err) => Some(err),
            Self::Tda(err) => Some(err),
            Self::Tensor(err) => Some(err),
            Self::Transform(err) => Some(err),
            Self::Zip(err) => Some(err),
            _ => None,
        }
    }

    #[cfg(feature = "applied")]
    #[verified_engine::verified]
    fn applied_inner(&self) -> Option<&dyn FacadeInnerError> {
        match self {
            Self::Applied(err) => Some(err),
            Self::Battery(err) => Some(err),
            Self::ClinicalTrial(err) => Some(err),
            Self::Engineering(err) => Some(err),
            Self::GameTheory(err) => Some(err),
            Self::Isosurface(err) => Some(err),
            Self::Kalman(err) => Some(err),
            Self::Lora(err) => Some(err),
            Self::Pharmacokinetics(err) => Some(err),
            Self::VoxelGrid(err) => Some(err),
            _ => None,
        }
    }

    #[cfg(feature = "physics")]
    #[verified_engine::verified]
    fn physics_inner(&self) -> Option<&dyn FacadeInnerError> {
        match self {
            Self::Physics(err) => Some(err),
            Self::Chaos(err) => Some(err),
            Self::DoseFluence(err) => Some(err),
            Self::Fluid(err) => Some(err),
            Self::Gating(err) => Some(err),
            Self::HighEnergy(err) => Some(err),
            Self::Nuclear(err) => Some(err),
            Self::Radar(err) => Some(err),
            Self::SolidState(err) => Some(err),
            Self::StandardModel(err) => Some(err),
            Self::StatMech(err) => Some(err),
            _ => None,
        }
    }

    #[cfg(any(
        feature = "pure_math",
        feature = "ai",
        feature = "applied",
        feature = "biology",
        feature = "epidemiology",
        feature = "physics"
    ))]
    #[verified_engine::verified]
    fn bio_ai_epi_inner(&self) -> Option<&dyn FacadeInnerError> {
        match self {
            #[cfg(feature = "ai")]
            Self::AI(err) => Some(err),
            #[cfg(feature = "biology")]
            Self::Biology(err) => Some(err),
            #[cfg(feature = "biology")]
            Self::HodgkinHuxley(err) => Some(err),
            #[cfg(feature = "biology")]
            Self::Kinetics(err) => Some(err),
            #[cfg(feature = "biology")]
            Self::ReactionDiffusion(err) => Some(err),
            #[cfg(feature = "epidemiology")]
            Self::EpidemiologyDomain(err) => Some(err),
            #[cfg(feature = "epidemiology")]
            Self::Epidemiology(err) => Some(err),
            #[cfg(feature = "epidemiology")]
            Self::NetworkModel(err) => Some(err),
            _ => None,
        }
    }

    #[cfg(any(
        feature = "pure_math",
        feature = "ai",
        feature = "applied",
        feature = "biology",
        feature = "epidemiology",
        feature = "physics"
    ))]
    #[verified_engine::verified]
    fn domain_inner(&self) -> Option<&dyn FacadeInnerError> {
        let found = self.bio_ai_epi_inner();
        #[cfg(feature = "pure_math")]
        let found = found.or_else(|| self.pure_math_inner());
        #[cfg(feature = "applied")]
        let found = found.or_else(|| self.applied_inner());
        #[cfg(feature = "physics")]
        let found = found.or_else(|| self.physics_inner());
        found
    }

    #[verified_engine::verified]
    fn classify(&self) -> Option<InnerErrorRef<'_>> {
        match self {
            Self::Eval(err) => Some(InnerErrorRef::Domain(err)),
            Self::Fit(err) => Some(InnerErrorRef::Domain(err)),
            Self::Glsl(err) => Some(InnerErrorRef::Domain(err)),
            Self::Lex(err) => Some(InnerErrorRef::Domain(err)),
            Self::Math(err) => Some(InnerErrorRef::Domain(err)),
            Self::MathCommons(err) => Some(InnerErrorRef::Domain(err)),
            Self::Parse(err) => Some(InnerErrorRef::Domain(err)),
            Self::Io(err) => Some(InnerErrorRef::Io(err)),
            #[cfg(any(
                feature = "pure_math",
                feature = "ai",
                feature = "applied",
                feature = "biology",
                feature = "epidemiology",
                feature = "physics"
            ))]
            _ => self.domain_inner().map(InnerErrorRef::Domain),
        }
    }
}

impl std::fmt::Display for Error {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.classify() {
            Some(InnerErrorRef::Domain(err)) => std::fmt::Display::fmt(err.as_std_error(), f),
            Some(InnerErrorRef::Io(err)) => std::fmt::Display::fmt(err, f),
            None => Ok(()),
        }
    }
}

impl std::error::Error for Error {
    #[verified_engine::verified]
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self.classify() {
            Some(InnerErrorRef::Domain(err)) => Some(err.as_std_error()),
            Some(InnerErrorRef::Io(err)) => Some(err),
            None => None,
        }
    }
}

impl Diagnostic for Error {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        match self.classify() {
            Some(InnerErrorRef::Domain(err)) => err.as_diagnostic().severity(),
            Some(InnerErrorRef::Io(_)) | None => Severity::Error,
        }
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = match self.classify() {
            Some(InnerErrorRef::Domain(err)) => err.as_diagnostic().metadata(),
            Some(InnerErrorRef::Io(err)) => {
                let mut m = HashMap::new();
                m.insert("error_type".to_string(), "IoError".to_string());
                m.insert("description".to_string(), err.to_string());
                m
            }
            None => HashMap::new(),
        };
        map.insert(
            "umbrella_type".to_string(),
            "math_explorer::error::Error".to_string(),
        );
        map
    }
}
