#![cfg_attr(any(), verified(opt_out = "metadata"))]

use std::collections::HashMap;

/// A structured representation of mathematical constraints for a parameter.
///
/// Encapsulates numeric ranges and adjustment steps used by user interfaces
/// and simulation solvers to validate parameter inputs.
///
/// # Example
///
/// ```rust
/// use scientific_metadata::theory::ParameterConstraint;
///
/// let constraint = ParameterConstraint {
///     min: 0.0,
///     max: 10.0,
///     step: 0.1,
/// };
/// assert_eq!(constraint.min, 0.0);
/// assert_eq!(constraint.max, 10.0);
/// assert_eq!(constraint.step, 0.1);
/// assert!(constraint.min <= constraint.max);
/// assert!(constraint.step > 0.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ParameterConstraint {
    /// Minimum allowable numerical value for the parameter.
    pub min: f64,
    /// Maximum allowable numerical value for the parameter.
    pub max: f64,
    /// Recommended adjustment step size for interactive UI controls.
    pub step: f64,
}

/// A standardized trait for exporting human-readable, verified descriptions
/// and parameter metadata for mathematical models.
///
/// Implementing this trait allows simulation models across the workspace to expose
/// verified descriptions, phonetic representations for accessibility, literature citations,
/// and parameter constraint metadata without coupling calculation routines to UI or database dependencies.
///
/// # Example
///
/// ```rust
/// use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
/// use std::collections::HashMap;
///
/// struct Pendulum {
///     length: f64,
/// }
///
/// impl TheoryDescribable for Pendulum {
///     fn theory_description(&self) -> String {
///         format!("Simple pendulum of length {:.2} meters", self.length)
///     }
///
///     fn phonetic_description(&self) -> String {
///         "Simple pendulum".to_string()
///     }
///
///     fn theory_citation(&self) -> String {
///         "Landau, L. D., & Lifshitz, E. M. (1976). Mechanics.".to_string()
///     }
///
///     fn available_descriptions(&self) -> HashMap<String, String> {
///         let mut descs = HashMap::new();
///         descs.insert("default".to_string(), "Pendulum oscillation".to_string());
///         descs
///     }
///
///     fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
///         let mut params = HashMap::new();
///         params.insert(
///             "length".to_string(),
///             ParameterConstraint {
///                 min: 0.1,
///                 max: 10.0,
///                 step: 0.1,
///             },
///         );
///         params
///     }
/// }
///
/// let pendulum = Pendulum { length: 2.0 };
/// assert_eq!(pendulum.theory_description(), "Simple pendulum of length 2.00 meters");
/// assert_eq!(pendulum.phonetic_description(), "Simple pendulum");
/// assert_eq!(pendulum.theory_citation(), "Landau, L. D., & Lifshitz, E. M. (1976). Mechanics.");
/// assert_eq!(pendulum.theory_latex_equation(), None);
/// assert_eq!(pendulum.theory_parameters()["length"].min, 0.1);
/// assert_eq!(pendulum.theory_parameters()["length"].max, 10.0);
/// ```
pub trait TheoryDescribable {
    /// Returns the verified human-readable description for accessibility and UI tooltips.
    fn theory_description(&self) -> String;

    /// Returns the human-verified phonetic string for screen readers and speech synthesis.
    fn phonetic_description(&self) -> String;

    /// Returns the academic or bibliographic citation for the mathematical model.
    fn theory_citation(&self) -> String;

    /// Returns a list of all available descriptions for a given mathematical state, regime, or operation.
    fn available_descriptions(&self) -> HashMap<String, String>;

    /// Returns theoretical constraints for parameter fields of the model.
    ///
    /// By default, returns an empty map for models without exposed parameters.
    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        HashMap::new()
    }

    /// Returns an optional LaTeX formula string representing the model's governing equation.
    ///
    /// By default, returns `None` to maintain backwards compatibility for models without explicit equations.
    fn theory_latex_equation(&self) -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockModel {
        description: String,
        phonetic: String,
        citation: String,
    }

    impl MockModel {
        fn new() -> Self {
            Self {
                description: "Verified test description".to_string(),
                phonetic: "test model".to_string(),
                citation: "Author et al. (2026)".to_string(),
            }
        }
    }

    impl TheoryDescribable for MockModel {
        fn theory_description(&self) -> String {
            self.description.clone()
        }

        fn phonetic_description(&self) -> String {
            self.phonetic.clone()
        }

        fn theory_citation(&self) -> String {
            self.citation.clone()
        }

        fn available_descriptions(&self) -> HashMap<String, String> {
            let mut map = HashMap::new();
            map.insert("default".to_string(), self.description.clone());
            map
        }
    }

    #[test]
    fn test_theory_latex_equation_default_and_override() {
        let model = MockModel::new();
        assert_eq!(model.theory_description(), "Verified test description");
        assert_eq!(model.phonetic_description(), "test model");
        assert_eq!(model.theory_citation(), "Author et al. (2026)");
        assert_eq!(
            model
                .available_descriptions()
                .get("default")
                .map(String::as_str),
            Some("Verified test description")
        );
        assert!(model.theory_parameters().is_empty());
        assert_eq!(model.theory_latex_equation(), None);

        struct ModelWithEq;
        impl TheoryDescribable for ModelWithEq {
            fn theory_description(&self) -> String {
                "Description".into()
            }
            fn phonetic_description(&self) -> String {
                "Description".into()
            }
            fn theory_citation(&self) -> String {
                "Citation".into()
            }
            fn available_descriptions(&self) -> HashMap<String, String> {
                HashMap::new()
            }
            fn theory_latex_equation(&self) -> Option<String> {
                Some(r"E = m c^2".into())
            }
        }

        let model_eq = ModelWithEq;
        assert_eq!(
            model_eq.theory_latex_equation(),
            Some("E = m c^2".to_string())
        );
    }
}
