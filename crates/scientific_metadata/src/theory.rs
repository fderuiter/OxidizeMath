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
#[derive(Debug, Clone)]
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
/// and dynamic parameter reflection without coupling calculation routines to UI or database dependencies.
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
///
///     fn get_parameter(&self, name: &str) -> Option<f64> {
///         match name {
///             "length" => Some(self.length),
///             _ => None,
///         }
///     }
///
///     fn set_parameter(&mut self, name: &str, value: f64) {
///         if name == "length" {
///             self.length = value;
///         }
///     }
/// }
///
/// let mut pendulum = Pendulum { length: 2.0 };
/// assert_eq!(pendulum.theory_description(), "Simple pendulum of length 2.00 meters");
/// assert_eq!(pendulum.phonetic_description(), "Simple pendulum");
/// assert_eq!(pendulum.theory_citation(), "Landau, L. D., & Lifshitz, E. M. (1976). Mechanics.");
/// assert_eq!(pendulum.theory_latex_equation(), None);
/// assert_eq!(pendulum.theory_parameters()["length"].min, 0.1);
/// assert_eq!(pendulum.get_parameter("length"), Some(2.0));
/// pendulum.set_parameter("length", 3.5);
/// assert_eq!(pendulum.get_parameter("length"), Some(3.5));
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

    /// Dynamically retrieves the current floating-point value of a named parameter, or `None` if not found.
    fn get_parameter(&self, _name: &str) -> Option<f64> {
        None
    }

    /// Dynamically updates the floating-point value of a named parameter.
    ///
    /// If the parameter name is unrecognized, this call is a no-op.
    fn set_parameter(&mut self, _name: &str, _value: f64) {}

    /// Returns an optional LaTeX formula string representing the model's governing equation.
    ///
    /// By default, returns `None` to maintain backwards compatibility for models without explicit equations.
    fn theory_latex_equation(&self) -> Option<String> {
        None
    }
}

/// A macro active within user interface contexts to fetch the theoretical description of a model.
///
/// Delegates to [`TheoryDescribable::theory_description`].
///
/// # Example
///
/// ```rust
/// use scientific_metadata::theory::TheoryDescribable;
/// use scientific_metadata::theory_description;
/// use std::collections::HashMap;
///
/// struct Model;
/// impl TheoryDescribable for Model {
///     fn theory_description(&self) -> String { "Verified Model Description".to_string() }
///     fn phonetic_description(&self) -> String { "Verified Model".to_string() }
///     fn theory_citation(&self) -> String { "Citation 2026".to_string() }
///     fn available_descriptions(&self) -> HashMap<String, String> { HashMap::new() }
/// }
///
/// let m = Model;
/// assert_eq!(theory_description!(m), "Verified Model Description");
/// ```
#[macro_export]
macro_rules! theory_description {
    ($obj:expr) => {
        $obj.theory_description()
    };
}

pub use crate::theory_description;
#[allow(deprecated)]
pub use crate::theory_verification;

/// Deprecated compatibility alias for [`theory_description!`].
#[deprecated(
    since = "0.4.0",
    note = "Use `theory_description!` instead to avoid naming conflicts with `pure_math::theory_verification!`"
)]
#[macro_export]
macro_rules! theory_verification {
    ($obj:expr) => {
        $crate::theory_description!($obj)
    };
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
    fn test_theory_description_macro_expansion() {
        let model = MockModel::new();
        let desc = theory_description!(model);
        assert_eq!(desc, "Verified test description");

        let desc_ref = theory_description!(&model);
        assert_eq!(desc_ref, "Verified test description");
    }

    #[test]
    #[allow(deprecated)]
    fn test_deprecated_theory_verification_macro_alias() {
        let model = MockModel::new();
        let desc = theory_verification!(model);
        assert_eq!(desc, "Verified test description");

        let desc_ref = theory_verification!(&model);
        assert_eq!(desc_ref, "Verified test description");
    }

    #[test]
    fn test_theory_latex_equation_default_and_override() {
        let model = MockModel::new();
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
