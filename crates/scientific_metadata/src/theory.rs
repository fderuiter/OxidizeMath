#![cfg_attr(any(), verified(opt_out = "metadata"))]

use std::collections::HashMap;

/// A structured representation of mathematical constraints for a parameter.
///
/// Defines acceptable boundaries and step intervals for numerical model parameters.
///
/// # Example
/// ```
/// use scientific_metadata::theory::ParameterConstraint;
///
/// let constraint = ParameterConstraint {
///     min: 0.0,
///     max: 100.0,
///     step: 0.5,
/// };
/// assert_eq!(constraint.min, 0.0);
/// assert_eq!(constraint.max, 100.0);
/// assert_eq!(constraint.step, 0.5);
/// ```
#[derive(Debug, Clone)]
pub struct ParameterConstraint {
    #[allow(missing_docs)]
    pub min: f64,
    #[allow(missing_docs)]
    pub max: f64,
    #[allow(missing_docs)]
    pub step: f64,
}

/// A standardized trait for exporting human-readable, verified descriptions
/// and parameter constraints for mathematical models.
///
/// # Example
/// ```
/// use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
/// use std::collections::HashMap;
///
/// struct HarmonicOscillator {
///     frequency: f64,
/// }
///
/// impl TheoryDescribable for HarmonicOscillator {
///     fn theory_description(&self) -> String {
///         "Harmonic oscillator simulation model".to_string()
///     }
///     fn phonetic_description(&self) -> String {
///         "har-mon-ik os-uh-lay-ter".to_string()
///     }
///     fn theory_citation(&self) -> String {
///         "Goldstein, Classical Mechanics (1980)".to_string()
///     }
///     fn available_descriptions(&self) -> HashMap<String, String> {
///         let mut descs = HashMap::new();
///         descs.insert("default".to_string(), self.theory_description());
///         descs
///     }
///     fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
///         let mut params = HashMap::new();
///         params.insert(
///             "frequency".to_string(),
///             ParameterConstraint {
///                 min: 0.1,
///                 max: 1000.0,
///                 step: 0.1,
///             },
///         );
///         params
///     }
///     fn get_parameter(&self, name: &str) -> Option<f64> {
///         if name == "frequency" {
///             Some(self.frequency)
///         } else {
///             None
///         }
///     }
///     fn set_parameter(&mut self, name: &str, value: f64) {
///         if name == "frequency" {
///             self.frequency = value;
///         }
///     }
/// }
///
/// let mut osc = HarmonicOscillator { frequency: 440.0 };
/// assert_eq!(osc.theory_description(), "Harmonic oscillator simulation model");
/// assert_eq!(osc.theory_parameters()["frequency"].min, 0.1);
/// assert_eq!(osc.get_parameter("frequency"), Some(440.0));
/// osc.set_parameter("frequency", 880.0);
/// assert_eq!(osc.get_parameter("frequency"), Some(880.0));
/// ```
pub trait TheoryDescribable {
    /// Returns the verified human-readable description for accessibility.
    fn theory_description(&self) -> String;

    /// Returns the human-verified phonetic string for screen readers.
    fn phonetic_description(&self) -> String;

    /// Returns the citation for the mathematical model.
    fn theory_citation(&self) -> String;

    /// Returns a list of all available descriptions for a given mathematical state or operation.
    fn available_descriptions(&self) -> HashMap<String, String>;

    /// Returns theoretical constraints for parameter fields of the model.
    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        HashMap::new()
    }

    /// Dynamically gets a parameter by name
    fn get_parameter(&self, _name: &str) -> Option<f64> {
        None
    }

    /// Dynamically sets a parameter by name
    fn set_parameter(&mut self, _name: &str, _value: f64) {}
}

/// A macro active within the GUI crate context to fetch the theoretical description.
#[macro_export]
macro_rules! theory_verification {
    ($obj:expr) => {
        $obj.theory_description()
    };
}
