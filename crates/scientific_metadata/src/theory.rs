#![cfg_attr(any(), verified(opt_out = "metadata"))]

use std::collections::HashMap;

/// A structured representation of mathematical constraints for a parameter.
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
/// for every mathematical model.
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
