use diagnostics::{Diagnostic, Severity};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug, PartialEq, Clone)]
#[allow(missing_docs)]
pub enum HodgkinHuxleyError {
    #[error("Invalid gating variable value: {0} (must be between 0 and 1)")]
    #[allow(missing_docs)]
    InvalidGatingVariable(f64),
    #[error("Invalid conductance value: {0} (must be non-negative)")]
    #[allow(missing_docs)]
    InvalidConductance(f64),
}

impl Diagnostic for HodgkinHuxleyError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "HodgkinHuxleyError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}
