use diagnostics::{Diagnostic, Severity};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum EpidemiologyError {
    /// Matrix V (Transition Matrix) is singular and cannot be inverted.
    SingularTransitionMatrix,
    /// Invalid Parameter (e.g., negative rate).
    #[allow(missing_docs)]
    InvalidParameter { name: String, value: f64 },
    /// Missing Parameter (e.g., required field not set in builder).
    #[allow(missing_docs)]
    MissingParameter { name: String },
    /// Wrapped centralized mathematical error.
    Math(math_commons::error::MathError),
}

impl Diagnostic for EpidemiologyError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "EpidemiologyError".to_string());
        if let Self::Math(math_err) = self {
            map.extend(math_err.metadata());
        }
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl std::fmt::Display for EpidemiologyError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for EpidemiologyError {}

impl From<math_commons::error::MathError> for EpidemiologyError {
    fn from(err: math_commons::error::MathError) -> Self {
        EpidemiologyError::Math(err)
    }
}
