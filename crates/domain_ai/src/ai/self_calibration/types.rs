//! Types used in self-calibration.

/// Type alias for an answer string.
pub type Answer = String;

/// Represents a response from a model.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Response {
    /// The text of the response.
    pub text: String,
    /// The probability assigned by the model.
    pub probability: f64,
    /// The extracted answer.
    pub answer: Answer,
}
