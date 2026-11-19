//! Type definitions for clinical trials.

use std::fmt;

/// Errors that can occur during clinical trial analysis.
#[derive(Debug, Clone)]
pub enum ClinicalTrialError {
    /// Sample size is too small for the requested test.
    #[allow(missing_docs)]
    InsufficientSampleSize { required: usize, actual: usize },
    /// Invalid input data (e.g., negative counts).
    InvalidData(String),
    /// Statistical calculation failed (e.g., convergence issue).
    StatisticalError(String),
    /// Zero total count in a contingency table.
    ZeroTotalCount,
}

impl fmt::Display for ClinicalTrialError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InsufficientSampleSize { required, actual } => {
                write!(
                    f,
                    "Insufficient sample size: required {}, got {}",
                    required, actual
                )
            }
            Self::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
            Self::StatisticalError(msg) => write!(f, "Statistical error: {}", msg),
            Self::ZeroTotalCount => write!(f, "Total count in contingency table is zero"),
        }
    }
}

impl std::error::Error for ClinicalTrialError {}

/// A 2x2 Contingency Table for binary outcomes.
///
/// | | Event | No Event |
/// |---|---|---|
/// | Treatment | a | b |
/// | Control | c | d |
#[derive(Debug, Clone, Copy)]
pub struct ContingencyTable {
    /// Treatment Group: Event count (a).
    pub treatment_event: u32,
    /// Treatment Group: No Event count (b).
    pub treatment_no_event: u32,
    /// Control Group: Event count (c).
    pub control_event: u32,
    /// Control Group: No Event count (d).
    pub control_no_event: u32,
}

impl ContingencyTable {
    /// Creates a new contingency table.
    #[verified_engine::verified]
    pub fn new(
        treatment_event: u32,
        treatment_no_event: u32,
        control_event: u32,
        control_no_event: u32,
    ) -> Result<Self, ClinicalTrialError> {
        let total = treatment_event as u64
            + treatment_no_event as u64
            + control_event as u64
            + control_no_event as u64;

        if total == 0 {
            return Err(ClinicalTrialError::ZeroTotalCount);
        }

        Ok(Self {
            treatment_event,
            treatment_no_event,
            control_event,
            control_no_event,
        })
    }

    /// Returns total count.
    #[verified_engine::verified]
    pub fn total(&self) -> f64 {
        (self.treatment_event
            + self.treatment_no_event
            + self.control_event
            + self.control_no_event) as f64
    }
}

/// A wrapper for continuous data from a study group.
///
/// Calculates and caches summary statistics to avoid re-computation.
#[derive(Debug, Clone)]
pub struct GroupData {
    data: Vec<f64>,
}

impl GroupData {
    /// Creates a new `GroupData` instance.
    #[verified_engine::verified]
    pub fn new(data: Vec<f64>) -> Result<Self, ClinicalTrialError> {
        if data.len() < 2 {
            return Err(ClinicalTrialError::InsufficientSampleSize {
                required: 2,
                actual: data.len(),
            });
        }

        // Sentinel: Ensure data is valid (finite).
        for (i, &x) in data.iter().enumerate() {
            if !x.is_finite() {
                return Err(ClinicalTrialError::InvalidData(format!(
                    "Data contains non-finite value (NaN or Infinity) at index {}: {}",
                    i, x
                )));
            }
        }

        Ok(Self { data })
    }

    /// Returns the raw data slice.
    #[verified_engine::verified]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Returns the number of samples.
    #[verified_engine::verified]
    pub fn n(&self) -> usize {
        self.data.len()
    }

    /// Calculates the mean.
    #[verified_engine::verified]
    pub fn mean(&self) -> f64 {
        let sum: f64 = self.data.iter().sum();
        sum / self.n() as f64
    }

    /// Calculates the variance.
    #[verified_engine::verified]
    pub fn variance(&self) -> f64 {
        let m = self.mean();
        let sum_sq_diff: f64 = self.data.iter().map(|x| (x - m).powi(2)).sum();
        sum_sq_diff / (self.n() - 1) as f64
    }
}

/// A finite, non-negative time value for survival analysis.
///
/// Ensures that time is valid (finite and `>= 0.0`), normalizing `-0.0` to `+0.0`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SurvivalTime(f64);

impl SurvivalTime {
    /// Creates a new `SurvivalTime`.
    ///
    /// Normalizes `-0.0` to `+0.0` so signed zeros sort identically under `f64::total_cmp`.
    ///
    /// # Errors
    /// Returns `ClinicalTrialError::InvalidData` if `t` is non-finite (`NaN` or `±Infinity`) or negative.
    #[verified_engine::verified]
    pub fn new(t: f64) -> Result<Self, ClinicalTrialError> {
        if !t.is_finite() || t < 0.0 {
            return Err(ClinicalTrialError::InvalidData(format!(
                "Time must be non-negative and finite, got {}",
                t
            )));
        }
        let normalized = if t == 0.0 { 0.0 } else { t };
        Ok(Self(normalized))
    }

    /// Returns the underlying `f64` value.
    #[verified_engine::verified]
    pub fn as_f64(&self) -> f64 {
        self.0
    }
}

impl fmt::Display for SurvivalTime {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
use std::ops::{Add, Div, Mul, Sub};

impl SurvivalTime {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn powf(&self, n: f64) -> Result<Self, ClinicalTrialError> {
        Self::new(self.0.powf(n))
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn sqrt(&self) -> Result<Self, ClinicalTrialError> {
        Self::new(self.0.sqrt())
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn ln(&self) -> Result<Self, ClinicalTrialError> {
        Self::new(self.0.ln())
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn exp(&self) -> Result<Self, ClinicalTrialError> {
        Self::new(self.0.exp())
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn abs(&self) -> Result<Self, ClinicalTrialError> {
        Self::new(self.0.abs())
    }
}

impl Add<f64> for SurvivalTime {
    type Output = Result<Self, ClinicalTrialError>;
    fn add(self, rhs: f64) -> Self::Output {
        Self::new(self.0 + rhs)
    }
}
impl Sub<f64> for SurvivalTime {
    type Output = Result<Self, ClinicalTrialError>;
    fn sub(self, rhs: f64) -> Self::Output {
        Self::new(self.0 - rhs)
    }
}
impl Mul<f64> for SurvivalTime {
    type Output = Result<Self, ClinicalTrialError>;
    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.0 * rhs)
    }
}
impl Div<f64> for SurvivalTime {
    type Output = Result<Self, ClinicalTrialError>;
    fn div(self, rhs: f64) -> Self::Output {
        Self::new(self.0 / rhs)
    }
}

impl Add<SurvivalTime> for f64 {
    type Output = Result<SurvivalTime, ClinicalTrialError>;
    fn add(self, rhs: SurvivalTime) -> Self::Output {
        SurvivalTime::new(self + rhs.0)
    }
}
impl Sub<SurvivalTime> for f64 {
    type Output = Result<SurvivalTime, ClinicalTrialError>;
    fn sub(self, rhs: SurvivalTime) -> Self::Output {
        SurvivalTime::new(self - rhs.0)
    }
}
impl Mul<SurvivalTime> for f64 {
    type Output = Result<SurvivalTime, ClinicalTrialError>;
    fn mul(self, rhs: SurvivalTime) -> Self::Output {
        SurvivalTime::new(self * rhs.0)
    }
}
impl Div<SurvivalTime> for f64 {
    type Output = Result<SurvivalTime, ClinicalTrialError>;
    fn div(self, rhs: SurvivalTime) -> Self::Output {
        SurvivalTime::new(self / rhs.0)
    }
}

impl Add<SurvivalTime> for SurvivalTime {
    type Output = Result<Self, ClinicalTrialError>;
    fn add(self, rhs: SurvivalTime) -> Self::Output {
        Self::new(self.0 + rhs.0)
    }
}
impl Sub<SurvivalTime> for SurvivalTime {
    type Output = Result<Self, ClinicalTrialError>;
    fn sub(self, rhs: SurvivalTime) -> Self::Output {
        Self::new(self.0 - rhs.0)
    }
}
impl Mul<SurvivalTime> for SurvivalTime {
    type Output = Result<Self, ClinicalTrialError>;
    fn mul(self, rhs: SurvivalTime) -> Self::Output {
        Self::new(self.0 * rhs.0)
    }
}
impl Div<SurvivalTime> for SurvivalTime {
    type Output = f64;
    fn div(self, rhs: SurvivalTime) -> f64 {
        self.0 / rhs.0
    }
}
