use super::types::{ClinicalTrialError, ContingencyTable, GroupData};
use statrs::distribution::{ChiSquared, ContinuousCDF, StudentsT};

/// Defines a strategy for performing a hypothesis test.
pub trait HypothesisTest<Input> {
    /// Performs the test on the given data.
    #[verified_engine::verified]
    fn test(&self, data: Input) -> Result<TestResult, ClinicalTrialError>;
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub struct TestResult {
    #[allow(missing_docs)]
    pub statistic: f64,
    #[allow(missing_docs)]
    pub p_value: f64,
    #[allow(missing_docs)]
    pub is_significant: bool, // Based on provided alpha
    #[allow(missing_docs)]
    pub confidence_interval: Option<(f64, f64)>, // For the difference
}

/// Strategy for an independent two-sample t-test (assuming equal variances).
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TTestIndependent {
    #[allow(missing_docs)]
    pub alpha: f64,
}

impl TTestIndependent {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl<'a> HypothesisTest<(&'a GroupData, &'a GroupData)> for TTestIndependent {
    #[verified_engine::verified]
    fn test(&self, data: (&'a GroupData, &'a GroupData)) -> Result<TestResult, ClinicalTrialError> {
        let (group1, group2) = data;
        let n1 = group1.n();
        let n2 = group2.n();

        // Already checked in GroupData::new, but safe to check again if internals change
        if n1 < 2 || n2 < 2 {
            return Err(ClinicalTrialError::InsufficientSampleSize {
                required: 2,
                actual: n1.min(n2),
            });
        }

        let mean1 = group1.mean();
        let mean2 = group2.mean();

        let var1 = group1.variance();
        let var2 = group2.variance();

        // Pooled variance
        let dof = (n1 + n2 - 2) as f64;
        let pooled_var = ((n1 as f64 - 1.0) * var1 + (n2 as f64 - 1.0) * var2) / dof;
        let pooled_std = pooled_var.sqrt();

        // Standard error of the difference
        let se_diff = pooled_std * (1.0 / n1 as f64 + 1.0 / n2 as f64).sqrt();

        // t-statistic
        let t_stat = (mean1 - mean2) / se_diff;

        // P-value (two-tailed)
        let t_dist = StudentsT::new(0.0, 1.0, dof)
            .map_err(|e| ClinicalTrialError::StatisticalError(e.to_string()))?;
        let p_value = 2.0 * (1.0 - t_dist.cdf(t_stat.abs()));

        // Confidence Interval for the difference
        // diff +/- t_crit * se_diff
        let t_crit = t_dist.inverse_cdf(1.0 - self.alpha / 2.0);
        let margin = t_crit * se_diff;
        let diff = mean1 - mean2;
        let ci = (diff - margin, diff + margin);

        Ok(TestResult {
            statistic: t_stat,
            p_value,
            is_significant: p_value < self.alpha,
            confidence_interval: Some(ci),
        })
    }
}

/// Strategy for a Chi-Square test for a 2x2 contingency table.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChiSquare2x2 {
    #[allow(missing_docs)]
    pub alpha: f64,
}

impl ChiSquare2x2 {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl<'a> HypothesisTest<&'a ContingencyTable> for ChiSquare2x2 {
    #[verified_engine::verified]
    fn test(&self, table: &'a ContingencyTable) -> Result<TestResult, ClinicalTrialError> {
        let total = table.total();
        // total == 0 check is done in ContingencyTable constructor.

        let a_f = table.treatment_event as f64;
        let b_f = table.treatment_no_event as f64;
        let c_f = table.control_event as f64;
        let d_f = table.control_no_event as f64;

        let row1 = a_f + b_f;
        let row2 = c_f + d_f;
        let col1 = a_f + c_f;
        let col2 = b_f + d_f;

        // Expected values
        let e_a = (row1 * col1) / total;
        let e_b = (row1 * col2) / total;
        let e_c = (row2 * col1) / total;
        let e_d = (row2 * col2) / total;

        if e_a == 0.0 || e_b == 0.0 || e_c == 0.0 || e_d == 0.0 {
            return Err(ClinicalTrialError::StatisticalError(
                "Expected frequencies too low (zero) for Chi-Square".to_string(),
            ));
        }

        let term_a = (a_f - e_a).powi(2) / e_a;
        let term_b = (b_f - e_b).powi(2) / e_b;
        let term_c = (c_f - e_c).powi(2) / e_c;
        let term_d = (d_f - e_d).powi(2) / e_d;

        let chi_sq = term_a + term_b + term_c + term_d;

        // DoF for 2x2 is 1
        let dist = ChiSquared::new(1.0)
            .map_err(|e| ClinicalTrialError::StatisticalError(e.to_string()))?;
        let p_value = 1.0 - dist.cdf(chi_sq);

        Ok(TestResult {
            statistic: chi_sq,
            p_value,
            is_significant: p_value < self.alpha,
            confidence_interval: None, // CI for chi-sq statistic is not standard; usually CI for OR/RR is used.
        })
    }
}

/// Performs an independent two-sample t-test (assuming equal variances).
///
/// # Arguments
/// * `group1` - Data for group 1.
/// * `group2` - Data for group 2.
/// * `alpha` - Significance level (e.g., 0.05).
#[verified_engine::verified]
pub fn t_test_independent(
    group1: &GroupData,
    group2: &GroupData,
    alpha: f64,
) -> Result<TestResult, ClinicalTrialError> {
    TTestIndependent::new(alpha).test((group1, group2))
}

/// Performs a Chi-Square test for a 2x2 contingency table.
///
/// # Arguments
/// * `table` - The 2x2 contingency table.
/// * `alpha` - Significance level.
#[verified_engine::verified]
pub fn chi_square_2x2(
    table: &ContingencyTable,
    alpha: f64,
) -> Result<TestResult, ClinicalTrialError> {
    ChiSquare2x2::new(alpha).test(table)
}
