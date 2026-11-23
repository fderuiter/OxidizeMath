//! Conjugate Bayesian two-arm clinical trial comparisons for binary and continuous endpoints.

use super::types::{ClinicalTrialError, ContingencyTable, GroupData};
use oxidize_core::rng::OxidizeRng;
use pure_math::pure_math::statistics::bayesian::{
    BayesFactor, BetaBinomial, NormalInverseGamma, empirical_credible_interval,
};
use pure_math::pure_math::statistics::distributions::ContinuousDistribution;

const MC_DRAWS: usize = 2048;

/// Posterior summary for a Bayesian two-arm clinical trial comparison (Treatment vs Control).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BayesianTwoArmResult {
    /// Posterior probability $P(\theta_T > \theta_C \mid D)$ or $P(\mu_T > \mu_C \mid D)$.
    pub posterior_prob_superior: f64,
    /// Analytical posterior mean difference $E[\Delta \mid D]$ (Treatment minus Control).
    pub posterior_mean_difference: f64,
    /// Equal-tailed posterior credible interval `(lower, upper)` for Treatment minus Control.
    pub credible_interval: (f64, f64),
    /// Credible interval mass level (e.g. `0.95`).
    pub credible_level: f64,
    /// Bayes factor $BF_{10}$ comparing distinct arm parameters ($M_1$) vs pooled null ($M_0$).
    pub bayes_factor_10: f64,
    /// Natural log Bayes factor $\ln BF_{10}$.
    pub ln_bayes_factor_10: f64,
}

/// Performs a conjugate Beta-Binomial two-arm comparison of binary event rates on a 2x2 [`ContingencyTable`].
///
/// Computes exact analytical Beta-Binomial marginal likelihoods for $BF_{10}$ (independent arm rates
/// vs pooled single rate) and evaluates posterior risk-difference quantiles and superiority probability
/// $P(\theta_T > \theta_C \mid D)$ on stack buffers via `rng`.
///
/// # Example
/// ```rust
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::clinical_trials::bayesian_testing::bayesian_two_arm_proportions;
/// use math_explorer::applied::clinical_trials::types::ContingencyTable;
/// use math_explorer::rng::OxidizeRng;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let table = ContingencyTable::new(36, 14, 18, 32)?;
/// let mut rng = OxidizeRng::from_seed(42);
/// let res = bayesian_two_arm_proportions(&table, 1.0, 1.0, 0.95, &mut rng)?;
/// assert!(res.posterior_prob_superior > 0.99);
/// assert!(res.bayes_factor_10 > 10.0);
/// # Ok(())
/// # }
/// ```
#[verified_engine::verified]
pub fn bayesian_two_arm_proportions(
    table: &ContingencyTable,
    prior_alpha: f64,
    prior_beta: f64,
    credible_level: f64,
    rng: &mut OxidizeRng,
) -> Result<BayesianTwoArmResult, ClinicalTrialError> {
    if !credible_level.is_finite() || credible_level <= 0.0 || credible_level >= 1.0 {
        return Err(ClinicalTrialError::InvalidData(format!(
            "Credible level must lie strictly in (0, 1), got {credible_level}"
        )));
    }
    let n_treat = (table.treatment_event as u64) + (table.treatment_no_event as u64);
    let n_ctrl = (table.control_event as u64) + (table.control_no_event as u64);
    if n_treat == 0 || n_ctrl == 0 {
        return Err(ClinicalTrialError::InsufficientSampleSize {
            required: 1,
            actual: 0,
        });
    }

    let prior = BetaBinomial::new(prior_alpha, prior_beta)
        .map_err(|e| ClinicalTrialError::InvalidData(e.to_string()))?;
    let post_t = prior
        .update(table.treatment_event as u64, n_treat)
        .map_err(|e| ClinicalTrialError::StatisticalError(e.to_string()))?;
    let post_c = prior
        .update(table.control_event as u64, n_ctrl)
        .map_err(|e| ClinicalTrialError::StatisticalError(e.to_string()))?;

    let a = table.treatment_event as f64;
    let b = table.treatment_no_event as f64;
    let c = table.control_event as f64;
    let d = table.control_no_event as f64;
    let ln_b0 = statrs::function::beta::ln_beta(prior_alpha, prior_beta);
    let ln_m1 = statrs::function::beta::ln_beta(prior_alpha + a, prior_beta + b)
        + statrs::function::beta::ln_beta(prior_alpha + c, prior_beta + d)
        - 2.0 * ln_b0;
    let ln_m0 = statrs::function::beta::ln_beta(prior_alpha + a + c, prior_beta + b + d) - ln_b0;
    let bf = BayesFactor::from_ln_marginals(ln_m1, ln_m0)
        .map_err(|e| ClinicalTrialError::StatisticalError(e.to_string()))?;

    let dist_t = post_t
        .posterior_dist()
        .map_err(|e| ClinicalTrialError::StatisticalError(e.to_string()))?;
    let dist_c = post_c
        .posterior_dist()
        .map_err(|e| ClinicalTrialError::StatisticalError(e.to_string()))?;

    let (sup_prob, ci) = sample_difference_summary(&dist_t, &dist_c, credible_level, rng)?;
    Ok(BayesianTwoArmResult {
        posterior_prob_superior: sup_prob,
        posterior_mean_difference: post_t.posterior_mean() - post_c.posterior_mean(),
        credible_interval: ci,
        credible_level,
        bayes_factor_10: bf.bf10,
        ln_bayes_factor_10: bf.ln_bf10,
    })
}

/// Performs a conjugate Normal-Inverse-Gamma two-arm comparison of continuous outcomes.
///
/// Uses a weakly informative empirical-scale prior $\mathrm{NIG}(\bar{y}_{\text{pool}}, \kappa_0 = 1, \alpha_0 = 2, \beta_0 = s^2_{\text{pool}})$
/// shared across arms to compute the exact marginal-likelihood Bayes factor $BF_{10}$ and Behrens-Fisher
/// Student-$t$ posterior difference distribution $\mu_T - \mu_C$.
#[verified_engine::verified]
pub fn bayesian_two_arm_continuous(
    treatment: &GroupData,
    control: &GroupData,
    credible_level: f64,
    rng: &mut OxidizeRng,
) -> Result<BayesianTwoArmResult, ClinicalTrialError> {
    if !credible_level.is_finite() || credible_level <= 0.0 || credible_level >= 1.0 {
        return Err(ClinicalTrialError::InvalidData(format!(
            "Credible level must lie strictly in (0, 1), got {credible_level}"
        )));
    }
    let nt = treatment.n();
    let nc = control.n();
    if nt < 2 || nc < 2 {
        return Err(ClinicalTrialError::InsufficientSampleSize {
            required: 2,
            actual: nt.min(nc),
        });
    }

    let grand_mean =
        ((nt as f64) * treatment.mean() + (nc as f64) * control.mean()) / ((nt + nc) as f64);
    let pooled_scale = (0.5 * (treatment.variance() + control.variance())).max(1e-6);
    let prior = NormalInverseGamma::new(grand_mean, 1.0, 2.0, pooled_scale)
        .map_err(|e| ClinicalTrialError::StatisticalError(e.to_string()))?;

    let post_t = prior
        .update_slice(treatment.as_slice())
        .map_err(|e| ClinicalTrialError::StatisticalError(e.to_string()))?;
    let post_c = prior
        .update_slice(control.as_slice())
        .map_err(|e| ClinicalTrialError::StatisticalError(e.to_string()))?;

    let mut pooled = Vec::with_capacity(nt + nc);
    pooled.extend_from_slice(treatment.as_slice());
    pooled.extend_from_slice(control.as_slice());

    let ln_m1 = prior
        .ln_marginal_likelihood(treatment.as_slice())
        .and_then(|lt| Ok(lt + prior.ln_marginal_likelihood(control.as_slice())?))
        .map_err(|e| ClinicalTrialError::StatisticalError(e.to_string()))?;
    let ln_m0 = prior
        .ln_marginal_likelihood(&pooled)
        .map_err(|e| ClinicalTrialError::StatisticalError(e.to_string()))?;
    let bf = BayesFactor::from_ln_marginals(ln_m1, ln_m0)
        .map_err(|e| ClinicalTrialError::StatisticalError(e.to_string()))?;

    let dist_t = post_t
        .marginal_mean_dist()
        .map_err(|e| ClinicalTrialError::StatisticalError(e.to_string()))?;
    let dist_c = post_c
        .marginal_mean_dist()
        .map_err(|e| ClinicalTrialError::StatisticalError(e.to_string()))?;

    let (sup_prob, ci) = sample_difference_summary(&dist_t, &dist_c, credible_level, rng)?;
    Ok(BayesianTwoArmResult {
        posterior_prob_superior: sup_prob,
        posterior_mean_difference: post_t.mu() - post_c.mu(),
        credible_interval: ci,
        credible_level,
        bayes_factor_10: bf.bf10,
        ln_bayes_factor_10: bf.ln_bf10,
    })
}

fn sample_difference_summary<D: ContinuousDistribution>(
    dist_t: &D,
    dist_c: &D,
    credible_level: f64,
    rng: &mut OxidizeRng,
) -> Result<(f64, (f64, f64)), ClinicalTrialError> {
    let mut diffs = [0.0_f64; MC_DRAWS];
    let mut scratch = [0.0_f64; MC_DRAWS];
    let mut superior_count = 0_usize;
    for slot in &mut diffs {
        let diff = dist_t.sample(rng) - dist_c.sample(rng);
        if diff > 0.0 {
            superior_count += 1;
        }
        *slot = diff;
    }
    let ci = empirical_credible_interval(&diffs, &mut scratch, credible_level)
        .map_err(|e| ClinicalTrialError::StatisticalError(e.to_string()))?;
    Ok((
        (superior_count as f64) / (MC_DRAWS as f64),
        (ci.lower, ci.upper),
    ))
}

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified_engine::verified(opt_out = "tests"))]

    use super::*;

    #[test]
    fn test_bayesian_two_arm_proportions_and_continuous() {
        let table = ContingencyTable::new(42, 8, 15, 35).unwrap();
        let mut rng = OxidizeRng::from_seed(101);
        let prop_res = bayesian_two_arm_proportions(&table, 1.0, 1.0, 0.95, &mut rng).unwrap();
        assert!(prop_res.posterior_prob_superior > 0.999);
        assert!(prop_res.posterior_mean_difference > 0.40);
        assert!(prop_res.credible_interval.0 > 0.25);
        assert!(prop_res.bayes_factor_10 > 100.0);

        let treat = GroupData::new(vec![12.2, 13.1, 11.8, 12.9, 13.4, 12.6, 12.7, 13.0]).unwrap();
        let ctrl = GroupData::new(vec![8.1, 7.9, 8.5, 8.0, 7.7, 8.3, 8.2, 7.8]).unwrap();
        let cont_res = bayesian_two_arm_continuous(&treat, &ctrl, 0.95, &mut rng).unwrap();
        assert!(cont_res.posterior_prob_superior > 0.999);
        assert!(cont_res.posterior_mean_difference > 3.5);
        assert!(cont_res.credible_interval.0 > 2.5);
        assert!(cont_res.bayes_factor_10 > 50.0);
    }
}

// [cite:clinical_trials_statistics]
