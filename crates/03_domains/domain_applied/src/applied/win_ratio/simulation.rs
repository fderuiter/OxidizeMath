//! # Mathematics of the Simulation Study
//!
//! This module implements the formulas used in the paper's simulation study.
//! It includes joint survival functions, the win ratio parameter, and derived
//! marginal and conditional survival functions and their PDFs.

pub use super::power_sweep::{
    MAX_ARM_SIZE, MAX_SWEEP_REPLICATES, MAX_SWEEP_STEPS, PowerSweepConfig, PowerSweepReport,
    PowerSweepWorkspace, SampleSizePowerPoint, run_power_sweep_into, run_weibull_power_sweep,
    simulate_weibull_trial_into,
};
pub use super::weibull_sampler::SimulatedSubject;

/// Shared-frailty bivariate Weibull joint survival model for composite fatal and non-fatal endpoints.
///
/// $$
/// S(t, x) = \exp\!\left(-(\lambda_1 t + \lambda_2 x)^\alpha\right)
/// $$
///
/// Reference: Pocock et al. (2012), DOI: 10.1093/eurheartj/ehr352; Oakes (2016), DOI: 10.1111/biom.12512;
/// [`docs/math_specs/domain_applied.md`](../../../../../../docs/math_specs/domain_applied.md).
///
/// # Stability
/// Evaluates conditional survival $G(x \mid c) = S(c, x)/S(c, 0)$ with explicit underflow protection when $S(c, 0) = 0$.
///
/// # Domain of Validity
/// Valid for hazard scales $\lambda_1, \lambda_2 > 0$, association shape $\alpha > 0$, and event times $t, x \ge 0$.
#[derive(Debug, Clone, Copy)]
pub struct BivariateWeibullModel {
    /// Scale parameter for the first event (T).
    pub lambda1: f64,
    /// Scale parameter for the second event (X).
    pub lambda2: f64,
    /// Shape parameter (correlation).
    pub alpha: f64,
}

impl BivariateWeibullModel {
    /// Creates a new `BivariateWeibullModel`.
    #[verified_engine::verified]
    pub fn new(lambda1: f64, lambda2: f64, alpha: f64) -> Self {
        Self {
            lambda1,
            lambda2,
            alpha,
        }
    }

    /// Joint survival function S(t, x).
    #[verified_engine::verified]
    pub fn joint_survival(&self, t: f64, x: f64) -> f64 {
        (-(self.lambda1 * t + self.lambda2 * x).powf(self.alpha)).exp()
    }

    /// Marginal survival function for T, S(t).
    #[verified_engine::verified]
    pub fn marginal_survival_t(&self, t: f64) -> f64 {
        self.joint_survival(t, 0.0)
    }

    /// PDF for T, f(t).
    #[verified_engine::verified]
    pub fn pdf_t(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        let l1_alpha = self.lambda1.powf(self.alpha);
        // f(t) = - dS(t)/dt = alpha * lambda1 * (lambda1 * t)^(alpha-1) * S(t)
        //      = alpha * lambda1^alpha * t^(alpha-1) * S(t)
        self.alpha * l1_alpha * t.powf(self.alpha - 1.0) * self.marginal_survival_t(t)
    }

    /// Conditional survival function for X given T > c, G(x|c) = S(c, x) / S(c, 0).
    #[verified_engine::verified]
    pub fn conditional_survival_x_given_t(&self, x: f64, c: f64) -> f64 {
        let s_c_x = self.joint_survival(c, x);
        let s_c_0 = self.marginal_survival_t(c);
        if s_c_0 == 0.0 { 0.0 } else { s_c_x / s_c_0 }
    }

    /// Conditional PDF for X given T > c, f(x|c).
    #[verified_engine::verified]
    pub fn pdf_x_given_t(&self, x: f64, c: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let term = self.lambda1 * c + self.lambda2 * x;
        if term < 0.0 {
            return 0.0;
        }
        let g = self.conditional_survival_x_given_t(x, c);
        // f(x|c) = - dG(x|c)/dx
        // G(x|c) = exp( - (lambda1*c + lambda2*x)^alpha ) / S(c,0)
        // dG/dx = G * (- alpha * (lambda1*c + lambda2*x)^(alpha-1) * lambda2)
        // f = alpha * lambda2 * (term)^(alpha-1) * G
        self.alpha * self.lambda2 * term.powf(self.alpha - 1.0) * g
    }
}

/// Parameters for the simulation study.
#[derive(Debug, Clone, Copy)]
pub struct SimulationParams {
    /// Lambda1 parameter.
    pub lambda1: f64,
    /// Lambda2 parameter.
    pub lambda2: f64,
    /// Alpha parameter (correlation).
    pub alpha: f64,
    /// Theta parameter (treatment effect).
    pub theta: f64,
}

impl SimulationParams {
    /// Creates a new `SimulationParams`.
    #[verified_engine::verified]
    pub fn new(lambda1: f64, lambda2: f64, alpha: f64, theta: f64) -> Self {
        Self {
            lambda1,
            lambda2,
            alpha,
            theta,
        }
    }

    /// Returns the model for the control group.
    #[verified_engine::verified]
    pub fn control_model(&self) -> BivariateWeibullModel {
        BivariateWeibullModel::new(self.lambda1, self.lambda2, self.alpha)
    }

    /// Returns the model for the treatment group.
    #[verified_engine::verified]
    pub fn treatment_model(&self) -> BivariateWeibullModel {
        BivariateWeibullModel::new(
            self.theta * self.lambda1,
            self.theta * self.lambda2,
            self.alpha,
        )
    }
}

// --- Joint Survival Functions ---

/// Joint survival function for the control group, S0(t,x).
#[verified_engine::verified]
pub fn joint_survival_function_control(t: f64, x: f64, params: &SimulationParams) -> f64 {
    params.control_model().joint_survival(t, x)
}

/// Joint survival function for the treatment group, S1(t,x).
#[verified_engine::verified]
pub fn joint_survival_function_treatment(t: f64, x: f64, params: &SimulationParams) -> f64 {
    params.treatment_model().joint_survival(t, x)
}

// --- Win Ratio Parameter ---

/// Win ratio parameter in the absence of censoring, PR_W.
#[verified_engine::verified]
pub fn win_ratio_parameter(params: &SimulationParams) -> f64 {
    // This depends on the specific relationship between control and treatment models defined by theta.
    1.0 / params.theta.powf(params.alpha)
}

// --- Derived Functions for T (Fatal Event) ---

/// Marginal survival function for T in the control group, S0(t).
#[verified_engine::verified]
pub fn marginal_survival_t_control(t: f64, params: &SimulationParams) -> f64 {
    params.control_model().marginal_survival_t(t)
}

/// Marginal survival function for T in the treatment group, S1(t).
#[verified_engine::verified]
pub fn marginal_survival_t_treatment(t: f64, params: &SimulationParams) -> f64 {
    params.treatment_model().marginal_survival_t(t)
}

/// PDF for T in the control group, f0(t).
#[verified_engine::verified]
pub fn pdf_t_control(t: f64, params: &SimulationParams) -> f64 {
    params.control_model().pdf_t(t)
}

/// PDF for T in the treatment group, f1(t).
#[verified_engine::verified]
pub fn pdf_t_treatment(t: f64, params: &SimulationParams) -> f64 {
    params.treatment_model().pdf_t(t)
}

// --- Derived Functions for X (Non-Fatal Event) given T > c ---

/// Conditional survival function for X in the control group, G0(x|c).
#[verified_engine::verified]
pub fn conditional_survival_x_given_t_control(x: f64, c: f64, params: &SimulationParams) -> f64 {
    params.control_model().conditional_survival_x_given_t(x, c)
}

/// Conditional survival function for X in the treatment group, G1(x|c).
#[verified_engine::verified]
pub fn conditional_survival_x_given_t_treatment(x: f64, c: f64, params: &SimulationParams) -> f64 {
    params
        .treatment_model()
        .conditional_survival_x_given_t(x, c)
}

/// Conditional PDF for X in the control group, f0(x|c).
#[verified_engine::verified]
pub fn pdf_x_given_t_control(x: f64, c: f64, params: &SimulationParams) -> f64 {
    params.control_model().pdf_x_given_t(x, c)
}

/// Conditional PDF for X in the treatment group, f1(x|c).
#[verified_engine::verified]
pub fn pdf_x_given_t_treatment(x: f64, c: f64, params: &SimulationParams) -> f64 {
    params.treatment_model().pdf_x_given_t(x, c)
}

#[inline]
pub(super) fn compare_simulated_pair(
    cmp: &super::censoring::RightCensoredComparator,
    trt_t: [f64; 2],
    trt_e: [u8; 2],
    ctrl_t: [f64; 2],
    ctrl_e: [u8; 2],
) -> (super::pair_comparison::ComparisonResult, usize) {
    let fatal = cmp.compare_raw(trt_t[0], trt_e[0] != 0, ctrl_t[0], ctrl_e[0] != 0);
    if fatal != super::pair_comparison::ComparisonResult::Tie {
        return (fatal, 0);
    }
    (
        cmp.compare_raw(trt_t[1], trt_e[1] != 0, ctrl_t[1], ctrl_e[1] != 0),
        1,
    )
}

fn build_two_tier_counts(
    n1: usize,
    n2: usize,
    tier_wins: [usize; 2],
    tier_losses: [usize; 2],
) -> [super::censoring::TierCount; 2] {
    use super::censoring::TierCount;
    let total_pairs = n1.saturating_mul(n2);
    let rem1 = total_pairs.saturating_sub(tier_wins[0] + tier_losses[0]);
    let rem2 = rem1.saturating_sub(tier_wins[1] + tier_losses[1]);
    [
        TierCount {
            tier_index: 0,
            wins: tier_wins[0],
            losses: tier_losses[0],
            ties_remaining: rem1,
        },
        TierCount {
            tier_index: 1,
            wins: tier_wins[1],
            losses: tier_losses[1],
            ties_remaining: rem2,
        },
    ]
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_simulated_cohorts_into(
    trt_times: &[[f64; 2]],
    trt_events: &[[u8; 2]],
    ctrl_times: &[[f64; 2]],
    ctrl_events: &[[u8; 2]],
    row_wins: &mut [u32],
    row_losses: &mut [u32],
    col_wins: &mut [u32],
    col_losses: &mut [u32],
    sig_alpha: f64,
) -> (
    Option<super::u_statistic::UnmatchedWinRatioSummary>,
    [super::censoring::TierCount; 2],
) {
    use super::censoring::RightCensoredComparator;
    use super::pair_comparison::ComparisonResult;
    use super::u_statistic::{SubjectMarginals, compute_unmatched_u_statistics};

    let (n1, n2) = (trt_times.len(), ctrl_times.len());
    row_wins[..n1].fill(0);
    row_losses[..n1].fill(0);
    col_wins[..n2].fill(0);
    col_losses[..n2].fill(0);

    let cmp = RightCensoredComparator::new();
    let mut tier_wins = [0usize; 2];
    let mut tier_losses = [0usize; 2];
    for i in 0..n1 {
        for j in 0..n2 {
            match compare_simulated_pair(
                &cmp,
                trt_times[i],
                trt_events[i],
                ctrl_times[j],
                ctrl_events[j],
            ) {
                (ComparisonResult::Win, k) => {
                    tier_wins[k] += 1;
                    row_wins[i] += 1;
                    col_wins[j] += 1;
                }
                (ComparisonResult::Loss, k) => {
                    tier_losses[k] += 1;
                    row_losses[i] += 1;
                    col_losses[j] += 1;
                }
                (ComparisonResult::Tie, _) => {}
            }
        }
    }

    let tiers = build_two_tier_counts(n1, n2, tier_wins, tier_losses);
    let marginals = SubjectMarginals {
        row_wins: &row_wins[..n1],
        row_losses: &row_losses[..n1],
        col_wins: &col_wins[..n2],
        col_losses: &col_losses[..n2],
    };
    let w_sum = tier_wins[0] + tier_wins[1];
    let l_sum = tier_losses[0] + tier_losses[1];
    (
        compute_unmatched_u_statistics(n1, n2, w_sum, l_sum, &marginals, sig_alpha),
        tiers,
    )
}

#[verified_engine::verified]
pub(super) fn tie_fallback_summary(n_arm: usize) -> super::u_statistic::UnmatchedWinRatioSummary {
    let pairs = n_arm * n_arm;
    super::u_statistic::UnmatchedWinRatioSummary {
        n_treatment: n_arm,
        n_control: n_arm,
        total_pairs: pairs,
        wins: 0,
        losses: 0,
        ties: pairs,
        win_ratio: 1.0,
        win_odds: 1.0,
        net_benefit: 0.0,
        ci_low: 1.0,
        ci_high: 1.0,
        log_wr_se: 0.0,
        z_score: 0.0,
        p_value: 1.0,
    }
}
