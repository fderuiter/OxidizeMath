//! Deterministic, zero-allocation Monte Carlo power sweeps across arm sizes for
//! [`super::simulation::SimulationParams`].

use crate::applied::clinical_trials::types::ClinicalTrialError;
use oxidize_core::rng::OxidizeRng;

use super::censoring::{RightCensoredComparator, TierCount};
use super::pair_comparison::ComparisonResult;
use super::simulation::SimulationParams;
use super::u_statistic::{
    UnmatchedScratch, UnmatchedWinRatioSummary, compute_unmatched_u_statistics,
};
use super::weibull_sampler::SimulatedSubject;

/// Maximum per-arm sample size permitted in WebAssembly and interactive sweeps.
pub const MAX_ARM_SIZE: usize = 500;

/// Maximum Monte Carlo replicate count permitted per sample-size grid step.
pub const MAX_SWEEP_REPLICATES: usize = 500;

/// Maximum number of sample-size grid steps in a single power sweep.
pub const MAX_SWEEP_STEPS: usize = 16;

/// Configuration for a seeded Monte Carlo composite win-ratio power sweep across arm sizes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerSweepConfig {
    /// Minimum sample size per arm ($n_{\min} \ge 2$).
    pub n_min: usize,
    /// Maximum sample size per arm ($n_{\min} \le n_{\max} \le 500$).
    pub n_max: usize,
    /// Number of evenly spaced sample-size grid points ($1 \le \text{steps} \le 16$).
    pub n_steps: usize,
    /// Number of simulated two-arm trials per grid point ($1 \le R \le 500$).
    pub replicates: usize,
    /// Administrative follow-up censoring horizon $c > 0$.
    pub censor_horizon: f64,
    /// Exponential random loss-to-follow-up hazard rate ($\ge 0$).
    pub dropout_rate: f64,
    /// Two-sided significance level $\alpha \in (0, 1)$ (e.g., `0.05`).
    pub sig_alpha: f64,
}

impl Default for PowerSweepConfig {
    fn default() -> Self {
        Self {
            n_min: 30,
            n_max: 150,
            n_steps: 5,
            replicates: 60,
            censor_horizon: 5.0,
            dropout_rate: 0.05,
            sig_alpha: 0.05,
        }
    }
}

impl PowerSweepConfig {
    /// Validates all sample-size, replicate, censoring, and significance bounds.
    #[verified_engine::verified]
    pub fn validate(&self) -> Result<(), ClinicalTrialError> {
        if self.n_min < 2 || self.n_max < self.n_min || self.n_max > MAX_ARM_SIZE {
            return Err(ClinicalTrialError::InvalidData(format!(
                "Arm sizes must satisfy 2 <= n_min <= n_max <= {MAX_ARM_SIZE}, got {}..={}",
                self.n_min, self.n_max
            )));
        }
        if self.n_steps == 0 || self.n_steps > MAX_SWEEP_STEPS {
            return Err(ClinicalTrialError::InvalidData(format!(
                "n_steps must be in 1..={MAX_SWEEP_STEPS}, got {}",
                self.n_steps
            )));
        }
        if self.replicates == 0 || self.replicates > MAX_SWEEP_REPLICATES {
            return Err(ClinicalTrialError::InvalidData(format!(
                "replicates must be in 1..={MAX_SWEEP_REPLICATES}, got {}",
                self.replicates
            )));
        }
        if !self.censor_horizon.is_finite() || self.censor_horizon <= 0.0 {
            return Err(ClinicalTrialError::InvalidData(
                "censor_horizon must be positive and finite".to_string(),
            ));
        }
        if !self.dropout_rate.is_finite() || self.dropout_rate < 0.0 {
            return Err(ClinicalTrialError::InvalidData(
                "dropout_rate must be non-negative and finite".to_string(),
            ));
        }
        if !self.sig_alpha.is_finite() || self.sig_alpha <= 0.0 || self.sig_alpha >= 1.0 {
            return Err(ClinicalTrialError::InvalidData(
                "sig_alpha must be finite in (0, 1)".to_string(),
            ));
        }
        Ok(())
    }

    /// Computes the `step_idx`-th sample size per arm in `0..self.n_steps`.
    #[verified_engine::verified]
    #[must_use]
    pub fn sample_size_at(&self, step_idx: usize) -> usize {
        if self.n_steps <= 1 {
            return self.n_max;
        }
        let span = self.n_max - self.n_min;
        let numer = span * step_idx + (self.n_steps - 1) / 2;
        self.n_min + numer / (self.n_steps - 1)
    }
}

/// Aggregated Monte Carlo power and tier waterfall metrics at a single per-arm sample size.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SampleSizePowerPoint {
    /// Subjects per arm ($N$).
    pub n_per_arm: usize,
    /// Empirical rejection rate ($\Pr(p < \alpha)$) across Monte Carlo replicates.
    pub empirical_power: f64,
    /// Mean finite win ratio across replicates (geometric mean on log-scale).
    pub mean_win_ratio: f64,
    /// Mean Tier 1 (fatal) win fraction across replicates.
    pub tier1_win_fraction: f64,
    /// Mean Tier 1 (fatal) loss fraction across replicates.
    pub tier1_loss_fraction: f64,
    /// Mean Tier 2 (non-fatal) win fraction across replicates.
    pub tier2_win_fraction: f64,
    /// Mean Tier 2 (non-fatal) loss fraction across replicates.
    pub tier2_loss_fraction: f64,
    /// Mean overall tie fraction across replicates.
    pub tie_fraction: f64,
}

/// Output report from a deterministic `BivariateWeibullModel` power sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerSweepReport {
    /// Theoretical uncensored win ratio $PR_W = \theta^{-\alpha}$.
    pub theoretical_uncensored_wr: f64,
    /// Representative trial U-statistic summary at `n_max`.
    pub representative_summary: UnmatchedWinRatioSummary,
    /// Representative trial tier waterfall (`[Fatal, Non-Fatal]`) at `n_max`.
    pub representative_tiers: [TierCount; 2],
    /// Power-by-$N$ curve points ordered from `n_min` to `n_max`.
    pub points: Vec<SampleSizePowerPoint>,
}

/// Preallocated buffers enabling zero-allocation execution under `VerifiedAllocator`.
#[derive(Debug, Clone)]
pub struct PowerSweepWorkspace {
    /// Preallocated treatment cohort buffer (`len >= MAX_ARM_SIZE`).
    pub treatment_buf: Vec<SimulatedSubject>,
    /// Preallocated control cohort buffer (`len >= MAX_ARM_SIZE`).
    pub control_buf: Vec<SimulatedSubject>,
    /// Preallocated U-statistic row/column/tier scratch buffers.
    pub scratch: UnmatchedScratch,
    /// Preallocated output power-curve slice (`len >= MAX_SWEEP_STEPS`).
    pub points_buf: Vec<SampleSizePowerPoint>,
}

impl PowerSweepWorkspace {
    /// Allocates buffers sized for up to `max_arm` subjects per arm and `max_steps` grid points.
    #[must_use]
    pub fn with_capacity(max_arm: usize, max_steps: usize) -> Self {
        Self {
            treatment_buf: vec![SimulatedSubject::default(); max_arm],
            control_buf: vec![SimulatedSubject::default(); max_arm],
            scratch: UnmatchedScratch::with_capacity(max_arm, max_arm, 2),
            points_buf: vec![SampleSizePowerPoint::default(); max_steps],
        }
    }
}

impl Default for PowerSweepWorkspace {
    fn default() -> Self {
        Self::with_capacity(MAX_ARM_SIZE, MAX_SWEEP_STEPS)
    }
}

#[inline]
fn compare_simulated_pair(
    cmp: &RightCensoredComparator,
    trt: &SimulatedSubject,
    ctrl: &SimulatedSubject,
) -> (ComparisonResult, usize) {
    let fatal_res = cmp.compare_raw(
        trt.fatal.time.as_f64(),
        trt.fatal.event_occurred,
        ctrl.fatal.time.as_f64(),
        ctrl.fatal.event_occurred,
    );
    if fatal_res != ComparisonResult::Tie {
        return (fatal_res, 0);
    }
    let nonfatal_res = cmp.compare_raw(
        trt.nonfatal.time.as_f64(),
        trt.nonfatal.event_occurred,
        ctrl.nonfatal.time.as_f64(),
        ctrl.nonfatal.event_occurred,
    );
    (nonfatal_res, 1)
}

fn evaluate_simulated_cohorts_into(
    treatment: &[SimulatedSubject],
    control: &[SimulatedSubject],
    sig_alpha: f64,
    scratch: &mut UnmatchedScratch,
) -> (Option<UnmatchedWinRatioSummary>, [TierCount; 2]) {
    let (n1, n2) = (treatment.len(), control.len());
    scratch.row_wins[..n1].fill(0);
    scratch.row_losses[..n1].fill(0);
    scratch.col_wins[..n2].fill(0);
    scratch.col_losses[..n2].fill(0);

    let cmp = RightCensoredComparator::new();
    let mut tier_wins = [0usize; 2];
    let mut tier_losses = [0usize; 2];
    for (i, trt) in treatment.iter().enumerate() {
        for (j, ctrl) in control.iter().enumerate() {
            match compare_simulated_pair(&cmp, trt, ctrl) {
                (ComparisonResult::Win, k) => {
                    tier_wins[k] += 1;
                    scratch.row_wins[i] += 1;
                    scratch.col_wins[j] += 1;
                }
                (ComparisonResult::Loss, k) => {
                    tier_losses[k] += 1;
                    scratch.row_losses[i] += 1;
                    scratch.col_losses[j] += 1;
                }
                (ComparisonResult::Tie, _) => {}
            }
        }
    }

    let total_pairs = n1.saturating_mul(n2);
    let rem1 = total_pairs.saturating_sub(tier_wins[0] + tier_losses[0]);
    let rem2 = rem1.saturating_sub(tier_wins[1] + tier_losses[1]);
    let tiers = [
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
    ];
    let marginals = scratch.marginals(n1, n2);
    let summary = compute_unmatched_u_statistics(
        n1,
        n2,
        tier_wins[0] + tier_wins[1],
        tier_losses[0] + tier_losses[1],
        &marginals,
        sig_alpha,
    );
    (summary, tiers)
}

/// Evaluates a single simulated two-arm trial of size `n_per_arm` in-place with zero heap allocation.
#[verified_engine::verified]
pub fn simulate_weibull_trial_into(
    params: &SimulationParams,
    n_per_arm: usize,
    censor_horizon: f64,
    dropout_rate: f64,
    sig_alpha: f64,
    rng: &mut OxidizeRng,
    workspace: &mut PowerSweepWorkspace,
) -> Result<(Option<UnmatchedWinRatioSummary>, [TierCount; 2]), ClinicalTrialError> {
    params.validate()?;
    let trt_model = params.treatment_model();
    let ctrl_model = params.control_model();
    for slot in workspace.treatment_buf.iter_mut().take(n_per_arm) {
        *slot = trt_model.try_sample_censored_subject(rng, censor_horizon, dropout_rate)?;
    }
    for slot in workspace.control_buf.iter_mut().take(n_per_arm) {
        *slot = ctrl_model.try_sample_censored_subject(rng, censor_horizon, dropout_rate)?;
    }
    Ok(evaluate_simulated_cohorts_into(
        &workspace.treatment_buf[..n_per_arm],
        &workspace.control_buf[..n_per_arm],
        sig_alpha,
        &mut workspace.scratch,
    ))
}

fn sweep_single_arm_size(
    params: &SimulationParams,
    cfg: &PowerSweepConfig,
    n_arm: usize,
    rng: &mut OxidizeRng,
    workspace: &mut PowerSweepWorkspace,
) -> Result<
    (
        SampleSizePowerPoint,
        Option<UnmatchedWinRatioSummary>,
        [TierCount; 2],
    ),
    ClinicalTrialError,
> {
    let (mut rejections, mut sum_ln_wr, mut finite_cnt) = (0usize, 0.0, 0usize);
    let mut frac_acc = [0.0_f64; 5];
    let mut last_summary = None;
    let mut last_tiers = [TierCount::default(); 2];
    let inv_pairs = 1.0 / ((n_arm * n_arm) as f64);

    for _ in 0..cfg.replicates {
        let (summary_opt, tiers) = simulate_weibull_trial_into(
            params,
            n_arm,
            cfg.censor_horizon,
            cfg.dropout_rate,
            cfg.sig_alpha,
            rng,
            workspace,
        )?;
        frac_acc[0] += (tiers[0].wins as f64) * inv_pairs;
        frac_acc[1] += (tiers[0].losses as f64) * inv_pairs;
        frac_acc[2] += (tiers[1].wins as f64) * inv_pairs;
        frac_acc[3] += (tiers[1].losses as f64) * inv_pairs;
        frac_acc[4] += (tiers[1].ties_remaining as f64) * inv_pairs;
        if let Some(s) = summary_opt {
            rejections += usize::from(s.p_value < cfg.sig_alpha);
            if s.win_ratio.is_finite() && s.win_ratio > 0.0 {
                sum_ln_wr += s.win_ratio.ln();
                finite_cnt += 1;
            }
            last_summary = Some(s);
        }
        last_tiers = tiers;
    }

    let inv_rep = 1.0 / (cfg.replicates as f64);
    let mean_win_ratio = if finite_cnt > 0 {
        (sum_ln_wr / (finite_cnt as f64)).exp()
    } else {
        super::simulation::win_ratio_parameter(params)
    };
    let pt = SampleSizePowerPoint {
        n_per_arm: n_arm,
        empirical_power: (rejections as f64) * inv_rep,
        mean_win_ratio,
        tier1_win_fraction: frac_acc[0] * inv_rep,
        tier1_loss_fraction: frac_acc[1] * inv_rep,
        tier2_win_fraction: frac_acc[2] * inv_rep,
        tier2_loss_fraction: frac_acc[3] * inv_rep,
        tie_fraction: frac_acc[4] * inv_rep,
    };
    Ok((pt, last_summary, last_tiers))
}

/// Executes a deterministic Monte Carlo power sweep into preallocated `workspace` buffers
/// with zero dynamic heap allocations, returning the populated step count and representative trial.
#[verified_engine::verified]
pub fn run_power_sweep_into(
    params: &SimulationParams,
    cfg: &PowerSweepConfig,
    seed: u64,
    workspace: &mut PowerSweepWorkspace,
) -> Result<(usize, UnmatchedWinRatioSummary, [TierCount; 2]), ClinicalTrialError> {
    params.validate()?;
    cfg.validate()?;
    if workspace.treatment_buf.len() < cfg.n_max
        || workspace.control_buf.len() < cfg.n_max
        || workspace.scratch.row_wins.len() < cfg.n_max
        || workspace.scratch.col_wins.len() < cfg.n_max
        || workspace.points_buf.len() < cfg.n_steps
    {
        return Err(ClinicalTrialError::InvalidData(
            "PowerSweepWorkspace capacity is smaller than requested sweep bounds".to_string(),
        ));
    }

    let mut rng = OxidizeRng::new(seed);
    let mut rep_summary = None;
    let mut rep_tiers = [TierCount::default(); 2];
    for step_idx in 0..cfg.n_steps {
        let n_arm = cfg.sample_size_at(step_idx);
        let (pt, last_sum, last_tiers) =
            sweep_single_arm_size(params, cfg, n_arm, &mut rng, workspace)?;
        workspace.points_buf[step_idx] = pt;
        if let Some(s) = last_sum {
            rep_summary = Some(s);
            rep_tiers = last_tiers;
        }
    }

    let summary = rep_summary.unwrap_or_else(|| tie_fallback_summary(cfg.n_max));
    Ok((cfg.n_steps, summary, rep_tiers))
}

#[verified_engine::verified]
fn tie_fallback_summary(n_arm: usize) -> UnmatchedWinRatioSummary {
    let pairs = n_arm * n_arm;
    UnmatchedWinRatioSummary {
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

/// Preallocates a [`PowerSweepWorkspace`] and executes a seeded `BivariateWeibullModel` power sweep.
#[verified_engine::verified]
pub fn run_weibull_power_sweep(
    params: &SimulationParams,
    cfg: &PowerSweepConfig,
    seed: u64,
) -> Result<PowerSweepReport, ClinicalTrialError> {
    cfg.validate()?;
    let mut workspace = PowerSweepWorkspace::with_capacity(cfg.n_max, cfg.n_steps);
    let (count, representative_summary, representative_tiers) =
        run_power_sweep_into(params, cfg, seed, &mut workspace)?;
    workspace.points_buf.truncate(count);
    Ok(PowerSweepReport {
        theoretical_uncensored_wr: super::simulation::win_ratio_parameter(params),
        representative_summary,
        representative_tiers,
        points: workspace.points_buf,
    })
}
