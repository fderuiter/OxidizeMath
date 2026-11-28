//! Deterministic, zero-allocation Monte Carlo power sweeps across arm sizes for
//! [`super::simulation::SimulationParams`].

use crate::applied::clinical_trials::types::ClinicalTrialError;
use math_memory::{ArenaLayout, ArenaRegion, OwnedByteArena};
use oxidize_core::rng::OxidizeRng;

use super::censoring::TierCount;
use super::simulation::{
    SimulationParams, evaluate_simulated_cohorts_into, tie_fallback_summary, win_ratio_parameter,
};
use super::u_statistic::UnmatchedWinRatioSummary;

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

#[derive(Copy, Clone, Debug)]
struct PowerSweepRegions {
    trt_times: ArenaRegion<[f64; 2]>,
    trt_events: ArenaRegion<[u8; 2]>,
    ctrl_times: ArenaRegion<[f64; 2]>,
    ctrl_events: ArenaRegion<[u8; 2]>,
    row_wins: ArenaRegion<u32>,
    row_losses: ArenaRegion<u32>,
    col_wins: ArenaRegion<u32>,
    col_losses: ArenaRegion<u32>,
    points_n: ArenaRegion<usize>,
    points_metrics: ArenaRegion<[f64; 7]>,
}

fn build_power_sweep_arena(
    max_arm: usize,
    max_steps: usize,
) -> (OwnedByteArena, PowerSweepRegions) {
    let mut l = ArenaLayout::new();
    let regs = PowerSweepRegions {
        trt_times: l.reserve::<[f64; 2]>(max_arm).expect("trt_times"),
        trt_events: l.reserve::<[u8; 2]>(max_arm).expect("trt_events"),
        ctrl_times: l.reserve::<[f64; 2]>(max_arm).expect("ctrl_times"),
        ctrl_events: l.reserve::<[u8; 2]>(max_arm).expect("ctrl_events"),
        row_wins: l.reserve::<u32>(max_arm).expect("row_wins"),
        row_losses: l.reserve::<u32>(max_arm).expect("row_losses"),
        col_wins: l.reserve::<u32>(max_arm).expect("col_wins"),
        col_losses: l.reserve::<u32>(max_arm).expect("col_losses"),
        points_n: l.reserve::<usize>(max_steps).expect("points_n"),
        points_metrics: l.reserve::<[f64; 7]>(max_steps).expect("points_metrics"),
    };
    (OwnedByteArena::from_layout(&l), regs)
}

/// Preallocated buffers backed by one [`OwnedByteArena`] enabling zero-allocation execution under `VerifiedAllocator`.
#[derive(Debug, Clone)]
pub struct PowerSweepWorkspace {
    arena: OwnedByteArena,
    regions: PowerSweepRegions,
    max_arm: usize,
    max_steps: usize,
}

impl PowerSweepWorkspace {
    /// Allocates buffers sized for up to `max_arm` subjects per arm and `max_steps` grid points.
    #[must_use]
    pub fn with_capacity(max_arm: usize, max_steps: usize) -> Self {
        let (arena, regions) = build_power_sweep_arena(max_arm, max_steps);
        Self {
            arena,
            regions,
            max_arm,
            max_steps,
        }
    }

    /// Reads the `idx`-th computed [`SampleSizePowerPoint`] from the arena.
    #[must_use]
    pub fn point_at(&self, idx: usize) -> SampleSizePowerPoint {
        let n_per_arm = self.arena.region_slice(self.regions.points_n)[idx];
        let m = self.arena.region_slice(self.regions.points_metrics)[idx];
        SampleSizePowerPoint {
            n_per_arm,
            empirical_power: m[0],
            mean_win_ratio: m[1],
            tier1_win_fraction: m[2],
            tier1_loss_fraction: m[3],
            tier2_win_fraction: m[4],
            tier2_loss_fraction: m[5],
            tie_fraction: m[6],
        }
    }

    /// Collects the first `count` computed [`SampleSizePowerPoint`]s into an owned `Vec`.
    #[must_use]
    pub fn collect_points(&self, count: usize) -> Vec<SampleSizePowerPoint> {
        let cap = count.min(self.max_steps);
        (0..cap).map(|idx| self.point_at(idx)).collect()
    }

    fn set_point(&mut self, idx: usize, pt: SampleSizePowerPoint) {
        self.arena.region_slice_mut(self.regions.points_n)[idx] = pt.n_per_arm;
        self.arena.region_slice_mut(self.regions.points_metrics)[idx] = [
            pt.empirical_power,
            pt.mean_win_ratio,
            pt.tier1_win_fraction,
            pt.tier1_loss_fraction,
            pt.tier2_win_fraction,
            pt.tier2_loss_fraction,
            pt.tie_fraction,
        ];
    }
}

impl Default for PowerSweepWorkspace {
    fn default() -> Self {
        Self::with_capacity(MAX_ARM_SIZE, MAX_SWEEP_STEPS)
    }
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
    if workspace.max_arm < n_per_arm {
        return Err(ClinicalTrialError::InvalidData(
            "PowerSweepWorkspace arm capacity is smaller than n_per_arm".to_string(),
        ));
    }
    let trt_model = params.treatment_model();
    let ctrl_model = params.control_model();
    let regs = workspace.regions;
    let mut cursor = workspace.arena.raw_cursor();
    let trt_times = &mut cursor.region_mut(regs.trt_times)[..n_per_arm];
    let trt_events = &mut cursor.region_mut(regs.trt_events)[..n_per_arm];
    let ctrl_times = &mut cursor.region_mut(regs.ctrl_times)[..n_per_arm];
    let ctrl_events = &mut cursor.region_mut(regs.ctrl_events)[..n_per_arm];
    let row_wins = &mut cursor.region_mut(regs.row_wins)[..n_per_arm];
    let row_losses = &mut cursor.region_mut(regs.row_losses)[..n_per_arm];
    let col_wins = &mut cursor.region_mut(regs.col_wins)[..n_per_arm];
    let col_losses = &mut cursor.region_mut(regs.col_losses)[..n_per_arm];

    for i in 0..n_per_arm {
        let s = trt_model.try_sample_censored_subject(rng, censor_horizon, dropout_rate)?;
        trt_times[i] = [s.fatal.time.as_f64(), s.nonfatal.time.as_f64()];
        trt_events[i] = [
            u8::from(s.fatal.event_occurred),
            u8::from(s.nonfatal.event_occurred),
        ];
    }
    for i in 0..n_per_arm {
        let s = ctrl_model.try_sample_censored_subject(rng, censor_horizon, dropout_rate)?;
        ctrl_times[i] = [s.fatal.time.as_f64(), s.nonfatal.time.as_f64()];
        ctrl_events[i] = [
            u8::from(s.fatal.event_occurred),
            u8::from(s.nonfatal.event_occurred),
        ];
    }
    Ok(evaluate_simulated_cohorts_into(
        trt_times,
        trt_events,
        ctrl_times,
        ctrl_events,
        row_wins,
        row_losses,
        col_wins,
        col_losses,
        sig_alpha,
    ))
}

type SweepStepResult = Result<
    (
        SampleSizePowerPoint,
        Option<UnmatchedWinRatioSummary>,
        [TierCount; 2],
    ),
    ClinicalTrialError,
>;

fn sweep_single_arm_size(
    params: &SimulationParams,
    cfg: &PowerSweepConfig,
    n_arm: usize,
    rng: &mut OxidizeRng,
    workspace: &mut PowerSweepWorkspace,
) -> SweepStepResult {
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
        win_ratio_parameter(params)
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
    if workspace.max_arm < cfg.n_max || workspace.max_steps < cfg.n_steps {
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
        workspace.set_point(step_idx, pt);
        if let Some(s) = last_sum {
            rep_summary = Some(s);
            rep_tiers = last_tiers;
        }
    }

    let summary = rep_summary.unwrap_or_else(|| tie_fallback_summary(cfg.n_max));
    Ok((cfg.n_steps, summary, rep_tiers))
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
    Ok(PowerSweepReport {
        theoretical_uncensored_wr: win_ratio_parameter(params),
        representative_summary,
        representative_tiers,
        points: workspace.collect_points(count),
    })
}
