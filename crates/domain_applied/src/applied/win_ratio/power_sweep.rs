//! Deterministic, zero-allocation Monte Carlo power sweeps across arm sizes for
//! [`super::simulation::SimulationParams`].

use crate::applied::clinical_trials::types::ClinicalTrialError;
use oxidize_core::rng::OxidizeRng;

use super::censoring::{RightCensoredComparator, TierCount};
use super::pair_comparison::ComparisonResult;
use super::simulation::SimulationParams;
use super::u_statistic::{UnmatchedWinRatioSummary, compute_unmatched_u_statistics};

pub use super::power_sweep_types::{
    MAX_ARM_SIZE, MAX_SWEEP_REPLICATES, MAX_SWEEP_STEPS, PowerSweepConfig, PowerSweepReport,
    SampleSizePowerPoint,
};

/// Preallocated buffers backed by [`math_memory::OwnedByteArena`] enabling zero-allocation
/// execution under `VerifiedAllocator`.
#[derive(Debug, Clone)]
pub struct PowerSweepWorkspace {
    arena: math_memory::OwnedByteArena,
    max_arm: usize,
    max_steps: usize,
    point_scalars_reg: math_memory::ArenaRegion<[f64; 7]>,
    point_n_arm_reg: math_memory::ArenaRegion<usize>,
}

impl PowerSweepWorkspace {
    /// Allocates buffers sized for up to `max_arm` subjects per arm and `max_steps` grid points
    /// inside a single [`math_memory::OwnedByteArena`].
    ///
    /// Panics only if arena layout or allocation fails; see [`Self::try_with_capacity`].
    #[must_use]
    pub fn with_capacity(max_arm: usize, max_steps: usize) -> Self {
        Self::try_with_capacity(max_arm, max_steps)
            .expect("PowerSweepWorkspace::with_capacity allocation failed")
    }

    /// Fallibly allocates buffers sized for up to `max_arm` subjects per arm and `max_steps` grid points.
    pub fn try_with_capacity(
        max_arm: usize,
        max_steps: usize,
    ) -> Result<Self, math_memory::ArenaError> {
        let mut layout = math_memory::ArenaLayout::new();
        let point_scalars_reg = layout.add_slice::<[f64; 7]>(max_steps)?;
        let point_n_arm_reg = layout.add_slice::<usize>(max_steps)?;
        for _ in 0..2 {
            let _ = layout.add_slice::<[f64; 2]>(max_arm)?;
        }
        for _ in 0..4 {
            let _ = layout.add_slice::<u32>(max_arm)?;
        }
        for _ in 0..2 {
            let _ = layout.add_slice::<[u8; 2]>(max_arm)?;
        }
        Ok(Self {
            arena: math_memory::OwnedByteArena::try_with_capacity(layout.total_bytes())?,
            max_arm,
            max_steps,
            point_scalars_reg,
            point_n_arm_reg,
        })
    }

    /// Maximum per-arm cohort capacity supported without reallocation.
    #[must_use]
    pub const fn max_arm_capacity(&self) -> usize {
        self.max_arm
    }

    /// Maximum sweep step capacity supported without reallocation.
    #[must_use]
    pub const fn max_steps_capacity(&self) -> usize {
        self.max_steps
    }

    /// Writes `pt` at `step_idx` inside the arena-backed output curve regions.
    pub fn set_point(&mut self, step_idx: usize, pt: SampleSizePowerPoint) {
        debug_assert!(step_idx < self.max_steps);
        self.arena.region_mut(self.point_n_arm_reg)[step_idx] = pt.n_per_arm;
        self.arena.region_mut(self.point_scalars_reg)[step_idx] = [
            pt.empirical_power,
            pt.mean_win_ratio,
            pt.tier1_win_fraction,
            pt.tier1_loss_fraction,
            pt.tier2_win_fraction,
            pt.tier2_loss_fraction,
            pt.tie_fraction,
        ];
    }

    /// Reads the `step_idx`-th [`SampleSizePowerPoint`] from the arena.
    #[must_use]
    pub fn point_at(&self, step_idx: usize) -> SampleSizePowerPoint {
        debug_assert!(step_idx < self.max_steps);
        let s = self.arena.region(self.point_scalars_reg)[step_idx];
        SampleSizePowerPoint {
            n_per_arm: self.arena.region(self.point_n_arm_reg)[step_idx],
            empirical_power: s[0],
            mean_win_ratio: s[1],
            tier1_win_fraction: s[2],
            tier1_loss_fraction: s[3],
            tier2_win_fraction: s[4],
            tier2_loss_fraction: s[5],
            tie_fraction: s[6],
        }
    }

    /// Materializes the first `count` sweep curve points as a `Vec<SampleSizePowerPoint>`.
    #[must_use]
    pub fn collect_points(&self, count: usize) -> Vec<SampleSizePowerPoint> {
        (0..count.min(self.max_steps))
            .map(|i| self.point_at(i))
            .collect()
    }
}

impl Default for PowerSweepWorkspace {
    fn default() -> Self {
        Self::with_capacity(MAX_ARM_SIZE, MAX_SWEEP_STEPS)
    }
}

#[inline]
fn compare_raw_pair(
    cmp: &RightCensoredComparator,
    trt_t: [f64; 2],
    trt_ev: [u8; 2],
    ctrl_t: [f64; 2],
    ctrl_ev: [u8; 2],
) -> (ComparisonResult, usize) {
    let fatal_res = cmp.compare_raw(trt_t[0], trt_ev[0] != 0, ctrl_t[0], ctrl_ev[0] != 0);
    if fatal_res != ComparisonResult::Tie {
        return (fatal_res, 0);
    }
    let nonfatal_res = cmp.compare_raw(trt_t[1], trt_ev[1] != 0, ctrl_t[1], ctrl_ev[1] != 0);
    (nonfatal_res, 1)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_raw_cohorts_into(
    trt_times: &[[f64; 2]],
    trt_events: &[[u8; 2]],
    ctrl_times: &[[f64; 2]],
    ctrl_events: &[[u8; 2]],
    sig_alpha: f64,
    row_wins: &mut [u32],
    row_losses: &mut [u32],
    col_wins: &mut [u32],
    col_losses: &mut [u32],
) -> (Option<UnmatchedWinRatioSummary>, [TierCount; 2]) {
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
            match compare_raw_pair(
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

    let total_pairs = n1.saturating_mul(n2);
    let rem1 = total_pairs.saturating_sub(tier_wins[0] + tier_losses[0]);
    let rem2 = rem1.saturating_sub(tier_wins[1] + tier_losses[1]);
    let tiers = [0, 1].map(|k| TierCount {
        tier_index: k,
        wins: tier_wins[k],
        losses: tier_losses[k],
        ties_remaining: if k == 0 { rem1 } else { rem2 },
    });
    let marginals = super::u_statistic::SubjectMarginals {
        row_wins: &row_wins[..n1],
        row_losses: &row_losses[..n1],
        col_wins: &col_wins[..n2],
        col_losses: &col_losses[..n2],
    };
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
    if n_per_arm > workspace.max_arm {
        return Err(ClinicalTrialError::InvalidData(
            "PowerSweepWorkspace capacity is smaller than n_per_arm".to_string(),
        ));
    }
    let (trt_model, ctrl_model, cap_arm, cap_steps) = (
        params.treatment_model(),
        params.control_model(),
        workspace.max_arm,
        workspace.max_steps,
    );
    let mut scope = workspace.arena.reborrow();
    let _ = scope
        .retain_slice::<[f64; 7]>(cap_steps)
        .expect("point_scalars");
    let _ = scope.retain_slice::<usize>(cap_steps).expect("point_n_arm");
    let trt_times = scope.retain_slice::<[f64; 2]>(cap_arm).expect("trt_times");
    let ctrl_times = scope.retain_slice::<[f64; 2]>(cap_arm).expect("ctrl_times");
    let row_wins = scope.retain_slice::<u32>(cap_arm).expect("row_wins");
    let row_losses = scope.retain_slice::<u32>(cap_arm).expect("row_losses");
    let col_wins = scope.retain_slice::<u32>(cap_arm).expect("col_wins");
    let col_losses = scope.retain_slice::<u32>(cap_arm).expect("col_losses");
    let trt_events = scope.retain_slice::<[u8; 2]>(cap_arm).expect("trt_events");
    let ctrl_events = scope.retain_slice::<[u8; 2]>(cap_arm).expect("ctrl_events");

    for i in 0..n_per_arm {
        let s = trt_model.try_sample_censored_subject(rng, censor_horizon, dropout_rate)?;
        trt_times[i] = [s.fatal.time.as_f64(), s.nonfatal.time.as_f64()];
        trt_events[i] = [
            u8::from(s.fatal.event_occurred),
            u8::from(s.nonfatal.event_occurred),
        ];
        let c = ctrl_model.try_sample_censored_subject(rng, censor_horizon, dropout_rate)?;
        ctrl_times[i] = [c.fatal.time.as_f64(), c.nonfatal.time.as_f64()];
        ctrl_events[i] = [
            u8::from(c.fatal.event_occurred),
            u8::from(c.nonfatal.event_occurred),
        ];
    }
    Ok(evaluate_raw_cohorts_into(
        &trt_times[..n_per_arm],
        &trt_events[..n_per_arm],
        &ctrl_times[..n_per_arm],
        &ctrl_events[..n_per_arm],
        sig_alpha,
        row_wins,
        row_losses,
        col_wins,
        col_losses,
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
    if workspace.max_arm_capacity() < cfg.n_max || workspace.max_steps_capacity() < cfg.n_steps {
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
    Ok(PowerSweepReport {
        theoretical_uncensored_wr: super::simulation::win_ratio_parameter(params),
        representative_summary,
        representative_tiers,
        points: workspace.collect_points(count),
    })
}
