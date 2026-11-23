//! Deterministic, zero-allocation nonparametric bootstrap confidence bands (Percentile and Efron
//! $\mathrm{BC}_a$ with leave-one-out jackknife acceleration) for the Kaplan-Meier estimator.

use super::{ClinicalTrialError, Observation};
use engine_prob::bootstrap::{BootstrapConfig, BootstrapMethod};
use engine_prob::math::{
    mean_and_sample_variance, sort_total_f64, sorted_quantile, standard_normal_cdf,
    standard_normal_inv_cdf,
};
use engine_prob::workspace::{OwnedSolverWorkspace, SolverWorkspace};
use oxidize_core::rng::OxidizeRng;

/// Point on a Kaplan-Meier survival curve augmented with nonparametric bootstrap uncertainty bands.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BootstrapSurvivalPoint {
    /// Distinct anchor time $t_k$.
    pub time: f64,
    /// Pointwise Kaplan-Meier product-limit estimate $\hat{S}(t_k) \in [0, 1]$.
    pub survival_probability: f64,
    /// Nonparametric bootstrap standard error $\widehat{\mathrm{SE}}_{\mathrm{boot}}(t_k)$.
    pub std_error: f64,
    /// Lower $(1 - \alpha)$ bootstrap confidence bound in `[0, 1]`.
    pub ci_lower: f64,
    /// Upper $(1 - \alpha)$ bootstrap confidence bound in `[0, 1]`.
    pub ci_upper: f64,
    /// Subjects at risk immediately prior to $t_k$.
    pub n_at_risk: usize,
    /// Uncensored events observed at $t_k$.
    pub n_events: usize,
    /// Right-censored observations at $t_k$.
    pub n_censored: usize,
}

/// Minimum `f64` scratch capacity required in [`SolverWorkspace`] for [`kaplan_meier_bootstrap_into`].
#[must_use]
#[verified_engine::verified]
pub const fn workspace_floats_for_km_bootstrap(n_obs: usize, n_resamples: usize) -> usize {
    n_obs.saturating_mul(n_resamples).saturating_add(n_obs)
}

/// Minimum `usize` scratch capacity required in [`SolverWorkspace`] for [`kaplan_meier_bootstrap_into`].
#[must_use]
#[verified_engine::verified]
pub const fn workspace_indices_for_km_bootstrap(n_obs: usize) -> usize {
    n_obs.saturating_mul(3)
}

#[verified_engine::verified]
fn validate_km_bootstrap_inputs(
    n_obs: usize,
    config: &BootstrapConfig,
) -> Result<(), ClinicalTrialError> {
    if n_obs < 2 {
        return Err(ClinicalTrialError::InsufficientSampleSize {
            required: 2,
            actual: n_obs,
        });
    }
    if config.n_resamples < 2 {
        return Err(ClinicalTrialError::InvalidData(
            "Bootstrap n_resamples must be >= 2".to_string(),
        ));
    }
    if !config.confidence_level.is_finite()
        || config.confidence_level <= 0.0
        || config.confidence_level >= 1.0
    {
        return Err(ClinicalTrialError::InvalidData(
            "Bootstrap confidence_level must be finite in (0, 1)".to_string(),
        ));
    }
    Ok(())
}

#[verified_engine::verified]
fn build_sorted_groups(
    observations: &[Observation],
    order: &mut [usize],
    group_id: &mut [usize],
    out: &mut [BootstrapSurvivalPoint],
) -> Result<usize, ClinicalTrialError> {
    for (idx, slot) in order.iter_mut().enumerate() {
        *slot = idx;
    }
    order.sort_unstable_by(|&a, &b| {
        observations[a]
            .time
            .as_f64()
            .total_cmp(&observations[b].time.as_f64())
    });

    let n = order.len();
    let mut i = 0usize;
    let mut m = 0usize;
    let mut n_at_risk = n;
    let mut survival = 1.0_f64;
    while i < n {
        if m >= out.len() {
            return Err(ClinicalTrialError::InvalidData(
                "Output buffer shorter than distinct Kaplan-Meier time points".to_string(),
            ));
        }
        let anchor_t = observations[order[i]].time.as_f64();
        let (mut d_k, mut c_k) = (0usize, 0usize);
        while i < n
            && (observations[order[i]].time.as_f64() - anchor_t).abs()
                < math_commons::registry::TOLERANCE_STANDARD
        {
            group_id[i] = m;
            if observations[order[i]].event_occurred {
                d_k += 1;
            } else {
                c_k += 1;
            }
            i += 1;
        }
        if d_k > 0 && n_at_risk > 0 {
            survival *= 1.0 - (d_k as f64) / (n_at_risk as f64);
        }
        let s_clamped = survival.clamp(0.0, 1.0);
        out[m] = BootstrapSurvivalPoint {
            time: anchor_t,
            survival_probability: s_clamped,
            std_error: 0.0,
            ci_lower: s_clamped,
            ci_upper: s_clamped,
            n_at_risk,
            n_events: d_k,
            n_censored: c_k,
        };
        n_at_risk -= d_k + c_k;
        m += 1;
    }
    Ok(m)
}

struct CohortGroups<'a> {
    observations: &'a [Observation],
    order: &'a [usize],
    group_id: &'a [usize],
    n_groups: usize,
}

#[verified_engine::verified]
fn eval_resample_into_matrix(
    groups: &CohortGroups<'_>,
    counts: &[usize],
    n_resamples: usize,
    col_idx: usize,
    boot_matrix: &mut [f64],
) {
    let n = groups.order.len();
    let mut cursor = 0usize;
    let mut at_risk = n;
    let mut survival = 1.0_f64;
    for g in 0..groups.n_groups {
        let (mut d_g, mut c_g) = (0usize, 0usize);
        while cursor < n && groups.group_id[cursor] == g {
            let mult = counts[cursor];
            if mult > 0 {
                if groups.observations[groups.order[cursor]].event_occurred {
                    d_g += mult;
                } else {
                    c_g += mult;
                }
            }
            cursor += 1;
        }
        if d_g > 0 && at_risk > 0 {
            survival *= 1.0 - (d_g as f64) / (at_risk as f64);
        }
        boot_matrix[math_commons::math_kernel::types::flatten_2d_index(col_idx, g, n_resamples)] =
            survival.clamp(0.0, 1.0);
        at_risk = at_risk.saturating_sub(d_g + c_g);
    }
}

#[verified_engine::verified]
fn eval_single_group_km(groups: &CohortGroups<'_>, omit_pos: usize, target_group: usize) -> f64 {
    let mut at_risk = groups.order.len() - 1;
    let mut survival = 1.0_f64;
    let mut cursor = 0usize;
    for g in 0..=target_group {
        let (mut d_g, mut c_g) = (0usize, 0usize);
        while cursor < groups.order.len() && groups.group_id[cursor] == g {
            if cursor != omit_pos {
                if groups.observations[groups.order[cursor]].event_occurred {
                    d_g += 1;
                } else {
                    c_g += 1;
                }
            }
            cursor += 1;
        }
        if d_g > 0 && at_risk > 0 {
            survival *= 1.0 - (d_g as f64) / (at_risk as f64);
        }
        at_risk = at_risk.saturating_sub(d_g + c_g);
    }
    survival.clamp(0.0, 1.0)
}

#[verified_engine::verified]
fn compute_bca_group_ci(
    theta_hat: f64,
    sorted_reps: &[f64],
    alpha_half: f64,
    groups: &CohortGroups<'_>,
    target_group: usize,
    jack_buf: &mut [f64],
) -> (f64, f64) {
    let b = sorted_reps.len();
    let less = sorted_reps.iter().filter(|&&v| v < theta_hat).count();
    if less == 0 || less == b {
        return (
            sorted_quantile(sorted_reps, alpha_half).clamp(0.0, 1.0),
            sorted_quantile(sorted_reps, 1.0 - alpha_half).clamp(0.0, 1.0),
        );
    }
    let z0 = standard_normal_inv_cdf((less as f64) / (b as f64));
    let n = groups.order.len();
    let mut jack_sum = 0.0;
    for (pos, slot) in jack_buf.iter_mut().enumerate().take(n) {
        let val = eval_single_group_km(groups, pos, target_group);
        *slot = val;
        jack_sum += val;
    }
    let jack_mean = jack_sum / (n as f64);
    let (mut num, mut den) = (0.0, 0.0);
    for &th_i in jack_buf.iter().take(n) {
        let diff = jack_mean - th_i;
        let sq = diff * diff;
        num += diff * sq;
        den += sq;
    }
    let accel = if den > 1e-30 {
        num / (6.0 * den.powf(1.5))
    } else {
        0.0
    };
    let adjust = |z_a: f64| -> f64 {
        let denom = 1.0 - accel * (z0 + z_a);
        if denom.abs() <= 1e-12 {
            0.5
        } else {
            standard_normal_cdf(z0 + (z0 + z_a) / denom)
        }
    };
    let z_lo = standard_normal_inv_cdf(alpha_half);
    let z_hi = standard_normal_inv_cdf(1.0 - alpha_half);
    let lo = sorted_quantile(sorted_reps, adjust(z_lo)).clamp(0.0, 1.0);
    let hi = sorted_quantile(sorted_reps, adjust(z_hi)).clamp(0.0, 1.0);
    (lo.min(hi), lo.max(hi))
}

#[verified_engine::verified]
fn summarize_bootstrap_rows(
    groups: &CohortGroups<'_>,
    config: &BootstrapConfig,
    boot_matrix: &mut [f64],
    jack_buf: &mut [f64],
    out: &mut [BootstrapSurvivalPoint],
) {
    let b = config.n_resamples;
    let alpha_half = 0.5 * (1.0 - config.confidence_level);
    let (mut prev_lo, mut prev_hi) = (1.0_f64, 1.0_f64);
    for g in 0..groups.n_groups {
        let row = &mut boot_matrix[g * b..(g + 1) * b];
        sort_total_f64(row);
        let (_, var) = mean_and_sample_variance(row).unwrap_or((0.0, 0.0));
        out[g].std_error = var.max(0.0).sqrt();
        let (raw_lo, raw_hi) = match config.method {
            BootstrapMethod::Percentile => (
                sorted_quantile(row, alpha_half).clamp(0.0, 1.0),
                sorted_quantile(row, 1.0 - alpha_half).clamp(0.0, 1.0),
            ),
            BootstrapMethod::BCa => compute_bca_group_ci(
                out[g].survival_probability,
                row,
                alpha_half,
                groups,
                g,
                jack_buf,
            ),
        };
        let mono_lo = raw_lo.min(prev_lo);
        let mono_hi = raw_hi.min(prev_hi).max(mono_lo);
        out[g].ci_lower = mono_lo;
        out[g].ci_upper = mono_hi;
        prev_lo = mono_lo;
        prev_hi = mono_hi;
    }
}

/// Computes deterministic nonparametric bootstrap confidence bands (`Percentile` or `BCa`) for
/// the Kaplan-Meier curve into `out` with **zero dynamic heap allocations** over [`SolverWorkspace`].
///
/// Returns the number of distinct anchor time points written to `out`.
///
/// # Example
/// ```
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::clinical_trials::survival_analysis::{
///     BootstrapSurvivalPoint, Observation, kaplan_meier_bootstrap_into,
///     workspace_floats_for_km_bootstrap, workspace_indices_for_km_bootstrap,
/// };
/// use math_explorer::applied::clinical_trials::types::SurvivalTime;
/// use math_explorer::prob::{BootstrapConfig, BootstrapMethod, OwnedSolverWorkspace};
/// use math_explorer::rng::OxidizeRng;
///
/// let obs = [
///     Observation { time: SurvivalTime::new(2.0).unwrap(), event_occurred: true },
///     Observation { time: SurvivalTime::new(3.0).unwrap(), event_occurred: false },
///     Observation { time: SurvivalTime::new(5.0).unwrap(), event_occurred: true },
///     Observation { time: SurvivalTime::new(8.0).unwrap(), event_occurred: true },
/// ];
/// let cfg = BootstrapConfig::new(200, 0.95).with_method(BootstrapMethod::BCa);
/// let mut owned = OwnedSolverWorkspace::with_capacity(
///     workspace_floats_for_km_bootstrap(obs.len(), cfg.n_resamples),
///     workspace_indices_for_km_bootstrap(obs.len()),
/// );
/// let mut out = [BootstrapSurvivalPoint {
///     time: 0.0, survival_probability: 1.0, std_error: 0.0, ci_lower: 1.0, ci_upper: 1.0,
///     n_at_risk: 0, n_events: 0, n_censored: 0,
/// }; 4];
/// let mut rng = OxidizeRng::new(42);
/// let mut ws = owned.as_workspace();
/// let m = kaplan_meier_bootstrap_into(&obs, &cfg, &mut rng, &mut ws, &mut out).unwrap();
/// assert_eq!(m, 4);
/// assert!(out[0].ci_lower <= out[0].ci_upper);
/// ```
#[verified_engine::verified]
pub fn kaplan_meier_bootstrap_into(
    observations: &[Observation],
    config: &BootstrapConfig,
    rng: &mut OxidizeRng,
    workspace: &mut SolverWorkspace<'_>,
    out: &mut [BootstrapSurvivalPoint],
) -> Result<usize, ClinicalTrialError> {
    let n = observations.len();
    validate_km_bootstrap_inputs(n, config)?;
    let b = config.n_resamples;

    let (order, mut idx_rem1) = workspace
        .split_indices(n)
        .map_err(|e| ClinicalTrialError::InvalidData(e.to_string()))?;
    let (group_id, mut idx_rem2) = idx_rem1
        .split_indices(n)
        .map_err(|e| ClinicalTrialError::InvalidData(e.to_string()))?;
    let (counts, mut ws_floats) = idx_rem2
        .split_indices(n)
        .map_err(|e| ClinicalTrialError::InvalidData(e.to_string()))?;

    let m = build_sorted_groups(observations, order, group_id, out)?;
    let (boot_matrix, mut f_rem) = ws_floats
        .split_floats(m * b)
        .map_err(|e| ClinicalTrialError::InvalidData(e.to_string()))?;
    let (jack_buf, _) = f_rem
        .split_floats(n)
        .map_err(|e| ClinicalTrialError::InvalidData(e.to_string()))?;

    let groups = CohortGroups {
        observations,
        order,
        group_id,
        n_groups: m,
    };
    for rep_idx in 0..b {
        counts.fill(0);
        for _ in 0..n {
            counts[rng.next_index(n)] += 1;
        }
        eval_resample_into_matrix(&groups, counts, b, rep_idx, boot_matrix);
    }
    summarize_bootstrap_rows(&groups, config, boot_matrix, jack_buf, out);
    Ok(m)
}

/// Convenience wrapper computing nonparametric bootstrap Kaplan-Meier survival bands (`Percentile` or `BCa`).
#[verified_engine::verified]
pub fn kaplan_meier_bootstrap(
    observations: &[Observation],
    config: &BootstrapConfig,
    rng: &mut OxidizeRng,
) -> Result<Vec<BootstrapSurvivalPoint>, ClinicalTrialError> {
    let n = observations.len();
    validate_km_bootstrap_inputs(n, config)?;
    let mut owned = OwnedSolverWorkspace::with_capacity(
        workspace_floats_for_km_bootstrap(n, config.n_resamples),
        workspace_indices_for_km_bootstrap(n),
    );
    let mut out = vec![
        BootstrapSurvivalPoint {
            time: 0.0,
            survival_probability: 1.0,
            std_error: 0.0,
            ci_lower: 1.0,
            ci_upper: 1.0,
            n_at_risk: 0,
            n_events: 0,
            n_censored: 0,
        };
        n
    ];
    let mut ws = owned.as_workspace();
    let m = kaplan_meier_bootstrap_into(observations, config, rng, &mut ws, &mut out)?;
    out.truncate(m);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::applied::clinical_trials::survival_analysis::kaplan_meier;
    use crate::applied::clinical_trials::types::SurvivalTime;

    fn sample_cohort() -> Vec<Observation> {
        let mk = |t: f64, ev: bool| Observation {
            time: SurvivalTime::new(t).unwrap(),
            event_occurred: ev,
        };
        vec![
            mk(1.5, true),
            mk(2.0, false),
            mk(3.5, true),
            mk(3.5, true),
            mk(5.0, false),
            mk(6.5, true),
            mk(8.0, true),
            mk(10.0, false),
        ]
    }

    #[test]
    #[verified_engine::verified]
    fn test_km_bootstrap_percentile_and_bca_deterministic_and_monotone() {
        let cohort = sample_cohort();
        let km = kaplan_meier(&cohort);

        let cfg_pct = BootstrapConfig::new(400, 0.95);
        let mut rng1 = OxidizeRng::new(0xBEEF_2026);
        let mut rng2 = OxidizeRng::new(0xBEEF_2026);
        let pct1 = kaplan_meier_bootstrap(&cohort, &cfg_pct, &mut rng1).unwrap();
        let pct2 = kaplan_meier_bootstrap(&cohort, &cfg_pct, &mut rng2).unwrap();
        assert_eq!(pct1, pct2);
        assert_eq!(pct1.len(), km.len());

        let cfg_bca = BootstrapConfig::new(400, 0.95).with_method(BootstrapMethod::BCa);
        let mut rng_bca = OxidizeRng::new(0xBEEF_2026);
        let bca = kaplan_meier_bootstrap(&cohort, &cfg_bca, &mut rng_bca).unwrap();
        assert_eq!(bca.len(), km.len());

        for idx in 0..km.len() {
            assert_eq!(pct1[idx].survival_probability, km[idx].survival_probability);
            assert!(pct1[idx].ci_lower >= 0.0 && pct1[idx].ci_upper <= 1.0);
            assert!(pct1[idx].ci_lower <= pct1[idx].ci_upper);
            assert!(bca[idx].ci_lower >= 0.0 && bca[idx].ci_upper <= 1.0);
            assert!(bca[idx].ci_lower <= bca[idx].ci_upper);
            if idx > 0 {
                assert!(pct1[idx].ci_lower <= pct1[idx - 1].ci_lower + 1e-12);
                assert!(pct1[idx].ci_upper <= pct1[idx - 1].ci_upper + 1e-12);
                assert!(bca[idx].ci_lower <= bca[idx - 1].ci_lower + 1e-12);
                assert!(bca[idx].ci_upper <= bca[idx - 1].ci_upper + 1e-12);
            }
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_km_bootstrap_rejects_small_cohort() {
        let single = [Observation {
            time: SurvivalTime::new(1.0).unwrap(),
            event_occurred: true,
        }];
        let mut rng = OxidizeRng::new(1);
        assert!(kaplan_meier_bootstrap(&single, &BootstrapConfig::default(), &mut rng).is_err());
    }
}
