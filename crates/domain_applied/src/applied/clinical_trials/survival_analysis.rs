//! Nonparametric Kaplan-Meier survival analysis, bootstrap confidence bands, and hazard ratios.

use super::types::{ClinicalTrialError, SurvivalTime};
use oxidize_core::rng::OxidizeRng;
use pure_math::pure_math::statistics::bootstrap::BootstrapMethod;
use pure_math::pure_math::statistics::monte_carlo::{
    normal_cdf_fast, normal_inv_cdf_fast, two_sided_normal_z,
};
use rand::RngCore;

/// Represents a single subject's outcome in a survival analysis study.
#[derive(Debug, Clone, Copy)]
pub struct Observation {
    /// The time at which the event occurred or the subject was censored.
    pub time: SurvivalTime,
    /// Indicates whether the event of interest (e.g., death, relapse) occurred.
    /// * `true`: Event occurred.
    /// * `false`: Censored (e.g., lost to follow-up, study ended).
    pub event_occurred: bool,
}

/// A point on the Kaplan-Meier survival curve.
///
/// Represents the state of the cohort at a specific time $t$.
#[derive(Debug, Clone)]
pub struct TimePoint {
    /// The time $t$ at which events occurred.
    pub time: f64,
    /// The estimated survival probability $S(t)$.
    pub survival_probability: f64,
    /// Number of subjects at risk just before time $t$.
    pub n_at_risk: usize,
    /// Number of events that occurred at time $t$.
    pub n_events: usize,
    /// Number of subjects censored at time $t$.
    pub n_censored: usize,
}

/// Kaplan-Meier survival point augmented with nonparametric bootstrap confidence bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BootstrapSurvivalPoint {
    /// Anchor event or censoring time $t$.
    pub time: f64,
    /// Point-estimate survival probability $\hat{S}(t) \in [0, 1]$.
    pub survival_probability: f64,
    /// Nonparametric bootstrap standard error of $\hat{S}(t)$.
    pub standard_error: f64,
    /// Lower confidence bound clamped to `[0.0, 1.0]`.
    pub ci_lower: f64,
    /// Upper confidence bound clamped to `[0.0, 1.0]`.
    pub ci_upper: f64,
    /// Number of subjects at risk just prior to $t$ in the observed cohort.
    pub n_at_risk: usize,
    /// Number of observed events at $t$.
    pub n_events: usize,
    /// Number of censored observations at $t$.
    pub n_censored: usize,
}

/// Computes the Kaplan-Meier survival curve (Product-Limit Estimator).
///
/// $$ \hat{S}(t) = \prod_{t_i \le t} \left(1 - \frac{d_i}{n_i}\right) $$
///
/// # Examples
///
/// ```
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::clinical_trials::survival_analysis::{kaplan_meier, Observation};
/// use math_explorer::applied::clinical_trials::types::SurvivalTime;
///
/// let obs = vec![
///     Observation { time: SurvivalTime::new(2.0).unwrap(), event_occurred: true },
///     Observation { time: SurvivalTime::new(2.0).unwrap(), event_occurred: true },
///     Observation { time: SurvivalTime::new(3.0).unwrap(), event_occurred: false },
///     Observation { time: SurvivalTime::new(4.0).unwrap(), event_occurred: true },
/// ];
///
/// let curve = kaplan_meier(&obs);
/// assert_eq!(curve.len(), 3);
/// assert_eq!(curve[0].survival_probability, 0.5);
/// assert_eq!(curve[1].survival_probability, 0.5);
/// assert_eq!(curve[2].survival_probability, 0.0);
/// ```
#[verified_engine::verified]
pub fn kaplan_meier(observations: &[Observation]) -> Vec<TimePoint> {
    let mut obs = observations.to_vec();
    obs.sort_unstable_by(|a, b| a.time.as_f64().total_cmp(&b.time.as_f64()));

    let mut curve = Vec::new();
    let mut current_survival = 1.0;
    let mut i = 0;
    let mut n_at_risk = obs.len();

    while i < obs.len() {
        let t = obs[i].time.as_f64();
        let mut n_events = usize::from(obs[i].event_occurred);
        let mut n_censored = usize::from(!obs[i].event_occurred);
        i += 1;

        while i < obs.len()
            && (obs[i].time.as_f64() - t).abs() < math_commons::registry::TOLERANCE_STANDARD
        {
            if obs[i].event_occurred {
                n_events += 1;
            } else {
                n_censored += 1;
            }
            i += 1;
        }

        if n_events > 0 {
            current_survival *= 1.0 - (n_events as f64 / n_at_risk as f64);
        }

        curve.push(TimePoint {
            time: t,
            survival_probability: current_survival,
            n_at_risk,
            n_events,
            n_censored,
        });
        n_at_risk -= n_events + n_censored;
    }

    curve
}

/// Computes nonparametric Efron (1981) bootstrap confidence bands (`Percentile` or `BCa`)
/// for the Kaplan-Meier survival curve over preallocated resample buffers.
///
/// # Errors
/// Returns [`ClinicalTrialError`] if fewer than 2 observations are supplied, `n_replicates == 0`,
/// or `confidence_level` lies outside `(0.0, 1.0)`.
#[verified_engine::verified]
pub fn kaplan_meier_bootstrap(
    observations: &[Observation],
    n_replicates: usize,
    confidence_level: f64,
    method: BootstrapMethod,
    rng: &mut OxidizeRng,
) -> Result<Vec<BootstrapSurvivalPoint>, ClinicalTrialError> {
    let n = observations.len();
    if n < 2 {
        return Err(ClinicalTrialError::InsufficientSampleSize {
            required: 2,
            actual: n,
        });
    }
    if n_replicates == 0 {
        return Err(ClinicalTrialError::InvalidData(
            "Bootstrap replicate count must be positive".to_string(),
        ));
    }
    let z_upper = two_sided_normal_z(confidence_level)
        .map_err(|e| ClinicalTrialError::InvalidData(e.to_string()))?;

    let base_curve = kaplan_meier(observations);
    let n_times = base_curve.len();
    let mut resampled = vec![observations[0]; n];
    let mut grid = vec![0.0_f64; n_times * n_replicates];
    let mut curve_eval = vec![0.0_f64; n_times];

    for rep in 0..n_replicates {
        for slot in &mut resampled {
            let idx = draw_index(rng, n);
            *slot = observations[idx];
        }
        evaluate_sorted_km_at_times(&mut resampled, &base_curve, &mut curve_eval);
        for (row, &s_val) in grid.chunks_exact_mut(n_replicates).zip(&curve_eval) {
            row[rep] = s_val;
        }
    }

    let mut jack_accels = vec![0.0_f64; n_times];
    if method == BootstrapMethod::BCa {
        compute_survival_jackknife_accels(observations, &base_curve, &mut jack_accels);
    }

    let alpha_half = 0.5 * (1.0 - confidence_level);
    let mut out = Vec::with_capacity(n_times);
    for (k, (tp, row)) in base_curve
        .iter()
        .zip(grid.chunks_exact_mut(n_replicates))
        .enumerate()
    {
        row.sort_unstable_by(f64::total_cmp);
        let mean = row.iter().sum::<f64>() / (n_replicates as f64);
        let se = if n_replicates > 1 {
            let ss: f64 = row.iter().map(|&v| (v - mean) * (v - mean)).sum();
            (ss / ((n_replicates - 1) as f64)).sqrt()
        } else {
            0.0
        };
        let (ci_lower, ci_upper) = match method {
            BootstrapMethod::Percentile => (
                quantile_sorted(row, alpha_half),
                quantile_sorted(row, 1.0 - alpha_half),
            ),
            BootstrapMethod::BCa => {
                bca_survival_bounds(row, tp.survival_probability, z_upper, jack_accels[k])
            }
        };
        out.push(BootstrapSurvivalPoint {
            time: tp.time,
            survival_probability: tp.survival_probability,
            standard_error: se,
            ci_lower: ci_lower.clamp(0.0, 1.0),
            ci_upper: ci_upper.clamp(0.0, 1.0),
            n_at_risk: tp.n_at_risk,
            n_events: tp.n_events,
            n_censored: tp.n_censored,
        });
    }
    Ok(out)
}

#[verified_engine::verified]
fn evaluate_sorted_km_at_times(
    sample: &mut [Observation],
    anchors: &[TimePoint],
    out_surv: &mut [f64],
) {
    sample.sort_unstable_by(|a, b| a.time.as_f64().total_cmp(&b.time.as_f64()));
    let tol = math_commons::registry::TOLERANCE_STANDARD;
    let mut i = 0_usize;
    let mut n_at_risk = sample.len();
    let mut surv = 1.0_f64;

    for (tp, slot) in anchors.iter().zip(out_surv.iter_mut()) {
        while i < sample.len() && sample[i].time.as_f64() <= tp.time + tol {
            let t_grp = sample[i].time.as_f64();
            let mut d_i = 0_usize;
            let mut c_i = 0_usize;
            while i < sample.len() && (sample[i].time.as_f64() - t_grp).abs() < tol {
                if sample[i].event_occurred {
                    d_i += 1;
                } else {
                    c_i += 1;
                }
                i += 1;
            }
            if d_i > 0 && n_at_risk > 0 {
                surv *= 1.0 - (d_i as f64) / (n_at_risk as f64);
            }
            n_at_risk = n_at_risk.saturating_sub(d_i + c_i);
        }
        *slot = surv.clamp(0.0, 1.0);
    }
}

#[verified_engine::verified]
fn compute_survival_jackknife_accels(
    observations: &[Observation],
    anchors: &[TimePoint],
    accels: &mut [f64],
) {
    let n = observations.len();
    let k_len = anchors.len();
    let mut loo = vec![observations[0]; n - 1];
    let mut eval_buf = vec![0.0_f64; k_len];
    let mut jack_grid = vec![0.0_f64; k_len * n];

    for drop_idx in 0..n {
        loo[..drop_idx].copy_from_slice(&observations[..drop_idx]);
        loo[drop_idx..].copy_from_slice(&observations[drop_idx + 1..]);
        evaluate_sorted_km_at_times(&mut loo, anchors, &mut eval_buf);
        for (row, &val) in jack_grid.chunks_exact_mut(n).zip(&eval_buf) {
            row[drop_idx] = val;
        }
    }

    for (row, accel_slot) in jack_grid.chunks_exact(n).zip(accels.iter_mut()) {
        let mean = row.iter().sum::<f64>() / (n as f64);
        let mut s2 = 0.0;
        let mut s3 = 0.0;
        for &v in row {
            let d = mean - v;
            let d2 = d * d;
            s2 += d2;
            s3 += d2 * d;
        }
        let denom = 6.0 * s2.powf(1.5);
        *accel_slot = if denom > 1e-30 { s3 / denom } else { 0.0 };
    }
}

#[verified_engine::verified]
fn bca_survival_bounds(
    sorted_reps: &[f64],
    point_est: f64,
    z_upper: f64,
    accel: f64,
) -> (f64, f64) {
    let b = sorted_reps.len();
    if sorted_reps[0] == sorted_reps[b - 1] {
        return (point_est, point_est);
    }
    let less = sorted_reps.iter().filter(|&&v| v < point_est).count();
    let eq = sorted_reps.iter().filter(|&&v| v == point_est).count();
    let min_p = 0.5 / (b as f64);
    let prop = ((less as f64) + 0.5 * (eq as f64)) / (b as f64);
    let z0 = normal_inv_cdf_fast(prop.clamp(min_p, 1.0 - min_p));

    let adj = |z_a: f64| -> f64 {
        let num = z0 + z_a;
        let den = 1.0 - accel * num;
        if den.abs() <= 1e-12 {
            if num >= 0.0 { 1.0 } else { 0.0 }
        } else {
            normal_cdf_fast(z0 + num / den).clamp(0.0, 1.0)
        }
    };
    (
        quantile_sorted(sorted_reps, adj(-z_upper)),
        quantile_sorted(sorted_reps, adj(z_upper)),
    )
}

#[inline]
#[verified_engine::verified]
fn draw_index(rng: &mut OxidizeRng, upper: usize) -> usize {
    let bound = upper as u64;
    let zone = u64::MAX - (u64::MAX % bound);
    loop {
        let r = rng.next_u64();
        if r < zone {
            return (r % bound) as usize;
        }
    }
}

#[inline]
#[verified_engine::verified]
fn quantile_sorted(sorted: &[f64], prob: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = prob.clamp(0.0, 1.0) * ((sorted.len() - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (pos - (lo as f64)) * (sorted[hi] - sorted[lo])
}

/// Calculates a simple Hazard Ratio estimate based on incidence density.
/// HR = (Events1 / TotalTime1) / (Events2 / TotalTime2)
#[verified_engine::verified]
pub fn try_estimate_hazard_ratio(
    group1: &[Observation],
    group2: &[Observation],
) -> Result<f64, ClinicalTrialError> {
    let process_group = |group: &[Observation]| -> Result<(f64, f64), ClinicalTrialError> {
        let mut events = 0.0;
        let mut time = 0.0;
        for obs in group {
            if obs.event_occurred {
                events += 1.0;
            }
            time += obs.time.as_f64();
        }
        Ok((events, time))
    };

    let (events1, time1) = process_group(group1)?;
    let (events2, time2) = process_group(group2)?;

    if time1 <= 0.0 {
        return Err(ClinicalTrialError::InvalidData(
            "Group 1 total time is zero or negative".to_string(),
        ));
    }
    if time2 <= 0.0 {
        return Err(ClinicalTrialError::InvalidData(
            "Group 2 total time is zero or negative".to_string(),
        ));
    }
    if events2 == 0.0 {
        return Err(ClinicalTrialError::StatisticalError(
            "No events in Group 2 (infinite Hazard Ratio)".to_string(),
        ));
    }

    Ok((events1 / time1) / (events2 / time2))
}

#[cfg(test)]
mod tests;

// [cite:clinical_trials_statistics]
