//! Zero-allocation pharmacokinetic exposure and therapeutic-window analytics.
//!
//! Computes peak (`C_max`), trough (`C_min`), trapezoidal area under the curve (`AUC_total`,
//! `AUC_0_tau`), theoretical and observed accumulation ratios (`R_ac`), Time-in-Therapeutic-Range
//! (`%` and hours), and supratherapeutic toxic time (`%` and hours) directly over caller slices.

use crate::error::PharmacokineticsError;

/// Lower and upper plasma concentration bounds defining a drug's therapeutic window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TherapeuticWindow {
    /// Minimum Effective Concentration (MEC) in mg/L ($\ge 0$).
    pub mec_mg_l: f64,
    /// Minimum Toxic Concentration (MTC) in mg/L ($> \text{MEC}$).
    pub mtc_mg_l: f64,
}

impl TherapeuticWindow {
    /// Constructs and validates a [`TherapeuticWindow`] with `0 <= mec_mg_l < mtc_mg_l`.
    #[verified_engine::verified]
    pub fn new(mec_mg_l: f64, mtc_mg_l: f64) -> Result<Self, PharmacokineticsError> {
        if !mec_mg_l.is_finite() || mec_mg_l < 0.0 {
            return Err(PharmacokineticsError::InvalidParameter(
                "MEC must be finite and non-negative".to_string(),
            ));
        }
        if !mtc_mg_l.is_finite() || mtc_mg_l <= mec_mg_l {
            return Err(PharmacokineticsError::InvalidParameter(
                "MTC must be finite and strictly greater than MEC".to_string(),
            ));
        }
        debug_assert!(mec_mg_l >= 0.0);
        debug_assert!(mtc_mg_l > mec_mg_l);
        Ok(Self { mec_mg_l, mtc_mg_l })
    }
}

/// Summary exposure and therapeutic-window metrics computed over a concentration-time profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExposureSummary {
    /// Peak plasma concentration $C_{\max}$ (mg/L) across the sampled horizon.
    pub c_max: f64,
    /// Time $t_{\text{peak}}$ (hours) at which $C_{\max}$ occurs.
    pub t_peak_h: f64,
    /// Trough plasma concentration $C_{\min}$ (mg/L) across the terminal dosing interval.
    pub c_min: f64,
    /// Total trapezoidal area under the concentration-time curve ($\text{mg}\cdot\text{h}/\text{L}$).
    pub auc_total: f64,
    /// Trapezoidal area under the curve over the terminal dosing interval $\text{AUC}_{0-\tau}$.
    pub auc_0_tau: f64,
    /// Theoretical one-compartment accumulation ratio $R_{\text{ac}} = 1 / (1 - e^{-k_e \tau})$.
    pub accumulation_ratio: f64,
    /// Empirical ratio of terminal-interval $\text{AUC}_{0-\tau}$ to initial-interval $\text{AUC}_{0-\tau}$.
    pub observed_accumulation_ratio: f64,
    /// Percentage of sampled horizon inside `[MEC, MTC]` (`0.0..=100.0`).
    pub time_in_range_pct: f64,
    /// Hours spent inside `[MEC, MTC]` across the sampled horizon.
    pub time_in_range_h: f64,
    /// Percentage of sampled horizon strictly above `MTC` (`0.0..=100.0`).
    pub toxic_pct: f64,
    /// Hours spent strictly above `MTC` across the sampled horizon.
    pub toxic_hours: f64,
    /// Percentage of sampled horizon strictly below `MEC` (`0.0..=100.0`).
    pub subtherapeutic_pct: f64,
    /// Hours spent strictly below `MEC` across the sampled horizon.
    pub subtherapeutic_hours: f64,
}

/// Validates that `times` is strictly increasing and matches finite non-negative `concentrations`.
#[verified_engine::verified]
fn validate_profile(times: &[f64], concentrations: &[f64]) -> Result<(), PharmacokineticsError> {
    if times.len() < 2 || times.len() != concentrations.len() {
        return Err(PharmacokineticsError::InvalidParameter(
            "Exposure analytics require matching slices with at least 2 sample points".to_string(),
        ));
    }
    if !times[0].is_finite() || !concentrations[0].is_finite() || concentrations[0] < 0.0 {
        return Err(PharmacokineticsError::InvalidParameter(
            "Sample times must be finite and concentrations finite and non-negative".to_string(),
        ));
    }
    let mut idx = 1;
    while idx < times.len() {
        let t_prev = times[idx - 1];
        let t_curr = times[idx];
        let c_curr = concentrations[idx];
        if !t_curr.is_finite() || t_curr <= t_prev || !c_curr.is_finite() || c_curr < 0.0 {
            return Err(PharmacokineticsError::InvalidParameter(
                "Sample times must be strictly increasing and concentrations finite >= 0"
                    .to_string(),
            ));
        }
        idx += 1;
    }
    debug_assert!(times.len() >= 2);
    debug_assert_eq!(times.len(), concentrations.len());
    Ok(())
}

/// Returns `(c_max, t_peak_h)` across a validated concentration-time slice pair.
#[verified_engine::verified]
pub fn peak_concentration(
    times: &[f64],
    concentrations: &[f64],
) -> Result<(f64, f64), PharmacokineticsError> {
    validate_profile(times, concentrations)?;
    let mut best_c = concentrations[0];
    let mut best_t = times[0];
    let mut idx = 1;
    while idx < concentrations.len() {
        if concentrations[idx] > best_c {
            best_c = concentrations[idx];
            best_t = times[idx];
        }
        idx += 1;
    }
    debug_assert!(best_c >= 0.0);
    debug_assert!(best_t >= times[0]);
    Ok((best_c, best_t))
}

/// Linearly interpolates the concentration at `query_t` clamped to `[times[0], times[last]]`.
#[verified_engine::verified]
fn interpolate_clamped(times: &[f64], concentrations: &[f64], query_t: f64) -> f64 {
    let last = times.len() - 1;
    if query_t <= times[0] {
        return concentrations[0];
    }
    if query_t >= times[last] {
        return concentrations[last];
    }
    let mut idx = 0;
    while idx < last {
        let t0 = times[idx];
        let t1 = times[idx + 1];
        if query_t >= t0 && query_t <= t1 {
            let span = t1 - t0;
            let alpha = (query_t - t0) / span;
            let out = concentrations[idx] + alpha * (concentrations[idx + 1] - concentrations[idx]);
            debug_assert!(out.is_finite());
            return out;
        }
        idx += 1;
    }
    concentrations[last]
}

/// Computes the minimum concentration `C_min` over `[start_h, end_h]` using boundary interpolation.
#[verified_engine::verified]
pub fn trough_concentration(
    times: &[f64],
    concentrations: &[f64],
    start_h: f64,
    end_h: f64,
) -> Result<f64, PharmacokineticsError> {
    validate_profile(times, concentrations)?;
    if !start_h.is_finite() || !end_h.is_finite() || end_h <= start_h {
        return Err(PharmacokineticsError::InvalidParameter(
            "trough_concentration requires finite start_h < end_h".to_string(),
        ));
    }
    let lo = start_h.max(times[0]);
    let hi = end_h.min(times[times.len() - 1]);
    if hi <= lo {
        return Err(PharmacokineticsError::InvalidParameter(
            "Requested trough interval does not overlap sampled horizon".to_string(),
        ));
    }
    let c_lo = interpolate_clamped(times, concentrations, lo);
    let c_hi = interpolate_clamped(times, concentrations, hi);
    let mut min_c = c_lo.min(c_hi);
    let mut idx = 0;
    while idx < times.len() {
        let t = times[idx];
        if t >= lo && t <= hi && concentrations[idx] < min_c {
            min_c = concentrations[idx];
        }
        idx += 1;
    }
    debug_assert!(min_c >= 0.0);
    debug_assert!(min_c.is_finite());
    Ok(min_c)
}

/// Computes total trapezoidal area under the curve ($\text{AUC}_{\text{total}}$) across `times`.
#[verified_engine::verified]
pub fn trapezoidal_auc(
    times: &[f64],
    concentrations: &[f64],
) -> Result<f64, PharmacokineticsError> {
    validate_profile(times, concentrations)?;
    let mut area = 0.0;
    let mut idx = 0;
    while idx + 1 < times.len() {
        let dt = times[idx + 1] - times[idx];
        area += 0.5 * (concentrations[idx] + concentrations[idx + 1]) * dt;
        idx += 1;
    }
    debug_assert!(area >= 0.0);
    debug_assert!(area.is_finite());
    Ok(area)
}

/// Computes piecewise-linear trapezoidal AUC restricted to `[start_h, end_h]`.
#[verified_engine::verified]
pub fn auc_interval(
    times: &[f64],
    concentrations: &[f64],
    start_h: f64,
    end_h: f64,
) -> Result<f64, PharmacokineticsError> {
    validate_profile(times, concentrations)?;
    if !start_h.is_finite() || !end_h.is_finite() || end_h <= start_h {
        return Err(PharmacokineticsError::InvalidParameter(
            "auc_interval requires finite start_h < end_h".to_string(),
        ));
    }
    let lo = start_h.max(times[0]);
    let hi = end_h.min(times[times.len() - 1]);
    if hi <= lo {
        return Ok(0.0);
    }
    let mut area = 0.0;
    let mut idx = 0;
    while idx + 1 < times.len() {
        let seg_lo = times[idx].max(lo);
        let seg_hi = times[idx + 1].min(hi);
        if seg_hi > seg_lo {
            let c0 = interpolate_clamped(times, concentrations, seg_lo);
            let c1 = interpolate_clamped(times, concentrations, seg_hi);
            area += 0.5 * (c0 + c1) * (seg_hi - seg_lo);
        }
        idx += 1;
    }
    debug_assert!(area >= 0.0);
    debug_assert!(area.is_finite());
    Ok(area)
}

/// Computes theoretical one-compartment steady-state accumulation $R_{\text{ac}} = 1 / (1 - e^{-k_e \tau})$.
#[verified_engine::verified]
pub fn accumulation_ratio(ke: f64, tau_h: f64) -> Result<f64, PharmacokineticsError> {
    if !ke.is_finite() || ke <= 0.0 || !tau_h.is_finite() || tau_h <= 0.0 {
        return Err(PharmacokineticsError::InvalidParameter(
            "accumulation_ratio requires positive finite ke and tau_h".to_string(),
        ));
    }
    let denom = 1.0 - (-ke * tau_h).exp();
    let ratio = 1.0 / denom;
    debug_assert!(ratio >= 1.0);
    debug_assert!(ratio.is_finite());
    Ok(ratio)
}

/// Counts `(subtherapeutic, in_range, toxic)` samples across `concentrations`.
#[verified_engine::verified]
fn count_window_samples(
    concentrations: &[f64],
    window: TherapeuticWindow,
) -> (usize, usize, usize) {
    let mut sub = 0usize;
    let mut in_range = 0usize;
    let mut toxic = 0usize;
    let mut idx = 0;
    while idx < concentrations.len() {
        let c = concentrations[idx];
        if c < window.mec_mg_l {
            sub += 1;
        } else if c > window.mtc_mg_l {
            toxic += 1;
        } else {
            in_range += 1;
        }
        idx += 1;
    }
    debug_assert_eq!(sub + in_range + toxic, concentrations.len());
    (sub, in_range, toxic)
}

/// Computes comprehensive exposure and therapeutic-window analytics with zero heap allocations.
#[verified_engine::verified]
pub fn summarize_exposure(
    times: &[f64],
    concentrations: &[f64],
    window: TherapeuticWindow,
    tau_h: f64,
    ke: f64,
) -> Result<ExposureSummary, PharmacokineticsError> {
    window.new_check()?;
    validate_profile(times, concentrations)?;
    let r_ac = accumulation_ratio(ke, tau_h)?;

    let (c_max, t_peak_h) = peak_concentration(times, concentrations)?;
    let auc_total = trapezoidal_auc(times, concentrations)?;
    let t_start = times[0];
    let t_end = times[times.len() - 1];
    let horizon_h = t_end - t_start;

    let term_start = if horizon_h >= tau_h {
        t_end - tau_h
    } else {
        t_start
    };
    let c_min = trough_concentration(times, concentrations, term_start, t_end)?;
    let auc_0_tau = auc_interval(times, concentrations, term_start, t_end)?;
    let first_end = (t_start + tau_h).min(t_end);
    let auc_first = auc_interval(times, concentrations, t_start, first_end)?;
    let observed_r_ac = if auc_first > math_commons::registry::TOLERANCE_STANDARD {
        auc_0_tau / auc_first
    } else {
        1.0
    };

    let (sub_n, range_n, toxic_n) = count_window_samples(concentrations, window);
    let n_f64 = concentrations.len() as f64;
    let sub_frac = sub_n as f64 / n_f64;
    let range_frac = range_n as f64 / n_f64;
    let toxic_frac = toxic_n as f64 / n_f64;
    debug_assert!((sub_frac + range_frac + toxic_frac - 1.0).abs() < 1e-12);

    Ok(ExposureSummary {
        c_max,
        t_peak_h,
        c_min,
        auc_total,
        auc_0_tau,
        accumulation_ratio: r_ac,
        observed_accumulation_ratio: observed_r_ac,
        time_in_range_pct: range_frac * 100.0,
        time_in_range_h: range_frac * horizon_h,
        toxic_pct: toxic_frac * 100.0,
        toxic_hours: toxic_frac * horizon_h,
        subtherapeutic_pct: sub_frac * 100.0,
        subtherapeutic_hours: sub_frac * horizon_h,
    })
}

impl TherapeuticWindow {
    #[verified_engine::verified]
    fn new_check(&self) -> Result<(), PharmacokineticsError> {
        Self::new(self.mec_mg_l, self.mtc_mg_l).map(|_| ())
    }
}
