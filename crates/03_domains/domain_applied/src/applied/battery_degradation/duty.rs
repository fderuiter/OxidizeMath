//! Coupled multi-segment duty-cycle state-of-health integration and warranty root finding.
// [cite:battery_degradation]

use super::model::{
    ElectrochemicalAgingModel, MAX_C_RATE, MAX_TEMPERATURE_C, MIN_C_RATE, MIN_TEMPERATURE_C,
};
use super::types::{Capacity, DepthOfDischarge};

/// Maximum number of duty-profile segments supported on stack buffers and URL state.
pub const MAX_DUTY_SEGMENTS: usize = 8;

/// Standard astronomical days per calendar year.
pub const DAYS_PER_YEAR: f64 = 365.25;

const MAX_SOLVER_DAYS: f64 = 365.25 * 500.0;
const BISECTION_STEPS: usize = 64;

fn clamp_finite(val: f64, min: f64, max: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(min, max)
    } else {
        fallback
    }
}

/// Single operating phase inside a repeating battery duty schedule (e.g. commute, fast-charge, storage).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DutySegment {
    /// Relative duration (in days) of this segment inside the repeating duty cycle (`0.01..=365.25`).
    pub duration_days: f64,
    /// Depth of discharge per cycle in percent (`0.0..=100.0`).
    pub dod_pct: f64,
    /// Equivalent cycles per day during this segment (`0.0..=10.0`; `0.0` models pure calendar storage).
    pub daily_cycles: f64,
    /// Ambient/cell temperature during this segment in degrees Celsius (`-20.0..=60.0`).
    pub temperature_c: f64,
    /// Charge/discharge C-rate during cycling (`0.05..=10.0`).
    pub c_rate: f64,
}

impl DutySegment {
    /// Constructs a clamped [`DutySegment`].
    #[must_use]
    #[verified_engine::verified]
    pub fn new(
        duration_days: f64,
        dod_pct: f64,
        daily_cycles: f64,
        temperature_c: f64,
        c_rate: f64,
    ) -> Self {
        Self {
            duration_days: clamp_finite(duration_days, 0.01, 365.25, 1.0),
            dod_pct: clamp_finite(dod_pct, 0.0, 100.0, 60.0),
            daily_cycles: clamp_finite(daily_cycles, 0.0, 10.0, 1.0),
            temperature_c: clamp_finite(temperature_c, MIN_TEMPERATURE_C, MAX_TEMPERATURE_C, 25.0),
            c_rate: clamp_finite(c_rate, MIN_C_RATE, MAX_C_RATE, 1.0),
        }
    }

    /// Standard weekday urban commute duty segment (`5 d`, `50% DoD`, `1.0 cyc/d`, `25 °C`, `0.5C`).
    #[must_use]
    #[verified_engine::verified]
    pub fn commute() -> Self {
        Self::new(5.0, 50.0, 1.0, 25.0, 0.5)
    }

    /// Highway DC fast-charge trip segment (`1 d`, `80% DoD`, `1.5 cyc/d`, `38 °C`, `2.5C`).
    #[must_use]
    #[verified_engine::verified]
    pub fn fast_charge() -> Self {
        Self::new(1.0, 80.0, 1.5, 38.0, 2.5)
    }

    /// Idle calendar storage segment (`daily_cycles = 0.0`) integrating only $\sqrt{t}$ SEI fade.
    #[must_use]
    #[verified_engine::verified]
    pub fn storage(duration_days: f64, temperature_c: f64) -> Self {
        Self::new(duration_days, 0.0, 0.0, temperature_c, 0.5)
    }
}

/// Decomposed coupled state-of-health breakdown at a given calendar horizon.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DutyPointBreakdown {
    /// Coupled remaining state-of-health fraction (`0.0..=1.0`).
    pub soh: f64,
    /// Cycling-only state-of-health fraction (`0.0..=1.0`) without SEI calendar fade.
    pub cycling_only_soh: f64,
    /// Fractional capacity loss attributable to cycling (`0.0..=1.0`).
    pub cycling_loss: f64,
    /// Fractional capacity loss attributable to parabolic $\sqrt{t}$ SEI calendar aging (`0.0..=1.0`).
    pub calendar_sei_loss: f64,
    /// Cumulative equivalent cycles accrued up to this time point.
    pub cumulative_cycles: f64,
}

/// Warranty and end-of-life milestone projections under a coupled duty profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DutyLifetimeSummary {
    /// Days to user-specified target capacity.
    pub days_to_target: f64,
    /// Calendar years (`days / 365.25`) to user-specified target capacity.
    pub years_to_target: f64,
    /// Cumulative cycles accrued at user-specified target capacity.
    pub cycles_to_target: f64,
    /// Days to first-life warranty milestone (`80%` SOH).
    pub days_to_80: f64,
    /// Calendar years to first-life warranty milestone (`80%` SOH).
    pub years_to_80: f64,
    /// Cumulative cycles accrued at `80%` SOH.
    pub cycles_to_80: f64,
    /// Days to second-life / end-of-life warranty milestone (`70%` SOH).
    pub days_to_70: f64,
    /// Calendar years to `70%` SOH.
    pub years_to_70: f64,
    /// Cumulative cycles accrued at `70%` SOH.
    pub cycles_to_70: f64,
    /// Duty-weighted average daily cycles (`0.0` for pure storage).
    pub effective_daily_cycles: f64,
}

#[derive(Clone, Copy)]
struct DutyEffectiveRates {
    daily_cycles: f64,
    daily_damage_rate: f64,
    daily_sei_rate: f64,
    exponent_z: f64,
}

fn aggregate_duty_rates(
    base_model: &ElectrochemicalAgingModel,
    segments: &[DutySegment],
) -> DutyEffectiveRates {
    let count = segments.len().min(MAX_DUTY_SEGMENTS);
    if count == 0 {
        let af = base_model.arrhenius_factor();
        return DutyEffectiveRates {
            daily_cycles: 0.0,
            daily_damage_rate: 0.0,
            daily_sei_rate: base_model.sei_rate() * af,
            exponent_z: base_model.exponent_z(),
        };
    }

    let mut total_weight = 0.0_f64;
    let mut weighted_cycles = 0.0_f64;
    let mut weighted_damage = 0.0_f64;
    let mut weighted_sei_sq = 0.0_f64;

    for seg_raw in &segments[..count] {
        let seg = DutySegment::new(
            seg_raw.duration_days,
            seg_raw.dod_pct,
            seg_raw.daily_cycles,
            seg_raw.temperature_c,
            seg_raw.c_rate,
        );
        let w = seg.duration_days;
        total_weight += w;

        let seg_model = base_model
            .with_temperature_c(seg.temperature_c)
            .with_c_rate(seg.c_rate);
        let k_cal = seg_model.sei_rate() * seg_model.arrhenius_factor();
        weighted_sei_sq += w * k_cal * k_cal;

        if seg.daily_cycles > 0.0 && seg.dod_pct > 0.0 {
            weighted_cycles += w * seg.daily_cycles;
            let n70 = seg_model
                .n70(DepthOfDischarge::new_clamped(seg.dod_pct))
                .as_f64();
            if n70 > 0.0 {
                weighted_damage += w * (seg.daily_cycles / n70);
            }
        }
    }

    let inv_w = 1.0 / total_weight.max(1.0e-12);
    DutyEffectiveRates {
        daily_cycles: weighted_cycles * inv_w,
        daily_damage_rate: weighted_damage * inv_w,
        daily_sei_rate: (weighted_sei_sq * inv_w).max(0.0).sqrt(),
        exponent_z: base_model.exponent_z(),
    }
}

fn eval_breakdown_from_rates(rates: DutyEffectiveRates, elapsed_days: f64) -> DutyPointBreakdown {
    if !elapsed_days.is_finite() || elapsed_days <= 0.0 {
        return DutyPointBreakdown {
            soh: 1.0,
            cycling_only_soh: 1.0,
            cycling_loss: 0.0,
            calendar_sei_loss: 0.0,
            cumulative_cycles: 0.0,
        };
    }
    let cyc_loss = if rates.daily_damage_rate > 0.0 {
        (0.3 * (rates.daily_damage_rate * elapsed_days).powf(rates.exponent_z)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let sei_loss = (rates.daily_sei_rate * elapsed_days.sqrt()).clamp(0.0, 1.0);
    DutyPointBreakdown {
        soh: (1.0 - cyc_loss - sei_loss).clamp(0.0, 1.0),
        cycling_only_soh: (1.0 - cyc_loss).clamp(0.0, 1.0),
        cycling_loss: cyc_loss,
        calendar_sei_loss: sei_loss,
        cumulative_cycles: rates.daily_cycles * elapsed_days,
    }
}

fn solve_days_for_loss(rates: DutyEffectiveRates, target_loss: f64) -> f64 {
    let loss = target_loss.clamp(0.0, 1.0);
    if loss <= 0.0 {
        return 0.0;
    }
    if rates.daily_damage_rate <= 0.0 {
        // Storage-only segment: integrate strictly the parabolic sqrt(t) SEI calendar term.
        if rates.daily_sei_rate <= 0.0 {
            return MAX_SOLVER_DAYS;
        }
        let ratio = loss / rates.daily_sei_rate;
        return (ratio * ratio).min(MAX_SOLVER_DAYS);
    }

    let cyc_upper = ((loss / 0.3).powf(1.0 / rates.exponent_z) / rates.daily_damage_rate)
        .clamp(1.0e-6, MAX_SOLVER_DAYS);
    let mut lo = 0.0_f64;
    let mut hi = cyc_upper;
    for _ in 0..BISECTION_STEPS {
        let mid = 0.5 * (lo + hi);
        let total = 0.3 * (rates.daily_damage_rate * mid).powf(rates.exponent_z)
            + rates.daily_sei_rate * mid.sqrt();
        if total < loss {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Evaluates coupled SOH, cycling fade, and $\sqrt{t}$ SEI fade at `elapsed_days` across `1..=8` duty segments.
#[must_use]
#[verified_engine::verified]
pub fn evaluate_duty_point(
    model: &ElectrochemicalAgingModel,
    segments: &[DutySegment],
    elapsed_days: f64,
) -> DutyPointBreakdown {
    let rates = aggregate_duty_rates(model, segments);
    eval_breakdown_from_rates(rates, elapsed_days)
}

/// Solves coupled warranty horizons (`target`, `80%` SOH, and `70%` SOH) across `1..=8` duty segments.
#[must_use]
#[verified_engine::verified]
pub fn estimate_duty_lifetime(
    model: &ElectrochemicalAgingModel,
    segments: &[DutySegment],
    target: Capacity,
) -> DutyLifetimeSummary {
    let rates = aggregate_duty_rates(model, segments);
    let days_target = solve_days_for_loss(rates, 1.0 - target.as_f64());
    let days_80 = solve_days_for_loss(rates, 0.20);
    let days_70 = solve_days_for_loss(rates, 0.30);

    DutyLifetimeSummary {
        days_to_target: days_target,
        years_to_target: days_target / DAYS_PER_YEAR,
        cycles_to_target: rates.daily_cycles * days_target,
        days_to_80: days_80,
        years_to_80: days_80 / DAYS_PER_YEAR,
        cycles_to_80: rates.daily_cycles * days_80,
        days_to_70: days_70,
        years_to_70: days_70 / DAYS_PER_YEAR,
        cycles_to_70: rates.daily_cycles * days_70,
        effective_daily_cycles: rates.daily_cycles,
    }
}
