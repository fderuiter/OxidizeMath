//! Shared numeric and scientific-notation formatters for ticks, hover tooltips,
//! accessibility announcements, callout badges, and regression formulas.

/// Lower magnitude threshold below which plot readouts and ticks switch to scientific notation.
pub(crate) const READOUT_SCI_LOWER: f64 = 1e-4;

/// Upper magnitude threshold at or above which plot readouts and ticks switch to scientific notation.
pub(crate) const READOUT_SCI_UPPER: f64 = 1e6;

/// Lower magnitude threshold below which curve-fit coefficients switch to scientific notation.
pub(crate) const FIT_SCI_LOWER: f64 = 1e-3;

/// Upper magnitude threshold at or above which curve-fit coefficients switch to scientific notation.
pub(crate) const FIT_SCI_UPPER: f64 = 1e5;

/// Rounds `value` to 12 significant decimal digits across the full finite `f64` range
/// (`1e-308..=1e308`), stripping IEEE-754 binary-to-decimal rounding artifacts.
#[must_use]
pub(crate) fn round_sig_figs_12(value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }
    if value == 0.0 {
        return 0.0;
    }
    format!("{value:.11e}").parse::<f64>().unwrap_or(value)
}

/// Formats `value` in scientific notation rounded to 12 significant digits with trailing
/// fractional zeros stripped (for example, `3.0 * 10_f64.powi(-8)` renders as `"3e-8"`).
#[must_use]
pub(crate) fn format_trimmed_sci_12(value: f64) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
    if value == 0.0 {
        return "0".to_owned();
    }
    let sci = format!("{value:.11e}");
    let Some((mantissa, exp)) = sci.split_once('e') else {
        return sci;
    };
    let trimmed = mantissa.trim_end_matches('0').trim_end_matches('.');
    if trimmed == "-0" || trimmed == "0" {
        return "0".to_owned();
    }
    format!("{trimmed}e{exp}")
}

/// Formats an axis tick mark with 12-significant-digit mantissa rounding, switching to
/// scientific notation outside `[1e-4, 1e6)`.
#[must_use]
pub(crate) fn format_axis_tick(value: f64, step_size: f64) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
    let rounded = round_sig_figs_12(value);
    let abs_v = rounded.abs();
    let step = step_size.abs();
    let num_decimals = if step.is_finite() && step >= READOUT_SCI_LOWER {
        (-step.log10().round()).max(0.0) as usize
    } else {
        0
    };
    if abs_v == 0.0 {
        return egui::emath::format_with_decimals_in_range(0.0, num_decimals..=num_decimals);
    }
    if !(READOUT_SCI_LOWER..READOUT_SCI_UPPER).contains(&abs_v) {
        return format_trimmed_sci_12(rounded);
    }
    let max_decimals = num_decimals.clamp(6, 12);
    egui::emath::format_with_decimals_in_range(rounded, num_decimals..=max_decimals)
}

/// Formats a data coordinate or metric readout using scientific notation outside `[1e-4, 1e6)`
/// (with `sci_decimals` fractional mantissa digits, e.g. `4` -> `"4.9000e-7"`) and fixed-point
/// inside `[1e-4, 1e6)`.
#[must_use]
pub(crate) fn format_readout_value(
    value: f64,
    fixed_decimals: usize,
    sci_decimals: usize,
) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
    let rounded = round_sig_figs_12(value);
    let abs_v = rounded.abs();
    if abs_v == 0.0 {
        return format!("{:.*}", fixed_decimals, 0.0_f64);
    }
    if !(READOUT_SCI_LOWER..READOUT_SCI_UPPER).contains(&abs_v) {
        return format!("{rounded:.sci_decimals$e}");
    }
    let min_sig_decimals = if abs_v < 1.0 {
        ((-abs_v.log10()).ceil().max(0.0) as usize + 1).min(6)
    } else {
        0
    };
    let decimals = fixed_decimals.max(min_sig_decimals);
    format!("{rounded:.decimals$}")
}

/// Formats a curve-fit model parameter to 4 significant digits, using scientific notation
/// outside `[1e-3, 1e5)`.
#[must_use]
pub(crate) fn format_fit_param(value: f64) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
    if value == 0.0 {
        return "0".to_owned();
    }
    let sci = format!("{value:.3e}");
    let rounded = sci.parse::<f64>().unwrap_or(value);
    let abs_r = rounded.abs();
    if abs_r == 0.0 {
        return "0".to_owned();
    }
    if !(FIT_SCI_LOWER..FIT_SCI_UPPER).contains(&abs_r) {
        return format!("{rounded:.3e}");
    }
    let exp10 = abs_r.log10().floor() as i32;
    let decimals = (3 - exp10).max(0) as usize;
    format!("{rounded:.decimals$}")
}
