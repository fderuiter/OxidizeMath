//! Human-readable 4-significant-digit and scientific-notation equation formatters for [`FitModel`].
// [cite:math_commons]

use super::FitModel;

const SCI_LOWER: f64 = 1e-3;
const SCI_UPPER: f64 = 1e5;
const SUPERSCRIPTS: [&str; 5] = ["", "", "²", "³", "⁴"];

/// Formats a finite model coefficient magnitude to 4 significant digits, switching to
/// scientific notation outside `[1e-3, 1e5)`.
#[must_use]
pub(super) fn fmt_param_sig4(val: f64) -> String {
    if !val.is_finite() {
        return val.to_string();
    }
    if val == 0.0 {
        return "0".to_owned();
    }
    let rounded = format!("{val:.3e}").parse::<f64>().unwrap_or(val);
    let abs_v = rounded.abs();
    if abs_v == 0.0 {
        return "0".to_owned();
    }
    if !(SCI_LOWER..SCI_UPPER).contains(&abs_v) {
        return format!("{rounded:.3e}");
    }
    let decimals = (3 - abs_v.log10().floor() as i32).max(0) as usize;
    format!("{rounded:.decimals$}")
}

fn fmt_signed_param(val: f64) -> String {
    if val < 0.0 {
        format!("-{}", fmt_param_sig4(-val))
    } else {
        format!("+{}", fmt_param_sig4(val))
    }
}

fn fmt_shifted_x(x0: f64) -> String {
    if x0 == 0.0 {
        "x".to_owned()
    } else if x0 > 0.0 {
        format!("(x - {})", fmt_param_sig4(x0))
    } else {
        format!("(x + {})", fmt_param_sig4(-x0))
    }
}

/// Formats a human-readable polynomial or log-log power-law formula string.
#[must_use]
pub fn format_polynomial_formula(coeffs: &[f64], log_x: bool, log_y: bool) -> String {
    format_polynomial_with_scale(coeffs, 1.0, log_x, log_y)
}

pub(super) fn format_polynomial_with_scale(
    coeffs: &[f64],
    u_scale: f64,
    log_x: bool,
    log_y: bool,
) -> String {
    if log_x && log_y && coeffs.len() == 2 {
        let a = 10.0_f64.powf(coeffs[0]);
        return format!(
            "y = {}·x^({})",
            fmt_param_sig4(a),
            fmt_param_sig4(coeffs[1])
        );
    }
    let lhs = if log_y { "log10(y)" } else { "y" };
    let var = if log_x { "log10(x)" } else { "x" };
    let degree = coeffs.len().saturating_sub(1);
    if degree == 0 {
        let c0 = coeffs.first().copied().unwrap_or(0.0);
        return format!("{lhs} = {}", fmt_param_sig4(c0));
    }
    let scale = u_scale.abs().max(1.0);
    let max_contrib = coeffs
        .iter()
        .enumerate()
        .map(|(p, &c)| c.abs() * scale.powi(p as i32))
        .fold(0.0_f64, f64::max);

    let mut out = format!("{lhs} = ");
    let mut emitted = 0usize;
    for power in (0..=degree).rev() {
        let coeff = coeffs[power];
        let contrib = coeff.abs() * scale.powi(power as i32);
        if max_contrib > 0.0 && contrib <= 1e-12 * max_contrib && emitted < degree && power > 0 {
            continue;
        }
        if max_contrib == 0.0 && power > 0 {
            continue;
        }
        out.push_str(&format_poly_term(coeff, power, var, log_x, emitted == 0));
        emitted += 1;
    }
    out
}

fn format_poly_term(coeff: f64, power: usize, var: &str, log_x: bool, is_first: bool) -> String {
    let sign_str = match (is_first, coeff < 0.0) {
        (true, true) => "-",
        (true, false) => "",
        (false, true) => " - ",
        (false, false) => " + ",
    };
    let abs_c = coeff.abs();
    let mag = fmt_param_sig4(abs_c);
    match power {
        0 => format!("{sign_str}{mag}"),
        1 if (abs_c - 1.0).abs() < 1e-6 => format!("{sign_str}{var}"),
        1 => format!("{sign_str}{mag}{var}"),
        p => {
            let base = if let Some(&sup) = SUPERSCRIPTS.get(p)
                && !sup.is_empty()
            {
                if log_x {
                    format!("({var}){sup}")
                } else {
                    format!("{var}{sup}")
                }
            } else {
                format!("{var}^{p}")
            };
            if (abs_c - 1.0).abs() < 1e-6 {
                format!("{sign_str}{base}")
            } else {
                format!("{sign_str}{mag}{base}")
            }
        }
    }
}

/// Formats a human-readable equation string for nonlinear [`FitModel`] presets.
#[must_use]
pub fn format_nonlinear_formula(model: FitModel, params: &[f64]) -> String {
    match model {
        FitModel::Exponential if params.len() >= 2 => {
            let x_var = fmt_shifted_x(params.get(2).copied().unwrap_or(0.0));
            format!(
                "y = {}·exp({}·{x_var})",
                fmt_param_sig4(params[0]),
                fmt_signed_param(params[1])
            )
        }
        FitModel::PowerLaw if params.len() >= 2 => {
            format!(
                "y = {}·x^({})",
                fmt_param_sig4(params[0]),
                fmt_param_sig4(params[1])
            )
        }
        FitModel::Logarithmic if params.len() >= 2 => {
            let sign = if params[1] >= 0.0 { "+" } else { "-" };
            format!(
                "y = {} {sign} {}·ln(x)",
                fmt_param_sig4(params[0]),
                fmt_param_sig4(params[1].abs())
            )
        }
        FitModel::Logistic if params.len() >= 3 => {
            let sign = if params[2] >= 0.0 { "-" } else { "+" };
            format!(
                "y = {} / (1 + exp(-{}·(x {sign} {})))",
                fmt_param_sig4(params[0]),
                fmt_param_sig4(params[1]),
                fmt_param_sig4(params[2].abs())
            )
        }
        FitModel::Sinusoidal if params.len() >= 4 => {
            let c_sign = if params[2] >= 0.0 { "+" } else { "-" };
            let d_sign = if params[3] >= 0.0 { "+" } else { "-" };
            format!(
                "y = {}·sin({}·x {c_sign} {}) {d_sign} {}",
                fmt_param_sig4(params[0]),
                fmt_param_sig4(params[1]),
                fmt_param_sig4(params[2].abs()),
                fmt_param_sig4(params[3].abs())
            )
        }
        _ => "y = NaN".to_string(),
    }
}
