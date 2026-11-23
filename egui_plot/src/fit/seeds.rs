//! Closed-form linearized parameter seed estimators for preset nonlinear models.

use crate::fit::{
    lm::sum_squared_errors,
    nonlinear::{eval_exp, eval_logistic, eval_power, eval_sin},
    poly::solve_dense_small,
};

/// Weighted simple linear regression $v \approx \text{intercept} + \text{slope} \cdot u$.
pub(crate) fn weighted_ols_line(samples: &[[f64; 3]]) -> Option<(f64, f64)> {
    if samples.len() < 2 {
        return None;
    }
    let (mut sw, mut su, mut sv, mut suu, mut suv) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &[u, v, w] in samples {
        if !u.is_finite() || !v.is_finite() || !w.is_finite() || w <= 0.0 {
            continue;
        }
        sw += w;
        su += w * u;
        sv += w * v;
        suu += w * u * u;
        suv += w * u * v;
    }
    let denom = sw * suu - su * su;
    if !denom.is_finite() || denom.abs() <= 1e-15 || sw <= 0.0 {
        return None;
    }
    let intercept = (suu * sv - su * suv) / denom;
    let slope = (sw * suv - su * sv) / denom;
    (intercept.is_finite() && slope.is_finite()).then_some((intercept, slope))
}

/// Seed $[a, b]$ for $y = a \exp(b x)$ (returns exact Log Mode fit when `log_mode` is `true`).
pub(crate) fn seed_exponential(points: &[[f64; 2]], log_mode: bool) -> Option<[f64; 2]> {
    let pos_energy: f64 = points
        .iter()
        .filter(|p| p[1] > 0.0)
        .map(|p| p[1] * p[1])
        .sum();
    let neg_energy: f64 = points
        .iter()
        .filter(|p| p[1] < 0.0)
        .map(|p| p[1] * p[1])
        .sum();
    let sign = if pos_energy >= neg_energy { 1.0 } else { -1.0 };

    let unweighted: Vec<[f64; 3]> = points
        .iter()
        .filter(|p| p[1] * sign > 0.0)
        .filter_map(|p| {
            let ly = (p[1] * sign).ln();
            ly.is_finite().then_some([p[0], ly, 1.0])
        })
        .collect();
    let (int_u, b_u) = weighted_ols_line(&unweighted)?;
    let cand_u = [sign * int_u.exp(), b_u];
    if log_mode || !cand_u[0].is_finite() {
        return cand_u[0].is_finite().then_some(cand_u);
    }

    let weighted: Vec<[f64; 3]> = points
        .iter()
        .filter(|p| p[1] * sign > 0.0)
        .filter_map(|p| {
            let ly = (p[1] * sign).ln();
            let w = (p[1] * p[1]).max(1e-18);
            (ly.is_finite() && w.is_finite()).then_some([p[0], ly, w])
        })
        .collect();
    if let Some((int_w, b_w)) = weighted_ols_line(&weighted) {
        let cand_w = [sign * int_w.exp(), b_w];
        if cand_w[0].is_finite()
            && sum_squared_errors(points, &cand_w, eval_exp)
                < sum_squared_errors(points, &cand_u, eval_exp)
        {
            return Some(cand_w);
        }
    }
    Some(cand_u)
}

/// Seed $[a, b]$ for $y = a x^b$ on $x > 0$ (exact Log Mode fit when `log_mode` is `true`).
pub(crate) fn seed_power_law(points: &[[f64; 2]], log_mode: bool) -> Option<[f64; 2]> {
    let pos_energy: f64 = points
        .iter()
        .filter(|p| p[1] > 0.0)
        .map(|p| p[1] * p[1])
        .sum();
    let neg_energy: f64 = points
        .iter()
        .filter(|p| p[1] < 0.0)
        .map(|p| p[1] * p[1])
        .sum();
    let sign = if pos_energy >= neg_energy { 1.0 } else { -1.0 };

    let unweighted: Vec<[f64; 3]> = points
        .iter()
        .filter(|p| p[0] > 0.0 && p[1] * sign > 0.0)
        .filter_map(|p| {
            let (lx, ly) = (p[0].ln(), (p[1] * sign).ln());
            (lx.is_finite() && ly.is_finite()).then_some([lx, ly, 1.0])
        })
        .collect();
    let (int_u, b_u) = weighted_ols_line(&unweighted)?;
    let cand_u = [sign * int_u.exp(), b_u];
    if log_mode || !cand_u[0].is_finite() {
        return cand_u[0].is_finite().then_some(cand_u);
    }

    let weighted: Vec<[f64; 3]> = points
        .iter()
        .filter(|p| p[0] > 0.0 && p[1] * sign > 0.0)
        .filter_map(|p| {
            let (lx, ly) = (p[0].ln(), (p[1] * sign).ln());
            let w = (p[1] * p[1]).max(1e-18);
            (lx.is_finite() && ly.is_finite() && w.is_finite()).then_some([lx, ly, w])
        })
        .collect();
    if let Some((int_w, b_w)) = weighted_ols_line(&weighted) {
        let cand_w = [sign * int_w.exp(), b_w];
        if cand_w[0].is_finite()
            && sum_squared_errors(points, &cand_w, eval_power)
                < sum_squared_errors(points, &cand_u, eval_power)
        {
            return Some(cand_w);
        }
    }
    Some(cand_u)
}

/// Exact closed-form parameters $[a, b]$ for $y = a \ln(x) + b$ on $x > 0$.
pub(crate) fn seed_logarithmic(points: &[[f64; 2]]) -> Option<[f64; 2]> {
    let samples: Vec<[f64; 3]> = points
        .iter()
        .filter(|p| p[0] > 0.0)
        .filter_map(|p| {
            let lx = p[0].ln();
            lx.is_finite().then_some([lx, p[1], 1.0])
        })
        .collect();
    let (intercept, slope) = weighted_ols_line(&samples)?;
    Some([slope, intercept])
}

/// Multi-candidate logit seed $[L, a, k]$ for $y = L / (1 + a \exp(-k x))$.
pub(crate) fn seed_logistic(points: &[[f64; 2]]) -> Option<[f64; 3]> {
    if points.len() < 3 {
        return None;
    }
    let y_max = points
        .iter()
        .map(|p| p[1])
        .fold(f64::NEG_INFINITY, f64::max);
    let y_min = points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
    let sign = if y_max <= 0.0 && y_min < 0.0 {
        -1.0
    } else {
        1.0
    };
    let peak = if sign > 0.0 { y_max } else { -y_min };
    if !peak.is_finite() || peak <= 0.0 {
        return None;
    }

    let mut best: Option<([f64; 3], f64)> = None;
    for mult in [1.001, 1.005, 1.02, 1.06, 1.15, 1.35, 1.8, 2.6] {
        let cap = peak * mult;
        if let Some(cand) = fit_logit_for_asymptote(points, cap, sign)
            && let Some(sse) = sum_squared_errors(points, &cand, eval_logistic)
            && best.is_none_or(|(_, best_sse)| sse < best_sse)
        {
            best = Some((cand, sse));
        }
    }
    best.map(|(theta, _)| theta)
}

fn fit_logit_for_asymptote(points: &[[f64; 2]], cap: f64, sign: f64) -> Option<[f64; 3]> {
    let mut samples = Vec::with_capacity(points.len());
    for p in points {
        let y_pos = p[1] * sign;
        if y_pos <= 0.0 || y_pos >= cap {
            continue;
        }
        let ratio = cap / y_pos - 1.0;
        let logit = ratio.ln();
        let weight = (y_pos * (cap - y_pos)).powi(2).max(1e-18);
        if logit.is_finite() && weight.is_finite() {
            samples.push([p[0], logit, weight]);
        }
    }
    let (ln_a, neg_k) = weighted_ols_line(&samples)?;
    let a = ln_a.clamp(-60.0, 60.0).exp();
    let k = -neg_k;
    (a.is_finite() && k.is_finite()).then_some([sign * cap, a, k])
}

/// Seed $[a, b, c, d]$ for $y = a \sin(b x + c) + d$ combining Jacquelin integral
/// frequency estimation and strided harmonic linear projection.
pub(crate) fn seed_sinusoidal(points: &[[f64; 2]]) -> Option<[f64; 4]> {
    if points.len() < 4 {
        return None;
    }
    let mut sorted = points.to_vec();
    sorted.sort_by(|u, v| u[0].total_cmp(&v[0]));
    let span = sorted[sorted.len() - 1][0] - sorted[0][0];
    if !span.is_finite() || span <= 1e-12 {
        return None;
    }
    let stride = (sorted.len() / 256).max(1);
    let sub: Vec<[f64; 2]> = sorted.iter().step_by(stride).copied().collect();

    let mut freqs = Vec::with_capacity(68);
    if let Some(b_jacq) = jacquelin_frequency(&sorted) {
        freqs.push(b_jacq);
    }
    if let Some(b_zc) = zero_crossing_frequency(&sorted, span) {
        freqs.push(b_zc);
    }
    let f_min = 0.25 * std::f64::consts::PI / span;
    let max_cycles = ((sorted.len() as f64) * 0.4).clamp(4.0, 32.0);
    let f_max = 2.0 * std::f64::consts::PI * max_cycles / span;
    for k in 0..64 {
        let t = k as f64 / 63.0;
        freqs.push(f_min + t * (f_max - f_min));
    }

    let mut best: Option<([f64; 4], f64)> = None;
    for b in freqs {
        if !b.is_finite() || b <= 1e-9 {
            continue;
        }
        if let Some(cand) = solve_harmonic_linear(&sub, b)
            && let Some(sse) = sum_squared_errors(&sub, &cand, eval_sin)
            && best.is_none_or(|(_, best_sse)| sse < best_sse)
        {
            best = Some((cand, sse));
        }
    }
    let (coarse, _) = best?;
    solve_harmonic_linear(&sorted, coarse[1]).or(Some(coarse))
}

fn jacquelin_frequency(sorted: &[[f64; 2]]) -> Option<f64> {
    let n = sorted.len();
    let y_mean = sorted.iter().map(|p| p[1]).sum::<f64>() / (n as f64);
    let mut s1 = vec![0.0; n];
    let mut s2 = vec![0.0; n];
    for i in 1..n {
        let dx = sorted[i][0] - sorted[i - 1][0];
        s1[i] = s1[i - 1] + 0.5 * ((sorted[i][1] - y_mean) + (sorted[i - 1][1] - y_mean)) * dx;
        s2[i] = s2[i - 1] + 0.5 * (s1[i] + s1[i - 1]) * dx;
    }
    let x0 = sorted[0][0];
    let mut ata = [[0.0_f64; 6]; 6];
    let mut atb = [0.0_f64; 6];
    for (i, p) in sorted.iter().enumerate() {
        let u = p[0] - x0;
        let row = [s2[i], u * u, u, 1.0];
        for r in 0..4 {
            atb[r] += row[r] * p[1];
            for c in 0..4 {
                ata[r][c] += row[r] * row[c];
            }
        }
    }
    let sol = solve_dense_small(&mut ata, &mut atb, 4)?;
    (sol[0] < -1e-12 && sol[0].is_finite()).then(|| (-sol[0]).sqrt())
}

fn zero_crossing_frequency(sorted: &[[f64; 2]], span: f64) -> Option<f64> {
    let y_mean = sorted.iter().map(|p| p[1]).sum::<f64>() / (sorted.len() as f64);
    let mut crossings = 0_usize;
    for w in sorted.windows(2) {
        let d0 = w[0][1] - y_mean;
        let d1 = w[1][1] - y_mean;
        if (d0 >= 0.0 && d1 < 0.0) || (d0 < 0.0 && d1 >= 0.0) {
            crossings += 1;
        }
    }
    (crossings >= 1).then(|| std::f64::consts::PI * (crossings as f64) / span)
}

fn solve_harmonic_linear(points: &[[f64; 2]], b: f64) -> Option<[f64; 4]> {
    let mut ata = [[0.0_f64; 6]; 6];
    let mut atb = [0.0_f64; 6];
    for p in points {
        let (s, c) = (b * p[0]).sin_cos();
        let row = [s, c, 1.0];
        for r in 0..3 {
            atb[r] += row[r] * p[1];
            for col in 0..3 {
                ata[r][col] += row[r] * row[col];
            }
        }
    }
    let sol = solve_dense_small(&mut ata, &mut atb, 3)?;
    let amp = sol[0].hypot(sol[1]);
    let phase = sol[1].atan2(sol[0]);
    Some([amp, b, phase, sol[2]])
}
