//! Zero-allocation closed-form parameter seed estimators for nonlinear curve models.
// [cite:math_commons]

use super::poly::{solve_dense_small_raw, symmetrize_upper};
use crate::error::FitError;
use crate::series::PointSeries;

const MAX_SIN_SAMPLES: usize = 256;

/// Streaming accumulator for weighted linear regression $v = \alpha + \beta u$ on the stack.
#[derive(Debug, Clone, Copy, Default)]
pub struct WeightedOlsAccum {
    sw: f64,
    su: f64,
    sv: f64,
    suu: f64,
    suv: f64,
    count: usize,
}

impl WeightedOlsAccum {
    /// Number of valid weighted observations accumulated so far.
    #[inline]
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        self.count
    }

    /// Adds a weighted `(u, v)` sample with positive finite weight `w`.
    #[inline]
    pub fn push(&mut self, u: f64, v: f64, w: f64) {
        if u.is_finite() && v.is_finite() && w.is_finite() && w > 0.0 {
            let wu = w * u;
            self.sw += w;
            self.su += wu;
            self.sv += w * v;
            self.suu += wu * u;
            self.suv += wu * v;
            self.count += 1;
        }
    }

    /// Solves for `(intercept, slope)` or returns [`FitError::SingularMatrix`].
    #[verified_engine::verified]
    pub fn solve(&self) -> Result<(f64, f64), FitError> {
        if self.count < 2 {
            return Err(FitError::InsufficientPoints {
                required: 2,
                actual: self.count,
            });
        }
        let det = self.sw * self.suu - self.su * self.su;
        if !det.is_finite() || det.abs() <= 1e-15 {
            return Err(FitError::SingularMatrix);
        }
        let intercept = (self.suu * self.sv - self.su * self.suv) / det;
        let slope = (self.sw * self.suv - self.su * self.sv) / det;
        if intercept.is_finite() && slope.is_finite() {
            Ok((intercept, slope))
        } else {
            Err(FitError::SingularMatrix)
        }
    }
}

/// Solves weighted linear regression $v = \alpha + \beta u$ over `[u, v, w]` triples.
#[verified_engine::verified]
pub fn weighted_ols_line(samples: &[[f64; 3]]) -> Result<(f64, f64), FitError> {
    weighted_ols_line_inner(samples)
}

fn weighted_ols_line_inner(samples: &[[f64; 3]]) -> Result<(f64, f64), FitError> {
    let mut acc = WeightedOlsAccum::default();
    for &[u, v, w] in samples {
        acc.push(u, v, w);
    }
    acc.solve()
}

/// Estimates closed-form initial parameters `[a, b]` for $y = a e^{b x}$ via $y^2$-weighted log-OLS.
#[verified_engine::verified]
pub fn seed_exponential<P: PointSeries + ?Sized>(points: &P) -> Result<[f64; 2], FitError> {
    let [a_u, b, x0] = seed_exponential_shifted(points)?;
    let a = a_u * (-b * x0).clamp(-700.0, 700.0).exp();
    if a.is_finite() && b.is_finite() {
        Ok([a, b])
    } else {
        Err(FitError::InvalidDomain {
            reason: "non-finite exponential seed parameters",
        })
    }
}

pub(super) fn seed_exponential_shifted<P: PointSeries + ?Sized>(
    points: &P,
) -> Result<[f64; 3], FitError> {
    let mut finite_n = 0usize;
    let mut x_min = f64::INFINITY;
    for i in 0..points.len() {
        let p = points.point_at(i);
        if p[0].is_finite() && p[1].is_finite() {
            finite_n += 1;
            x_min = x_min.min(p[0]);
        }
    }
    if finite_n < 2 {
        return Err(FitError::InsufficientPoints {
            required: 2,
            actual: finite_n,
        });
    }
    let x0 = if x_min.is_finite() { x_min } else { 0.0 };
    let mut acc = WeightedOlsAccum::default();
    for i in 0..points.len() {
        let p = points.point_at(i);
        if p[0].is_finite() && p[1].is_finite() && p[1] > 0.0 {
            acc.push(p[0] - x0, p[1].ln(), p[1] * p[1]);
        }
    }
    if acc.count < 2 {
        return Err(FitError::InvalidDomain {
            reason: "exponential fit requires at least 2 points with y > 0",
        });
    }
    let (ln_a, b) = acc.solve()?;
    let a_u = ln_a.exp();
    if a_u.is_finite() && b.is_finite() {
        Ok([a_u, b, x0])
    } else {
        Err(FitError::InvalidDomain {
            reason: "non-finite exponential seed parameters",
        })
    }
}

/// Estimates closed-form initial parameters `[a, b]` for $y = a x^b$ ($x > 0, y > 0$).
#[verified_engine::verified]
pub fn seed_power_law<P: PointSeries + ?Sized>(points: &P) -> Result<[f64; 2], FitError> {
    seed_power_law_inner(points)
}

fn seed_power_law_inner<P: PointSeries + ?Sized>(points: &P) -> Result<[f64; 2], FitError> {
    let mut finite_n = 0usize;
    let mut acc = WeightedOlsAccum::default();
    for i in 0..points.len() {
        let p = points.point_at(i);
        if !p[0].is_finite() || !p[1].is_finite() {
            continue;
        }
        finite_n += 1;
        if p[0] > 0.0 && p[1] > 0.0 {
            acc.push(p[0].ln(), p[1].ln(), p[1] * p[1]);
        }
    }
    if finite_n < 2 {
        return Err(FitError::InsufficientPoints {
            required: 2,
            actual: finite_n,
        });
    }
    if acc.count < 2 {
        return Err(FitError::InvalidDomain {
            reason: "power-law fit requires at least 2 points with x > 0 and y > 0",
        });
    }
    let (ln_a, b) = acc.solve()?;
    let a = ln_a.exp();
    if a.is_finite() && b.is_finite() {
        Ok([a, b])
    } else {
        Err(FitError::InvalidDomain {
            reason: "non-finite power-law seed parameters",
        })
    }
}

/// Estimates exact closed-form parameters `[a, b]` for $y = a + b \ln(x)$ ($x > 0$).
#[verified_engine::verified]
pub fn seed_logarithmic<P: PointSeries + ?Sized>(points: &P) -> Result<[f64; 2], FitError> {
    seed_logarithmic_inner(points)
}

fn seed_logarithmic_inner<P: PointSeries + ?Sized>(points: &P) -> Result<[f64; 2], FitError> {
    let mut finite_n = 0usize;
    let mut acc = WeightedOlsAccum::default();
    for i in 0..points.len() {
        let p = points.point_at(i);
        if !p[0].is_finite() || !p[1].is_finite() {
            continue;
        }
        finite_n += 1;
        if p[0] > 0.0 {
            acc.push(p[0].ln(), p[1], 1.0);
        }
    }
    if finite_n < 2 {
        return Err(FitError::InsufficientPoints {
            required: 2,
            actual: finite_n,
        });
    }
    if acc.count < 2 {
        return Err(FitError::InvalidDomain {
            reason: "logarithmic fit requires at least 2 points with x > 0",
        });
    }
    let (a, b) = acc.solve()?;
    Ok([a, b])
}

/// Estimates initial parameters `[L, k, x0]` for $y = L / (1 + e^{-k(x - x_0)})$.
#[verified_engine::verified]
pub fn seed_logistic<P: PointSeries + ?Sized>(points: &P) -> Result<[f64; 3], FitError> {
    seed_logistic_inner(points)
}

fn seed_logistic_inner<P: PointSeries + ?Sized>(points: &P) -> Result<[f64; 3], FitError> {
    let (finite_n, x_min, x_max, y_max) = scan_logistic_bounds(points);
    if finite_n < 3 {
        return Err(FitError::InsufficientPoints {
            required: 3,
            actual: finite_n,
        });
    }
    if !y_max.is_finite() || y_max <= 0.0 {
        return Err(FitError::InvalidDomain {
            reason: "logistic fit requires positive response values",
        });
    }
    let l_est = (1.05 * y_max).max(1e-9);
    let mut acc = WeightedOlsAccum::default();
    for i in 0..points.len() {
        let p = points.point_at(i);
        if p[0].is_finite() && p[1].is_finite() && p[1] > 0.0 && p[1] < l_est {
            let ratio = (l_est / p[1]) - 1.0;
            if ratio > 0.0 {
                let w = (p[1] * (l_est - p[1])).powi(2);
                acc.push(p[0] - x_min, ratio.ln(), w);
            }
        }
    }
    let x_mid = 0.5 * (x_min + x_max);
    if let Ok((intercept, slope)) = acc.solve() {
        let k = (-slope).clamp(1e-4, 1e4);
        let x0 = if k.abs() > 1e-9 {
            x_min + intercept / k
        } else {
            x_mid
        };
        Ok([l_est, k, x0])
    } else {
        let span = (x_max - x_min).abs().max(1.0);
        Ok([l_est, 4.0 / span, x_mid])
    }
}

fn scan_logistic_bounds<P: PointSeries + ?Sized>(points: &P) -> (usize, f64, f64, f64) {
    let mut finite_n = 0usize;
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for i in 0..points.len() {
        let p = points.point_at(i);
        if p[0].is_finite() && p[1].is_finite() {
            finite_n += 1;
            x_min = x_min.min(p[0]);
            x_max = x_max.max(p[0]);
            y_max = y_max.max(p[1]);
        }
    }
    (finite_n, x_min, x_max, y_max)
}

/// Estimates initial parameters `[A, B, C, D]` for $y = A \sin(B x + C) + D$ on the stack.
#[verified_engine::verified]
pub fn seed_sinusoidal<P: PointSeries + ?Sized>(points: &P) -> Result<[f64; 4], FitError> {
    seed_sinusoidal_inner(points)
}

fn seed_sinusoidal_inner<P: PointSeries + ?Sized>(points: &P) -> Result<[f64; 4], FitError> {
    let mut sample = [[0.0_f64; 2]; MAX_SIN_SAMPLES];
    let (finite_n, sample_len) = sample_sorted_points(points, &mut sample);
    if finite_n < 4 || sample_len < 4 {
        return Err(FitError::InsufficientPoints {
            required: 4,
            actual: finite_n,
        });
    }
    let sub = &sample[..sample_len];
    let span_x = sub[sample_len - 1][0] - sub[0][0];
    if !span_x.is_finite() || span_x <= 1e-12 {
        return Err(FitError::InvalidDomain {
            reason: "sinusoidal fit requires non-degenerate x span",
        });
    }

    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    let mut y_sum = 0.0_f64;
    for &[_, y] in sub {
        y_min = y_min.min(y);
        y_max = y_max.max(y);
        y_sum += y;
    }
    let mean_y = y_sum / (sample_len as f64);
    let amp_fallback = (0.5 * (y_max - y_min)).max(1e-6);

    let x0 = sub[0][0];
    let freqs = candidate_frequencies(sub, mean_y, span_x);
    let (best_b, best_lin) = search_best_harmonic(sub, &freqs, x0);
    if let Some((alpha, beta, d)) = best_lin {
        let amp = (alpha * alpha + beta * beta).sqrt().max(1e-6);
        let mut phase = (alpha.atan2(beta) - best_b * x0).rem_euclid(std::f64::consts::TAU);
        if phase > std::f64::consts::PI {
            phase -= std::f64::consts::TAU;
        }
        Ok([amp, best_b, phase, d])
    } else {
        Ok([
            amp_fallback,
            (std::f64::consts::TAU / span_x).max(1e-4),
            0.0,
            mean_y,
        ])
    }
}

fn sample_sorted_points<P: PointSeries + ?Sized>(
    points: &P,
    out: &mut [[f64; 2]; MAX_SIN_SAMPLES],
) -> (usize, usize) {
    let mut finite_n = 0usize;
    for i in 0..points.len() {
        let p = points.point_at(i);
        if p[0].is_finite() && p[1].is_finite() {
            finite_n += 1;
        }
    }
    if finite_n == 0 {
        return (0, 0);
    }
    let stride = (finite_n / MAX_SIN_SAMPLES).max(1);
    let mut seen = 0usize;
    let mut len = 0usize;
    for i in 0..points.len() {
        let p = points.point_at(i);
        if !p[0].is_finite() || !p[1].is_finite() {
            continue;
        }
        if seen.is_multiple_of(stride) && len < MAX_SIN_SAMPLES {
            out[len] = p;
            len += 1;
        }
        seen += 1;
    }
    for i in 1..len {
        let key = out[i];
        let mut j = i;
        while j > 0 && out[j - 1][0] > key[0] {
            out[j] = out[j - 1];
            j -= 1;
        }
        out[j] = key;
    }
    (finite_n, len)
}

fn candidate_frequencies(sub: &[[f64; 2]], mean_y: f64, span_x: f64) -> [f64; 67] {
    let mut freqs = [0.0_f64; 67];
    let b_min = (0.25 * std::f64::consts::TAU / span_x).max(1e-6);
    let max_cycles = ((sub.len() as f64) * 0.25).clamp(4.0, 24.0);
    let b_max = (max_cycles * std::f64::consts::TAU / span_x).max(b_min * 2.0);
    freqs[0] = jacquelin_frequency(sub, mean_y).unwrap_or(0.0);
    freqs[1] = zero_crossing_frequency(sub, mean_y, span_x).unwrap_or(0.0);
    freqs[2] = std::f64::consts::TAU / span_x;
    for (idx, slot) in freqs.iter_mut().enumerate().skip(3) {
        let t = (idx - 3) as f64 / 63.0;
        *slot = b_min + t * (b_max - b_min);
    }
    freqs
}

fn zero_crossing_frequency(sorted: &[[f64; 2]], mean_y: f64, span_x: f64) -> Option<f64> {
    let mut crossings = 0usize;
    for i in 1..sorted.len() {
        let y0 = sorted[i - 1][1] - mean_y;
        let y1 = sorted[i][1] - mean_y;
        if (y0 <= 0.0 && y1 > 0.0) || (y0 >= 0.0 && y1 < 0.0) {
            crossings += 1;
        }
    }
    (crossings >= 1).then_some(std::f64::consts::PI * (crossings as f64) / span_x)
}

fn search_best_harmonic(
    sub: &[[f64; 2]],
    freqs: &[f64; 67],
    x0: f64,
) -> (f64, Option<(f64, f64, f64)>) {
    let mut best_sse = f64::INFINITY;
    let mut best_b = freqs[2].max(1e-4);
    let mut best_lin = None;
    for &b in freqs {
        if !b.is_finite() || b <= 1e-9 {
            continue;
        }
        let Ok((alpha, beta, d, sse)) = solve_harmonic_linear(sub, b, x0) else {
            continue;
        };
        if sse.is_finite() && sse < best_sse {
            best_sse = sse;
            best_b = b;
            best_lin = Some((alpha, beta, d));
        }
    }
    (best_b, best_lin)
}

fn jacquelin_frequency(sorted: &[[f64; 2]], mean_y: f64) -> Option<f64> {
    let n = sorted.len();
    if n < 5 {
        return None;
    }
    let x0 = sorted[0][0];
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut m = [[0.0_f64; 5]; 5];
    let mut rhs = [0.0_f64; 5];

    for i in 0..n {
        if i > 0 {
            let dx = sorted[i][0] - sorted[i - 1][0];
            let next_s1 = s1 + 0.5 * ((sorted[i][1] - mean_y) + (sorted[i - 1][1] - mean_y)) * dx;
            s2 += 0.5 * (next_s1 + s1) * dx;
            s1 = next_s1;
        }
        let dx0 = sorted[i][0] - x0;
        let row = [s2, dx0 * dx0, dx0, 1.0];
        let y = sorted[i][1] - mean_y;
        for r in 0..4 {
            rhs[r] += row[r] * y;
            for c in r..4 {
                m[r][c] += row[r] * row[c];
            }
        }
    }
    symmetrize_upper(&mut m, 4);
    let sol = solve_dense_small_raw(&mut m, &rhs, 4).ok()?;
    let b_sq = -sol[0];
    (b_sq.is_finite() && b_sq > 1e-12).then(|| b_sq.sqrt())
}

fn solve_harmonic_linear(
    points: &[[f64; 2]],
    b: f64,
    x0: f64,
) -> Result<(f64, f64, f64, f64), FitError> {
    let mut mat = [[0.0_f64; 5]; 5];
    let mut rhs = [0.0_f64; 5];
    let mut sum_y2 = 0.0_f64;
    for &[x, y] in points {
        let (s, c) = (b * (x - x0)).sin_cos();
        let basis = [c, s, 1.0];
        sum_y2 += y * y;
        for r in 0..3 {
            rhs[r] += basis[r] * y;
            for col in r..3 {
                mat[r][col] += basis[r] * basis[col];
            }
        }
    }
    symmetrize_upper(&mut mat, 3);
    let rhs_orig = rhs;
    let sol = solve_dense_small_raw(&mut mat, &rhs, 3)?;
    let explained = sol[0] * rhs_orig[0] + sol[1] * rhs_orig[1] + sol[2] * rhs_orig[2];
    let sse = (sum_y2 - explained).max(0.0);
    Ok((sol[0], sol[1], sol[2], sse))
}
