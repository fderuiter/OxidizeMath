//! Zero-allocation closed-form parameter seed estimators for nonlinear curve models.
// [cite:math_commons]

use super::poly::{solve_dense_small_raw, symmetrize_upper};
use crate::error::FitError;
use crate::series::PointSeries;
use math_scalars::RealField;

const MAX_SIN_SAMPLES: usize = 256;

/// Streaming accumulator for weighted linear regression $v = \alpha + \beta u$ on the stack.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WeightedOlsAccum<T = f64> {
    sw: T,
    su: T,
    sv: T,
    suu: T,
    suv: T,
    count: usize,
}

impl<T: RealField> WeightedOlsAccum<T> {
    /// Number of valid weighted observations accumulated so far.
    #[inline]
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        self.count
    }

    /// Adds a weighted `(u, v)` sample with positive finite weight `w`.
    #[inline]
    pub fn push(&mut self, u: T, v: T, w: T) {
        if u.is_finite() && v.is_finite() && w.is_finite() && w > T::ZERO {
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
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn solve(&self) -> Result<(T, T), FitError> {
        if self.count < 2 {
            return Err(FitError::InsufficientPoints {
                required: 2,
                actual: self.count,
            });
        }
        let det = self.sw * self.suu - self.su * self.su;
        let det_tol = (T::EPSILON * T::from_f64(16.0)).max(T::from_f64(1e-15));
        if !det.is_finite() || det.abs() <= det_tol {
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
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn weighted_ols_line<T: RealField>(samples: &[[T; 3]]) -> Result<(T, T), FitError> {
    let mut acc = WeightedOlsAccum::<T>::default();
    for &[u, v, w] in samples {
        acc.push(u, v, w);
    }
    acc.solve()
}

/// Estimates closed-form initial parameters `[a, b]` for $y = a e^{b x}$ via $y^2$-weighted log-OLS.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn seed_exponential<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
) -> Result<[T; 2], FitError> {
    let [a_u, b, x0] = seed_exponential_shifted(points)?;
    let lim = T::MAX_VALUE.ln() - T::TWO;
    let a = a_u * (-b * x0).clamp(-lim, lim).exp();
    if a.is_finite() && b.is_finite() {
        Ok([a, b])
    } else {
        Err(FitError::InvalidDomain {
            reason: "non-finite exponential seed parameters",
        })
    }
}

pub(super) fn seed_exponential_shifted<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
) -> Result<[T; 3], FitError> {
    let mut finite_n = 0usize;
    let mut x_min = T::INFINITY;
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
    let x0 = if x_min.is_finite() { x_min } else { T::ZERO };
    let mut acc = WeightedOlsAccum::<T>::default();
    for i in 0..points.len() {
        let p = points.point_at(i);
        if p[0].is_finite() && p[1].is_finite() && p[1] > T::ZERO {
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
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn seed_power_law<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
) -> Result<[T; 2], FitError> {
    let mut finite_n = 0usize;
    let mut acc = WeightedOlsAccum::<T>::default();
    for i in 0..points.len() {
        let p = points.point_at(i);
        if !p[0].is_finite() || !p[1].is_finite() {
            continue;
        }
        finite_n += 1;
        if p[0] > T::ZERO && p[1] > T::ZERO {
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
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn seed_logarithmic<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
) -> Result<[T; 2], FitError> {
    let mut finite_n = 0usize;
    let mut acc = WeightedOlsAccum::<T>::default();
    for i in 0..points.len() {
        let p = points.point_at(i);
        if !p[0].is_finite() || !p[1].is_finite() {
            continue;
        }
        finite_n += 1;
        if p[0] > T::ZERO {
            acc.push(p[0].ln(), p[1], T::ONE);
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
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn seed_logistic<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
) -> Result<[T; 3], FitError> {
    let (finite_n, x_min, x_max, y_max) = scan_logistic_bounds(points);
    if finite_n < 3 {
        return Err(FitError::InsufficientPoints {
            required: 3,
            actual: finite_n,
        });
    }
    if !y_max.is_finite() || y_max <= T::ZERO {
        return Err(FitError::InvalidDomain {
            reason: "logistic fit requires positive response values",
        });
    }
    let l_est = (T::from_f64(1.05) * y_max).max(T::from_f64(1e-9));
    let mut acc = WeightedOlsAccum::<T>::default();
    for i in 0..points.len() {
        let p = points.point_at(i);
        if p[0].is_finite() && p[1].is_finite() && p[1] > T::ZERO && p[1] < l_est {
            let ratio = (l_est / p[1]) - T::ONE;
            if ratio > T::ZERO {
                let w = (p[1] * (l_est - p[1])).powi(2);
                acc.push(p[0] - x_min, ratio.ln(), w);
            }
        }
    }
    let x_mid = T::HALF * (x_min + x_max);
    if let Ok((intercept, slope)) = acc.solve() {
        let k = (-slope).clamp(T::from_f64(1e-4), T::from_f64(1e4));
        let x0 = if k.abs() > T::from_f64(1e-9) {
            x_min + intercept / k
        } else {
            x_mid
        };
        Ok([l_est, k, x0])
    } else {
        let span = (x_max - x_min).abs().max(T::ONE);
        Ok([l_est, T::from_f64(4.0) / span, x_mid])
    }
}

fn scan_logistic_bounds<T: RealField, P: PointSeries<T> + ?Sized>(points: &P) -> (usize, T, T, T) {
    let mut finite_n = 0usize;
    let mut x_min = T::INFINITY;
    let mut x_max = T::NEG_INFINITY;
    let mut y_max = T::NEG_INFINITY;
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
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn seed_sinusoidal<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
) -> Result<[T; 4], FitError> {
    let mut sample = [[T::ZERO; 2]; MAX_SIN_SAMPLES];
    let (finite_n, sample_len) = sample_sorted_points(points, &mut sample);
    if finite_n < 4 || sample_len < 4 {
        return Err(FitError::InsufficientPoints {
            required: 4,
            actual: finite_n,
        });
    }
    let sub = &sample[..sample_len];
    let span_x = sub[sample_len - 1][0] - sub[0][0];
    if !span_x.is_finite() || span_x <= T::from_f64(1e-12) {
        return Err(FitError::InvalidDomain {
            reason: "sinusoidal fit requires non-degenerate x span",
        });
    }

    let mut y_min = T::INFINITY;
    let mut y_max = T::NEG_INFINITY;
    let mut y_sum = T::ZERO;
    for &[_, y] in sub {
        y_min = y_min.min(y);
        y_max = y_max.max(y);
        y_sum += y;
    }
    let mean_y = y_sum / T::from_usize(sample_len);
    let amp_fallback = (T::HALF * (y_max - y_min)).max(T::from_f64(1e-6));

    let x0 = sub[0][0];
    let freqs = candidate_frequencies(sub, mean_y, span_x);
    let (best_b, best_lin) = search_best_harmonic(sub, &freqs, x0);
    if let Some((alpha, beta, d)) = best_lin {
        let amp = alpha.hypot(beta).max(T::from_f64(1e-6));
        let mut phase = (alpha.atan2(beta) - best_b * x0).rem_euclid(T::TAU);
        if phase > T::PI {
            phase -= T::TAU;
        }
        Ok([amp, best_b, phase, d])
    } else {
        Ok([
            amp_fallback,
            (T::TAU / span_x).max(T::from_f64(1e-4)),
            T::ZERO,
            mean_y,
        ])
    }
}

fn sample_sorted_points<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
    out: &mut [[T; 2]; MAX_SIN_SAMPLES],
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

fn candidate_frequencies<T: RealField>(sub: &[[T; 2]], mean_y: T, span_x: T) -> [T; 67] {
    let mut freqs = [T::ZERO; 67];
    let b_min = (T::from_f64(0.25) * T::TAU / span_x).max(T::from_f64(1e-6));
    let max_cycles =
        (T::from_usize(sub.len()) * T::from_f64(0.25)).clamp(T::from_f64(4.0), T::from_f64(24.0));
    let b_max = (max_cycles * T::TAU / span_x).max(b_min * T::TWO);
    freqs[0] = jacquelin_frequency(sub, mean_y).unwrap_or(T::ZERO);
    freqs[1] = zero_crossing_frequency(sub, mean_y, span_x).unwrap_or(T::ZERO);
    freqs[2] = T::TAU / span_x;
    let inv63 = T::ONE / T::from_usize(63);
    for (idx, slot) in freqs.iter_mut().enumerate().skip(3) {
        let t = T::from_usize(idx - 3) * inv63;
        *slot = b_min + t * (b_max - b_min);
    }
    freqs
}

fn zero_crossing_frequency<T: RealField>(sorted: &[[T; 2]], mean_y: T, span_x: T) -> Option<T> {
    let mut crossings = 0usize;
    for i in 1..sorted.len() {
        let y0 = sorted[i - 1][1] - mean_y;
        let y1 = sorted[i][1] - mean_y;
        if (y0 <= T::ZERO && y1 > T::ZERO) || (y0 >= T::ZERO && y1 < T::ZERO) {
            crossings += 1;
        }
    }
    (crossings >= 1).then_some(T::PI * T::from_usize(crossings) / span_x)
}

fn search_best_harmonic<T: RealField>(
    sub: &[[T; 2]],
    freqs: &[T; 67],
    x0: T,
) -> (T, Option<(T, T, T)>) {
    let mut best_sse = T::INFINITY;
    let mut best_b = freqs[2].max(T::from_f64(1e-4));
    let mut best_lin = None;
    for &b in freqs {
        if !b.is_finite() || b <= T::from_f64(1e-9) {
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

fn jacquelin_frequency<T: RealField>(sorted: &[[T; 2]], mean_y: T) -> Option<T> {
    let n = sorted.len();
    if n < 5 {
        return None;
    }
    let x0 = sorted[0][0];
    let mut s1 = T::ZERO;
    let mut s2 = T::ZERO;
    let mut m = [[T::ZERO; 5]; 5];
    let mut rhs = [T::ZERO; 5];

    for i in 0..n {
        if i > 0 {
            let dx = sorted[i][0] - sorted[i - 1][0];
            let next_s1 =
                s1 + T::HALF * ((sorted[i][1] - mean_y) + (sorted[i - 1][1] - mean_y)) * dx;
            s2 += T::HALF * (next_s1 + s1) * dx;
            s1 = next_s1;
        }
        let dx0 = sorted[i][0] - x0;
        let row = [s2, dx0 * dx0, dx0, T::ONE];
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
    (b_sq.is_finite() && b_sq > T::from_f64(1e-12)).then(|| b_sq.sqrt())
}

fn solve_harmonic_linear<T: RealField>(
    points: &[[T; 2]],
    b: T,
    x0: T,
) -> Result<(T, T, T, T), FitError> {
    let mut mat = [[T::ZERO; 5]; 5];
    let mut rhs = [T::ZERO; 5];
    let mut sum_y2 = T::ZERO;
    for &[x, y] in points {
        let (s, c) = (b * (x - x0)).sin_cos();
        let basis = [c, s, T::ONE];
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
    let sse = (sum_y2 - explained).max(T::ZERO);
    Ok((sol[0], sol[1], sol[2], sse))
}
