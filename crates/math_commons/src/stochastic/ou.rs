//! Exact discrete-time AR(1) Ornstein-Uhlenbeck parameter calibration and zero-allocation
//! quantile/ensemble summary statistics.
// [cite:math_commons]

use math_scalars::RealField;

const MIN_THETA: f64 = 1e-6;
const CONSTANT_SERIES_VAR_TOL: f64 = 1e-30;

/// Calibrated Ornstein-Uhlenbeck parameters $(\mu, \theta, \sigma)$ for
/// $dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t$.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OuCalibration<T = f64> {
    /// Long-term mean level $\mu$.
    pub mu: T,
    /// Mean-reversion rate $\theta > 0$.
    pub theta: T,
    /// Diffusion volatility $\sigma \ge 0$.
    pub sigma: T,
}

/// Zero-allocation ensemble summary statistics over a 1D sample slice.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SampleSummary<T = f64> {
    /// Arithmetic sample mean.
    pub mean: T,
    /// Sample standard deviation ($N - 1$ denominator when $N > 1$, else `0.0`).
    pub std_dev: T,
    /// Sample median (50th percentile).
    pub median: T,
    /// Lower 5th percentile.
    pub percentile_05: T,
    /// Upper 95th percentile.
    pub percentile_95: T,
    /// Fraction of samples strictly exceeding `threshold`, or `0.0` when `threshold` is `None`.
    pub prob_above_threshold: T,
}

/// Calibrates Ornstein-Uhlenbeck parameters $(\mu, \theta, \sigma)$ from uniformly spaced
/// observations $X_0, \dots, X_{n-1}$ at step `dt > 0` via exact discrete-time AR(1) conditional
/// Gaussian likelihood ($X_{i+1} = c + \phi X_i + \varepsilon_i$, $\phi = e^{-\theta \Delta t}$)
/// with stationary diffusion scaling $\sigma = \sqrt{2\theta \cdot \hat{V}_{\mathrm{stat}}}$.
///
/// Returns `None` when `observations.len() < 3`, `dt <= 0.0`, or any input is non-finite.
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn calibrate_ou<T: RealField>(observations: &[T], dt: T) -> Option<OuCalibration<T>> {
    debug_assert!(observations.len() < 3 || observations.len() >= 3);
    let n = observations.len();
    if n < 3 || !dt.is_finite() || dt <= T::ZERO || !observations.iter().copied().all(T::is_finite)
    {
        return None;
    }

    let n_scalar = T::from_usize(n);
    let mean_all = observations.iter().copied().fold(T::ZERO, |a, b| a + b) / n_scalar;
    let centered_ss = observations
        .iter()
        .map(|&x| (x - mean_all) * (x - mean_all))
        .fold(T::ZERO, |a, b| a + b);
    let sample_var = centered_ss / T::from_usize(n - 1);

    if sample_var <= T::from_f64(CONSTANT_SERIES_VAR_TOL) {
        return Some(OuCalibration {
            mu: mean_all,
            theta: T::ONE,
            sigma: T::ZERO,
        });
    }

    Some(calibrate_nonconstant_ar1(
        observations,
        dt,
        mean_all,
        centered_ss,
        sample_var,
    ))
}

#[cfg_attr(feature = "std", verified_engine::verified)]
fn calibrate_nonconstant_ar1<T: RealField>(
    observations: &[T],
    dt: T,
    mean_all: T,
    centered_ss: T,
    sample_var: T,
) -> OuCalibration<T> {
    let n = observations.len();
    let m = n - 1;
    debug_assert!(m >= 2 && dt > T::ZERO);

    let x_slice = &observations[..m];
    let y_slice = &observations[1..];
    let m_scalar = T::from_usize(m);
    let mean_x = x_slice.iter().copied().fold(T::ZERO, |a, b| a + b) / m_scalar;
    let mean_y = y_slice.iter().copied().fold(T::ZERO, |a, b| a + b) / m_scalar;

    let mut s_xx = T::ZERO;
    let mut s_xy = T::ZERO;
    let mut c_1 = T::ZERO;
    let mut diff_ss = T::ZERO;
    for (&xi, &yi) in x_slice.iter().zip(y_slice.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        s_xx += dx * dx;
        s_xy += dx * dy;
        c_1 += (xi - mean_all) * (yi - mean_all);
        let step_d = yi - xi;
        diff_ss += step_d * step_d;
    }

    let r1 = c_1 / centered_ss;
    let phi_ols = if s_xx > T::from_f64(CONSTANT_SERIES_VAR_TOL) {
        s_xy / s_xx
    } else {
        r1
    };
    let eps6 = T::from_f64(1e-6);
    let min_theta = T::from_f64(MIN_THETA);

    if phi_ols > eps6 && phi_ols < T::ONE - eps6 {
        calibrate_interior_ar1(
            x_slice,
            y_slice,
            dt,
            (mean_all, mean_x, mean_y),
            phi_ols,
            sample_var,
        )
    } else if phi_ols >= T::ONE - eps6 {
        let phi_yw = r1.clamp(eps6, T::ONE - eps6);
        let theta = (-phi_yw.ln() / dt).max(min_theta);
        let sigma = ((diff_ss / m_scalar) / dt).sqrt();
        OuCalibration {
            mu: mean_all,
            theta,
            sigma: sigma.max(T::ZERO),
        }
    } else {
        let phi_pos = r1.clamp(T::from_f64(1e-3), T::ONE - eps6);
        let theta = (-phi_pos.ln() / dt).max(min_theta);
        let sigma = (T::from_f64(2.0) * theta * sample_var).sqrt();
        OuCalibration {
            mu: mean_all,
            theta,
            sigma: sigma.max(T::ZERO),
        }
    }
}

#[cfg_attr(feature = "std", verified_engine::verified)]
fn calibrate_interior_ar1<T: RealField>(
    x_slice: &[T],
    y_slice: &[T],
    dt: T,
    means: (T, T, T),
    phi: T,
    sample_var: T,
) -> OuCalibration<T> {
    let m = x_slice.len();
    debug_assert!(m >= 2 && phi > T::ZERO && phi < T::ONE);
    let (mean_all, mean_x, mean_y) = means;
    let theta = (-phi.ln() / dt).max(T::from_f64(MIN_THETA));

    let mu = if T::ONE - phi >= T::from_f64(0.05) {
        (mean_y - phi * mean_x) / (T::ONE - phi)
    } else {
        mean_all
    };

    let c_hat = mean_y - phi * mean_x;
    let rss = x_slice
        .iter()
        .zip(y_slice.iter())
        .map(|(&xi, &yi)| {
            let eps = yi - (c_hat + phi * xi);
            eps * eps
        })
        .fold(T::ZERO, |a, b| a + b);
    let dof = T::from_usize(if m > 2 { m - 2 } else { m });
    let sigma_eps_sq = rss / dof;

    let two = T::from_f64(2.0);
    let one_minus_exp = -(-two * theta * dt).expm1();
    let sigma_sq = if one_minus_exp > T::ZERO && sigma_eps_sq > T::ZERO {
        sigma_eps_sq * (two * theta / one_minus_exp)
    } else {
        two * theta * sample_var
    };

    OuCalibration {
        mu,
        theta,
        sigma: sigma_sq.max(T::ZERO).sqrt(),
    }
}

/// Computes summary statistics and empirical quantiles over `samples` using `scratch` for
/// in-place sorting with zero heap allocations.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub(super) fn summarize_samples_into<T: RealField>(
    samples: &[T],
    threshold: Option<T>,
    scratch: &mut [T],
) -> Option<SampleSummary<T>> {
    let n = samples.len();
    debug_assert!(n > 0 && scratch.len() >= n);
    if n == 0 || scratch.len() < n || !samples.iter().copied().all(T::is_finite) {
        return None;
    }

    let sorted = &mut scratch[..n];
    sorted.copy_from_slice(samples);
    sorted.sort_unstable_by(T::total_cmp);

    let n_scalar = T::from_usize(n);
    let mean = sorted.iter().copied().fold(T::ZERO, |a, b| a + b) / n_scalar;
    let std_dev = if n > 1 {
        let var = sorted
            .iter()
            .map(|&v| (v - mean) * (v - mean))
            .fold(T::ZERO, |a, b| a + b)
            / T::from_usize(n - 1);
        var.max(T::ZERO).sqrt()
    } else {
        T::ZERO
    };

    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        T::from_f64(0.5) * (sorted[n / 2 - 1] + sorted[n / 2])
    };
    let idx05 = (((n - 1) as f64) * 0.05) as usize;
    let idx95 = (((n - 1) as f64) * 0.95) as usize;

    let prob_above_threshold = match threshold {
        Some(th) => {
            let above = sorted.iter().filter(|&&v| v > th).count();
            T::from_usize(above) / n_scalar
        }
        None => T::ZERO,
    };

    Some(SampleSummary {
        mean,
        std_dev,
        median,
        percentile_05: sorted[idx05],
        percentile_95: sorted[idx95],
        prob_above_threshold,
    })
}
