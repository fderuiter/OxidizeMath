//! Exact discrete-time AR(1) Ornstein-Uhlenbeck parameter calibration and zero-allocation
//! quantile/ensemble summary statistics.
// [cite:math_commons]

const MIN_THETA: f64 = 1e-6;
const CONSTANT_SERIES_VAR_TOL: f64 = 1e-30;

/// Calibrated Ornstein-Uhlenbeck parameters $(\mu, \theta, \sigma)$ for
/// $dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OuCalibration {
    /// Long-term mean level $\mu$.
    pub mu: f64,
    /// Mean-reversion rate $\theta > 0$.
    pub theta: f64,
    /// Diffusion volatility $\sigma \ge 0$.
    pub sigma: f64,
}

/// Zero-allocation ensemble summary statistics over a 1D sample slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleSummary {
    /// Arithmetic sample mean.
    pub mean: f64,
    /// Sample standard deviation ($N - 1$ denominator when $N > 1$, else `0.0`).
    pub std_dev: f64,
    /// Sample median (50th percentile).
    pub median: f64,
    /// Lower 5th percentile.
    pub percentile_05: f64,
    /// Upper 95th percentile.
    pub percentile_95: f64,
    /// Fraction of samples strictly exceeding `threshold`, or `0.0` when `threshold` is `None`.
    pub prob_above_threshold: f64,
}

/// Calibrates Ornstein-Uhlenbeck parameters $(\mu, \theta, \sigma)$ from uniformly spaced
/// observations $X_0, \dots, X_{n-1}$ at step `dt > 0` via exact discrete-time AR(1) conditional
/// Gaussian likelihood ($X_{i+1} = c + \phi X_i + \varepsilon_i$, $\phi = e^{-\theta \Delta t}$)
/// with stationary diffusion scaling $\sigma = \sqrt{2\theta \cdot \hat{V}_{\mathrm{stat}}}$.
///
/// Returns `None` when `observations.len() < 3`, `dt <= 0.0`, or any input is non-finite.
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn calibrate_ou(observations: &[f64], dt: f64) -> Option<OuCalibration> {
    debug_assert!(observations.len() < 3 || observations.len() >= 3);
    let n = observations.len();
    if n < 3 || !dt.is_finite() || dt <= 0.0 || !observations.iter().all(|x| x.is_finite()) {
        return None;
    }

    let mean_all = observations.iter().copied().sum::<f64>() / (n as f64);
    let centered_ss = observations
        .iter()
        .map(|&x| (x - mean_all) * (x - mean_all))
        .sum::<f64>();
    let sample_var = centered_ss / ((n - 1) as f64);

    if sample_var <= CONSTANT_SERIES_VAR_TOL {
        return Some(OuCalibration {
            mu: mean_all,
            theta: 1.0,
            sigma: 0.0,
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
fn calibrate_nonconstant_ar1(
    observations: &[f64],
    dt: f64,
    mean_all: f64,
    centered_ss: f64,
    sample_var: f64,
) -> OuCalibration {
    let n = observations.len();
    let m = n - 1;
    debug_assert!(m >= 2 && dt > 0.0);

    let x_slice = &observations[..m];
    let y_slice = &observations[1..];
    let mean_x = x_slice.iter().copied().sum::<f64>() / (m as f64);
    let mean_y = y_slice.iter().copied().sum::<f64>() / (m as f64);

    let mut s_xx = 0.0_f64;
    let mut s_xy = 0.0_f64;
    let mut c_1 = 0.0_f64;
    let mut diff_ss = 0.0_f64;
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
    let phi_ols = if s_xx > CONSTANT_SERIES_VAR_TOL {
        s_xy / s_xx
    } else {
        r1
    };

    if phi_ols > 1e-6 && phi_ols < 1.0 - 1e-6 {
        calibrate_interior_ar1(
            x_slice,
            y_slice,
            dt,
            (mean_all, mean_x, mean_y),
            phi_ols,
            sample_var,
        )
    } else if phi_ols >= 1.0 - 1e-6 {
        let phi_yw = r1.clamp(1e-6, 1.0 - 1e-6);
        let theta = (-libm::log(phi_yw) / dt).max(MIN_THETA);
        let sigma = libm::sqrt((diff_ss / (m as f64)) / dt);
        OuCalibration {
            mu: mean_all,
            theta,
            sigma: sigma.max(0.0),
        }
    } else {
        // Ultra-fast mean reversion (e.g. theta * dt >> 1 where lag-1 correlation ~ 0):
        // stationary variance is sigma^2 / (2 * theta), so sigma = sqrt(2 * theta * sample_var).
        let phi_pos = r1.clamp(1e-3, 1.0 - 1e-6);
        let theta = (-libm::log(phi_pos) / dt).max(MIN_THETA);
        let sigma = libm::sqrt(2.0 * theta * sample_var);
        OuCalibration {
            mu: mean_all,
            theta,
            sigma: sigma.max(0.0),
        }
    }
}

#[cfg_attr(feature = "std", verified_engine::verified)]
fn calibrate_interior_ar1(
    x_slice: &[f64],
    y_slice: &[f64],
    dt: f64,
    means: (f64, f64, f64),
    phi: f64,
    sample_var: f64,
) -> OuCalibration {
    let m = x_slice.len();
    debug_assert!(m >= 2 && phi > 0.0 && phi < 1.0);
    let (mean_all, mean_x, mean_y) = means;
    let theta = (-libm::log(phi) / dt).max(MIN_THETA);

    let mu = if 1.0 - phi >= 0.05 {
        (mean_y - phi * mean_x) / (1.0 - phi)
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
        .sum::<f64>();
    let dof = if m > 2 { (m - 2) as f64 } else { m as f64 };
    let sigma_eps_sq = rss / dof;

    // Exact transition variance: sigma_eps^2 = (sigma^2 / (2 * theta)) * (1 - exp(-2 * theta * dt))
    // Equivalently sigma = sqrt(2 * theta * V_stat) with V_stat = sigma_eps^2 / (1 - phi^2).
    let one_minus_exp = -libm::expm1(-2.0 * theta * dt);
    let sigma_sq = if one_minus_exp > 0.0 && sigma_eps_sq > 0.0 {
        sigma_eps_sq * (2.0 * theta / one_minus_exp)
    } else {
        2.0 * theta * sample_var
    };

    OuCalibration {
        mu,
        theta,
        sigma: libm::sqrt(sigma_sq.max(0.0)),
    }
}

/// Computes summary statistics and empirical quantiles over `samples` using `scratch` for
/// in-place sorting with zero heap allocations.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub(super) fn summarize_samples_into(
    samples: &[f64],
    threshold: Option<f64>,
    scratch: &mut [f64],
) -> Option<SampleSummary> {
    let n = samples.len();
    debug_assert!(n > 0 && scratch.len() >= n);
    if n == 0 || scratch.len() < n || !samples.iter().all(|x| x.is_finite()) {
        return None;
    }

    let sorted = &mut scratch[..n];
    sorted.copy_from_slice(samples);
    sorted.sort_unstable_by(f64::total_cmp);

    let mean = sorted.iter().copied().sum::<f64>() / (n as f64);
    let std_dev = if n > 1 {
        let var = sorted.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / ((n - 1) as f64);
        libm::sqrt(var.max(0.0))
    } else {
        0.0
    };

    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    };
    let idx05 = (((n - 1) as f64) * 0.05) as usize;
    let idx95 = (((n - 1) as f64) * 0.95) as usize;

    let prob_above_threshold = match threshold {
        Some(th) => {
            let above = sorted.iter().filter(|&&v| v > th).count();
            (above as f64) / (n as f64)
        }
        None => 0.0,
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
