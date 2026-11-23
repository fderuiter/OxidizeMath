//! Zero-allocation mean and median convenience wrappers for [`super::bootstrap`].

use super::{
    bca_bootstrap, finalize_sorted_interval, percentile_bootstrap, s3_over_s2_accel,
    validate_bootstrap_args, BootstrapInterval,
};
use crate::error::ProbError;
use crate::math::{mean_and_sample_variance, sort_total_f64, sorted_quantile};
use crate::workspace::SolverWorkspace;
use oxidize_core::rng::OxidizeRng;

fn slice_mean(xs: &[f64]) -> f64 {
    xs.iter().copied().sum::<f64>() / (xs.len() as f64)
}

fn slice_median_in_place(xs: &mut [f64]) -> f64 {
    let n = xs.len();
    let mid = n / 2;
    let (left, mid_val, _) = xs.select_nth_unstable_by(mid, f64::total_cmp);
    if n % 2 == 1 {
        *mid_val
    } else {
        let max_left = left.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (max_left + *mid_val)
    }
}

/// Zero-allocation percentile bootstrap confidence interval for the arithmetic mean of `sample`.
pub fn percentile_bootstrap_mean(
    sample: &[f64],
    n_resamples: usize,
    confidence: f64,
    rng: &mut OxidizeRng,
    workspace: &mut SolverWorkspace<'_>,
) -> Result<BootstrapInterval, ProbError> {
    percentile_bootstrap(sample, n_resamples, confidence, rng, workspace, slice_mean)
}

/// Zero-allocation Efron BCa bootstrap confidence interval for the arithmetic mean of `sample`.
pub fn bca_bootstrap_mean(
    sample: &[f64],
    n_resamples: usize,
    confidence: f64,
    rng: &mut OxidizeRng,
    workspace: &mut SolverWorkspace<'_>,
) -> Result<BootstrapInterval, ProbError> {
    bca_bootstrap(sample, n_resamples, confidence, rng, workspace, slice_mean)
}

/// Zero-allocation percentile bootstrap confidence interval for the sample median of `sample`.
pub fn percentile_bootstrap_median(
    sample: &[f64],
    n_resamples: usize,
    confidence: f64,
    rng: &mut OxidizeRng,
    workspace: &mut SolverWorkspace<'_>,
) -> Result<BootstrapInterval, ProbError> {
    let n = sample.len();
    validate_bootstrap_args(n, n_resamples, confidence)?;
    if sample.iter().any(|x| !x.is_finite()) {
        return Err(ProbError::NonFiniteInput {
            context: "bootstrap sample",
        });
    }
    let (reps, mut rem) = workspace.split_floats(n_resamples)?;
    let (draw_buf, _) = rem.split_floats(n)?;

    draw_buf.copy_from_slice(sample);
    let point_estimate = slice_median_in_place(draw_buf);

    for rep_slot in reps.iter_mut() {
        for slot in draw_buf.iter_mut() {
            *slot = sample[rng.next_index(n)];
        }
        *rep_slot = slice_median_in_place(draw_buf);
    }
    let (_, rep_var) = mean_and_sample_variance(reps)?;
    let std_error = libm::sqrt(rep_var);
    sort_total_f64(reps);
    let alpha_half = 0.5 * (1.0 - confidence);
    Ok(BootstrapInterval {
        point_estimate,
        lower: sorted_quantile(reps, alpha_half),
        upper: sorted_quantile(reps, 1.0 - alpha_half),
        std_error,
        confidence,
        bias_z0: 0.0,
        acceleration: 0.0,
    })
}

fn sorted_median_jackknife_accel(sorted: &[f64], jack_buf: &mut [f64]) -> f64 {
    let n = sorted.len();
    if n < 3 {
        return 0.0;
    }
    let mut sum_j = 0.0;
    if (n - 1).is_multiple_of(2) {
        let m = n / 2;
        for (k, slot) in jack_buf.iter_mut().enumerate().take(n) {
            let jv = if k < m {
                0.5 * (sorted[m] + sorted[m + 1])
            } else if k == m {
                0.5 * (sorted[m - 1] + sorted[m + 1])
            } else {
                0.5 * (sorted[m - 1] + sorted[m])
            };
            *slot = jv;
            sum_j += jv;
        }
    } else {
        let m = (n - 1) / 2;
        for (k, slot) in jack_buf.iter_mut().enumerate().take(n) {
            let jv = if k <= m { sorted[m + 1] } else { sorted[m] };
            *slot = jv;
            sum_j += jv;
        }
    }
    s3_over_s2_accel(&jack_buf[..n], sum_j / (n as f64))
}

/// Zero-allocation Efron BCa bootstrap confidence interval for the sample median of `sample`.
pub fn bca_bootstrap_median(
    sample: &[f64],
    n_resamples: usize,
    confidence: f64,
    rng: &mut OxidizeRng,
    workspace: &mut SolverWorkspace<'_>,
) -> Result<BootstrapInterval, ProbError> {
    let n = sample.len();
    validate_bootstrap_args(n, n_resamples, confidence)?;
    if sample.iter().any(|x| !x.is_finite()) {
        return Err(ProbError::NonFiniteInput {
            context: "bootstrap sample",
        });
    }
    let (reps, mut rem1) = workspace.split_floats(n_resamples)?;
    let (draw_buf, mut rem2) = rem1.split_floats(n)?;
    let (jack_buf, _) = rem2.split_floats(n)?;

    draw_buf.copy_from_slice(sample);
    sort_total_f64(draw_buf);
    let point_estimate = if n.is_multiple_of(2) {
        0.5 * (draw_buf[n / 2 - 1] + draw_buf[n / 2])
    } else {
        draw_buf[n / 2]
    };
    let accel = sorted_median_jackknife_accel(draw_buf, jack_buf);

    for rep_slot in reps.iter_mut() {
        for slot in draw_buf.iter_mut() {
            *slot = sample[rng.next_index(n)];
        }
        *rep_slot = slice_median_in_place(draw_buf);
    }
    finalize_sorted_interval(reps, point_estimate, confidence, Some(accel))
}
