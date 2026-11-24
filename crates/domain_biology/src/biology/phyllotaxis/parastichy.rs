//! Opposing parastichy spiral pair $(m, n)$ analysis and Fibonacci transition verification.

use super::vogel::PrimordiumPoint;

/// Summary of visible opposing parastichy spiral families and packing uniformity at outer radius $R_N$.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ParastichyAnalysis {
    /// Lower parastichy spiral count $\min(m, n)$ (e.g. $F_k$).
    pub primary_count: usize,
    /// Higher opposing parastichy spiral count $\max(m, n)$ (e.g. $F_{k+1}$).
    pub secondary_count: usize,
    /// Step size of the clockwise-winding spiral family.
    pub clockwise_step: usize,
    /// Step size of the counter-clockwise-winding spiral family.
    pub counter_clockwise_step: usize,
    /// `true` when `(primary_count, secondary_count)` form consecutive Fibonacci numbers $(F_k, F_{k+1})$.
    pub is_fibonacci_pair: bool,
    /// `true` when a rational divergence angle collapses neighbors onto exact radial orthostichies.
    pub has_radial_spokes: bool,
    /// Normalized nearest-neighbor packing efficiency in $(0, 1]$ relative to hexagonal disc packing.
    pub packing_uniformity: f64,
}

impl Default for ParastichyAnalysis {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            primary_count: 1,
            secondary_count: 1,
            clockwise_step: 1,
            counter_clockwise_step: 1,
            is_fibonacci_pair: true,
            has_radial_spokes: false,
            packing_uniformity: 1.0,
        }
    }
}

/// Checks whether `(a, b)` (in ascending order) are consecutive Fibonacci numbers $(F_k, F_{k+1})$
/// for $F = (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, \dots)$.
#[verified_engine::verified]
pub fn is_consecutive_fibonacci_pair(a: usize, b: usize) -> bool {
    if a == 0 || b == 0 {
        return false;
    }
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    let mut f_prev = 1_usize;
    let mut f_curr = 1_usize;
    while f_curr <= hi {
        if f_prev == lo && f_curr == hi {
            return true;
        }
        let next = f_prev.saturating_add(f_curr);
        if next == f_curr {
            break;
        }
        f_prev = f_curr;
        f_curr = next;
    }
    false
}

/// Wraps an angle in radians into $(-\pi, \pi]$.
#[inline]
#[verified_engine::verified]
fn wrap_signed_pi(angle: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let mut wrapped = angle % tau;
    if wrapped > std::f64::consts::PI {
        wrapped -= tau;
    } else if wrapped <= -std::f64::consts::PI {
        wrapped += tau;
    }
    wrapped
}

/// Identifies the dominant visible opposing parastichy steps $(m, n)$ at the outer rim of a
/// Vogel disc and computes nearest-neighbor packing uniformity without heap allocations.
#[verified_engine::verified]
pub fn analyze_parastichies(
    points: &[PrimordiumPoint],
    divergence_rad: f64,
    scale: f64,
) -> ParastichyAnalysis {
    let n = points.len();
    if n <= 2 {
        return ParastichyAnalysis::default();
    }

    let ref_idx = n - 1;
    let r_ref = points[ref_idx].radius;
    let spoke_eps = 1e-6_f64;

    let mut best_cw_step = 1_usize;
    let mut best_cw_dist_sq = f64::INFINITY;
    let mut best_ccw_step = 1_usize;
    let mut best_ccw_dist_sq = f64::INFINITY;
    let mut spoke_step: Option<usize> = None;

    let max_step = ref_idx.min(256);
    for step in 1..=max_step {
        let delta = wrap_signed_pi((step as f64) * divergence_rad);
        let r_prev = points[ref_idx - step].radius;
        let dist_sq = r_ref * r_ref + r_prev * r_prev - 2.0 * r_ref * r_prev * delta.cos();

        if delta.abs() <= spoke_eps && spoke_step.is_none() {
            spoke_step = Some(step);
        }
        if delta < -spoke_eps && dist_sq < best_cw_dist_sq {
            best_cw_dist_sq = dist_sq;
            best_cw_step = step;
        } else if delta > spoke_eps && dist_sq < best_ccw_dist_sq {
            best_ccw_dist_sq = dist_sq;
            best_ccw_step = step;
        }
    }

    if let Some(q) = spoke_step {
        let packing = compute_packing_uniformity(points, scale);
        return ParastichyAnalysis {
            primary_count: q,
            secondary_count: q,
            clockwise_step: q,
            counter_clockwise_step: q,
            is_fibonacci_pair: false,
            has_radial_spokes: true,
            packing_uniformity: packing,
        };
    }

    let (primary, secondary) = if best_cw_step <= best_ccw_step {
        (best_cw_step, best_ccw_step)
    } else {
        (best_ccw_step, best_cw_step)
    };
    let packing = compute_packing_uniformity(points, scale);

    ParastichyAnalysis {
        primary_count: primary,
        secondary_count: secondary,
        clockwise_step: best_cw_step,
        counter_clockwise_step: best_ccw_step,
        is_fibonacci_pair: is_consecutive_fibonacci_pair(primary, secondary),
        has_radial_spokes: false,
        packing_uniformity: packing,
    }
}

/// Evaluates packing uniformity in $(0, 1]$ by comparing the mean interior nearest-neighbor
/// distance against the hexagonal upper bound $d_{\text{hex}} = c\sqrt{2\pi / \sqrt{3}} \approx 1.9046 c$.
#[verified_engine::verified]
fn compute_packing_uniformity(points: &[PrimordiumPoint], scale: f64) -> f64 {
    let n = points.len();
    if n < 6 {
        return 1.0;
    }
    let sample_start = n / 4;
    let sample_end = (3 * n / 4).max(sample_start + 1).min(n);
    let max_lookback = 144_usize;
    let mut min_dist_sum = 0.0_f64;
    let mut sample_count = 0_usize;

    for i in sample_start..sample_end {
        let pi = points[i];
        let mut best_d2 = f64::INFINITY;
        let lo = i.saturating_sub(max_lookback);
        let hi = (i + max_lookback + 1).min(n);
        for (j, pj) in points.iter().enumerate().take(hi).skip(lo) {
            if i == j {
                continue;
            }
            let dx = pi.x - pj.x;
            let dy = pi.y - pj.y;
            let d2 = dx * dx + dy * dy;
            if d2 < best_d2 {
                best_d2 = d2;
            }
        }
        if best_d2.is_finite() {
            min_dist_sum += best_d2.sqrt();
            sample_count += 1;
        }
    }

    if sample_count == 0 {
        return 1.0;
    }
    let mean_nn = min_dist_sum / (sample_count as f64);
    let ideal_hex = scale * (std::f64::consts::TAU / 3.0_f64.sqrt()).sqrt();
    (mean_nn / ideal_hex).clamp(0.0, 1.0)
}

/// Fills `out_indices` with indices `arm_index, arm_index + step, arm_index + 2*step, ... < count`
/// without reallocating when `out_indices.capacity() >= count`.
#[verified_engine::verified]
pub fn fill_parastichy_chain(
    count: usize,
    arm_index: usize,
    parastichy_step: usize,
    out_indices: &mut Vec<usize>,
) {
    out_indices.clear();
    let step = parastichy_step.max(1);
    let mut idx = arm_index;
    while idx < count {
        out_indices.push(idx);
        idx += step;
    }
}
