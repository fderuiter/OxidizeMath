//! Pure numerical divergence metrics for counterfactual shadow twin comparisons.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

/// Minimum positive separation threshold used as reference baseline for logarithmic slope.
const MIN_LOG_SLOPE_REFERENCE: f64 = 1e-30;

/// Minimum elapsed time `t - t_ref` before computing finite-time logarithmic separation slope.
const MIN_LOG_SLOPE_DT: f64 = 1e-12;

/// Incrementally tracks `||x_B(t) - x_A(t)||_2`, peak separation, and finite-time
/// logarithmic separation slope `ln(||δx(t)|| / ||δx_ref||) / (t - t_ref)` with zero allocations.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SeparationTracker {
    fork_time: f64,
    ref_sample: Option<(f64, f64)>,
    latest_time: f64,
    latest_l2: f64,
    peak_l2: f64,
    latest_log_slope: f64,
}

impl SeparationTracker {
    /// Initializes a tracker anchored at `(fork_time, initial_l2)`.
    #[must_use]
    pub fn new(fork_time: f64, initial_l2: f64) -> Self {
        let clean_t = if fork_time.is_finite() {
            fork_time
        } else {
            0.0
        };
        let clean_d = sanitize_non_negative(initial_l2);
        let ref_sample = (clean_d > MIN_LOG_SLOPE_REFERENCE).then_some((clean_t, clean_d));
        Self {
            fork_time: clean_t,
            ref_sample,
            latest_time: clean_t,
            latest_l2: clean_d,
            peak_l2: clean_d,
            latest_log_slope: 0.0,
        }
    }

    /// Records a lockstep sample `(current_time, l2_separation)` and updates peak & log slope.
    pub fn observe(&mut self, current_time: f64, l2_separation: f64) {
        if !current_time.is_finite() {
            return;
        }
        let d = sanitize_non_negative(l2_separation);
        self.latest_time = current_time;
        self.latest_l2 = d;
        if d > self.peak_l2 {
            self.peak_l2 = d;
        }
        match self.ref_sample {
            None => {
                if d > MIN_LOG_SLOPE_REFERENCE {
                    self.ref_sample = Some((current_time, d));
                }
                self.latest_log_slope = 0.0;
            }
            Some((ref_t, ref_d)) => {
                let dt = current_time - ref_t;
                if dt > MIN_LOG_SLOPE_DT && d > MIN_LOG_SLOPE_REFERENCE && ref_d > 0.0 {
                    let slope = (d / ref_d).ln() / dt;
                    self.latest_log_slope = if slope.is_finite() { slope } else { 0.0 };
                } else {
                    self.latest_log_slope = 0.0;
                }
            }
        }
    }

    /// Simulation time `t_k` at which the fork occurred.
    #[must_use]
    pub const fn fork_time(&self) -> f64 {
        self.fork_time
    }

    /// Latest observed time `t`.
    #[must_use]
    pub const fn current_time(&self) -> f64 {
        self.latest_time
    }

    /// Latest Euclidean separation `||x_B(t) - x_A(t)||_2`.
    #[must_use]
    pub const fn l2_separation(&self) -> f64 {
        self.latest_l2
    }

    /// Peak separation `max_{τ ∈ [t_k, t]} ||x_B(τ) - x_A(τ)||_2`.
    #[must_use]
    pub const fn peak_separation(&self) -> f64 {
        self.peak_l2
    }

    /// Finite-time logarithmic separation slope `ln(||δx(t)|| / ||δx_ref||) / Δt`.
    #[must_use]
    pub const fn log_separation_slope(&self) -> f64 {
        self.latest_log_slope
    }
}

#[inline]
fn sanitize_non_negative(val: f64) -> f64 {
    if val.is_finite() && val > 0.0 {
        val
    } else {
        0.0
    }
}

/// Computes exact Euclidean distance `||b - a||_2` between two 3D state vectors.
#[must_use]
pub fn l2_distance_3d(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    let sum_sq = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
    sanitize_non_negative(sum_sq.sqrt())
}

/// Computes spatial `(L2, L_inf)` difference norms across two equal-length scalar fields `u_a` and `u_b`.
///
/// `L2` is the Euclidean field norm `sqrt(sum_i (u_b[i] - u_a[i])^2)` and `L_inf` is `max_i |u_b[i] - u_a[i]|`.
#[must_use]
pub fn field_difference_norms(u_a: &[f64], u_b: &[f64]) -> (f64, f64) {
    let n = u_a.len().min(u_b.len());
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut sum_sq = 0.0_f64;
    let mut max_abs = 0.0_f64;
    for idx in 0..n {
        let a = u_a[idx];
        let b = u_b[idx];
        if a.is_finite() && b.is_finite() {
            let diff = (b - a).abs();
            sum_sq = diff.mul_add(diff, sum_sq);
            if diff > max_abs {
                max_abs = diff;
            }
        }
    }
    (
        sanitize_non_negative(sum_sq.sqrt()),
        sanitize_non_negative(max_abs),
    )
}

/// Computes peak infections averted `ΔI_peak = max_t I_A(t) - max_t I_B(t)` between baseline and intervention curves.
#[must_use]
pub fn sir_peak_infections_averted(i_branch_a: &[[f64; 2]], i_branch_b: &[[f64; 2]]) -> f64 {
    let peak_a = i_branch_a
        .iter()
        .map(|p| p[1])
        .filter(|v| v.is_finite())
        .fold(0.0_f64, f64::max);
    let peak_b = i_branch_b
        .iter()
        .map(|p| p[1])
        .filter(|v| v.is_finite())
        .fold(0.0_f64, f64::max);
    let delta = peak_a - peak_b;
    if delta.is_finite() {
        delta
    } else {
        0.0
    }
}

/// Computes transverse magnetization echo gain `ΔM_xy = ||M_xy,B||_2 - ||M_xy,A||_2`.
#[must_use]
pub fn mri_transverse_echo_gain(m_branch_a: [f64; 3], m_branch_b: [f64; 3]) -> f64 {
    let mxy_a = m_branch_a[0].hypot(m_branch_a[1]);
    let mxy_b = m_branch_b[0].hypot(m_branch_b[1]);
    let delta = mxy_b - mxy_a;
    if delta.is_finite() {
        delta
    } else {
        0.0
    }
}
