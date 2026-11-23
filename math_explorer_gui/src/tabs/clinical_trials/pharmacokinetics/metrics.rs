//! Zero-allocation pharmacokinetic summary metric evaluation.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use egui_plot::PlotPoint;
use math_explorer::applied::pharmacokinetics::{half_life, t_max};

/// Scalar input parameters required to derive analytical and numerical PK metrics.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MetricContext {
    pub(crate) ka: f64,
    pub(crate) ke: f64,
    pub(crate) volume_l: f64,
    pub(crate) interval_hours: f64,
    pub(crate) trough_end_hours: f64,
    pub(crate) mec: f64,
    pub(crate) mtc: f64,
}

/// Summary pharmacokinetic exposure, elimination, accumulation, and window metrics.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PkMetrics {
    /// Peak plasma concentration `C_max` (`mg/L`).
    pub c_max: f64,
    /// Post-absorption trough plasma concentration `C_min` (`mg/L`).
    pub c_min: f64,
    /// Elimination half-life `T_1/2 = ln(2) / ke` (`h`).
    pub half_life_h: f64,
    /// Analytical single-dose time to peak `T_max` (`h`).
    pub t_max_h: f64,
    /// Total systemic clearance `CL = ke * V_d` (`L/h`).
    pub clearance_l_h: f64,
    /// Trapezoidal area under the concentration-time curve `AUC` (`mg·h/L`).
    pub auc_trapezoidal: f64,
    /// Steady-state dosing accumulation ratio `R_ac = 1 / (1 - e^(-ke * tau))`.
    pub accumulation_ratio: f64,
    /// Percentage of horizon time where `MEC <= C(t) <= MTC` (`0.0..=100.0`).
    pub therapeutic_window_pct: f64,
}

impl PkMetrics {
    /// Returns `true` when every derived metric is finite.
    #[must_use]
    pub fn is_finite(self) -> bool {
        [
            self.c_max,
            self.c_min,
            self.half_life_h,
            self.t_max_h,
            self.clearance_l_h,
            self.auc_trapezoidal,
            self.accumulation_ratio,
            self.therapeutic_window_pct,
        ]
        .into_iter()
        .all(f64::is_finite)
    }
}

/// Computes the exact fraction (`0.0..=1.0`) of linear segment `y0 -> y1` inside `[low, high]`.
#[inline]
fn segment_in_window_fraction(y0: f64, y1: f64, low: f64, high: f64) -> f64 {
    if high <= low {
        return 0.0;
    }
    let dy = y1 - y0;
    if dy.abs() <= 1.0e-15 {
        return if y0 >= low && y0 <= high { 1.0 } else { 0.0 };
    }
    let (s0, s1) = if dy > 0.0 {
        ((low - y0) / dy, (high - y0) / dy)
    } else {
        ((high - y0) / dy, (low - y0) / dy)
    };
    let a = s0.clamp(0.0, 1.0);
    let b = s1.clamp(0.0, 1.0);
    (b - a).max(0.0)
}

/// Finds the post-first-peak trough concentration across `[t_first_peak, trough_end_hours]`.
fn compute_post_peak_trough(points: &[PlotPoint], trough_end_hours: f64) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let mut first_peak_idx = 0usize;
    for idx in 1..points.len() {
        if points[idx].y >= points[idx - 1].y {
            first_peak_idx = idx;
        } else if points[first_peak_idx].y > 0.0 {
            break;
        }
    }
    if points[first_peak_idx].y <= 0.0 {
        return 0.0;
    }

    let mut trough = points[first_peak_idx].y;
    let mut found = false;
    for pt in &points[first_peak_idx..] {
        if pt.x <= trough_end_hours + 1.0e-9 {
            trough = trough.min(pt.y);
            found = true;
        }
    }
    if found {
        trough.max(0.0)
    } else {
        points.last().map_or(0.0, |p| p.y.max(0.0))
    }
}

/// Evaluates all 8 PK summary metrics over `points` without heap allocation.
#[must_use]
pub(crate) fn evaluate_pk_metrics(points: &[PlotPoint], ctx: MetricContext) -> PkMetrics {
    let half_life_h = half_life(ctx.ke);
    let t_max_h = t_max(ctx.ka, ctx.ke);
    let clearance_l_h = ctx.ke * ctx.volume_l;
    let denom = 1.0 - (-ctx.ke * ctx.interval_hours).exp();
    let accumulation_ratio = if denom > 1.0e-12 { 1.0 / denom } else { 1.0 };

    if points.len() < 2 {
        let single_c = points.first().map_or(0.0, |p| p.y.max(0.0));
        return PkMetrics {
            c_max: single_c,
            c_min: single_c,
            half_life_h,
            t_max_h,
            clearance_l_h,
            auc_trapezoidal: 0.0,
            accumulation_ratio,
            therapeutic_window_pct: 0.0,
        };
    }

    let mut c_max = 0.0_f64;
    let mut auc = 0.0_f64;
    let mut in_window_time = 0.0_f64;
    let mut total_span = 0.0_f64;

    for pair in points.windows(2) {
        let p0 = pair[0];
        let p1 = pair[1];
        c_max = c_max.max(p0.y).max(p1.y);
        let dt = (p1.x - p0.x).max(0.0);
        auc += 0.5 * (p0.y + p1.y) * dt;
        in_window_time += dt * segment_in_window_fraction(p0.y, p1.y, ctx.mec, ctx.mtc);
        total_span += dt;
    }

    let c_min = compute_post_peak_trough(points, ctx.trough_end_hours);
    let therapeutic_window_pct = if total_span > 0.0 {
        (100.0 * in_window_time / total_span).clamp(0.0, 100.0)
    } else {
        0.0
    };

    PkMetrics {
        c_max,
        c_min,
        half_life_h,
        t_max_h,
        clearance_l_h,
        auc_trapezoidal: auc,
        accumulation_ratio,
        therapeutic_window_pct,
    }
}

// [cite:clinical_trials_statistics]
