#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Phase-space view mode, `t_slice` quiver/slope-field evaluation, and 2D trace projection
//! for [`OdeSolverTool`].

#[path = "phase/canvas_seeding.rs"]
mod canvas_seeding;

use super::systems::CustomOdeSpec;
use super::OdeSolverTool;
use crate::framework::{decimate_xy_series_minmax, TraceSnapshot, MAX_DISPLAY_POLYLINE_POINTS};
use crate::presets::OdePreset;
use egui_plot::PlotPoint;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

const QUIVER_GRID_2D: usize = 15;
const SLOPE_COLS_1D: usize = 17;
const SLOPE_ROWS_1D: usize = 13;

/// Active coordinate projection mode for the [`OdeSolverTool`] plot canvas.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlotViewMode {
    /// Standard time-domain curves `y(t)` (and `v(t)` when coupled).
    #[default]
    TimeSeries,
    /// 2D `(y, v)` phase portrait for 2-state systems or 1D `(t, y)` slope field for scalar ODEs.
    PhasePortrait,
}

impl PlotViewMode {
    /// Human-readable label for UI selectors.
    #[inline]
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::TimeSeries => "Time Series",
            Self::PhasePortrait => "Phase Portrait",
        }
    }
}

/// Predicate omitting default `view_mode` during JSON serialization for backward compatibility.
#[inline]
#[must_use]
pub(crate) const fn is_default_view_mode(mode: &PlotViewMode) -> bool {
    matches!(mode, PlotViewMode::TimeSeries)
}

/// Predicate omitting zero `t_slice` during JSON serialization for backward compatibility.
#[inline]
#[must_use]
pub(crate) fn is_zero_t_slice(t_slice: &f64) -> bool {
    t_slice.is_finite() && t_slice.abs() <= f64::EPSILON
}

/// Cached normalized `egui_plot::Arrows` origin and tip buffers keyed by parameter/slice state.
#[derive(Clone, Debug, Default)]
pub(crate) struct QuiverCache {
    signature: u64,
    origins: Arc<Vec<PlotPoint>>,
    tips: Arc<Vec<PlotPoint>>,
}

fn series_bounds_with_seed(series: &[f64], seed: f64, min_half_span: f64) -> (f64, f64) {
    let mut lo = if seed.is_finite() {
        seed.clamp(-1e4, 1e4)
    } else {
        -min_half_span
    };
    let mut hi = lo;
    for &val in series {
        if val.is_finite() && val.abs() <= 1e4 {
            lo = lo.min(val);
            hi = hi.max(val);
        }
    }
    let center = 0.5 * (lo + hi);
    let half = (0.5 * (hi - lo) * 1.18).max(min_half_span).min(1e4);
    (center - half, center + half)
}

fn push_normalized_arrow(
    origins: &mut Vec<PlotPoint>,
    tips: &mut Vec<PlotPoint>,
    (x, y): (f64, f64),
    (dx, dy): (f64, f64),
    (cell_w, cell_h): (f64, f64),
) {
    if !dx.is_finite() || !dy.is_finite() || cell_w <= 0.0 || cell_h <= 0.0 {
        return;
    }
    let ux = dx / cell_w;
    let uy = dy / cell_h;
    let norm = ux.hypot(uy);
    if !norm.is_finite() || norm <= 1e-12 {
        return;
    }
    let scale = 0.38;
    let tip_x = x + scale * cell_w * (ux / norm);
    let tip_y = y + scale * cell_h * (uy / norm);
    if tip_x.is_finite() && tip_y.is_finite() {
        origins.push(PlotPoint::new(x, y));
        tips.push(PlotPoint::new(tip_x, tip_y));
    }
}

fn project_snapshot_to_phase_orbit(snapshot: &TraceSnapshot) -> Option<Arc<Vec<PlotPoint>>> {
    let sy = snapshot.series.first()?;
    let sv = snapshot.series.get(1)?;
    let len = sy.points.len().min(sv.points.len());
    if len == 0 {
        return None;
    }
    let mut ys = Vec::with_capacity(len);
    let mut vs = Vec::with_capacity(len);
    for idx in 0..len {
        let y = sy.points[idx][1];
        let v = sv.points[idx][1];
        if y.is_finite() && v.is_finite() {
            ys.push(y);
            vs.push(v);
        }
    }
    if ys.is_empty() {
        return None;
    }
    Some(decimate_xy_series_minmax(
        &ys,
        &vs,
        MAX_DISPLAY_POLYLINE_POINTS,
    ))
}

impl OdeSolverTool {
    /// Returns the active [`PlotViewMode`].
    #[cfg(test)]
    #[inline]
    #[must_use]
    pub const fn view_mode(&self) -> PlotViewMode {
        self.view_mode
    }

    /// Switches [`PlotViewMode`] (`TimeSeries` or `PhasePortrait`) and refreshes callouts.
    pub fn set_view_mode(&mut self, mode: PlotViewMode) {
        if self.view_mode != mode {
            self.view_mode = mode;
            self.bump_revision();
            self.update_callouts();
        }
    }

    /// Returns the instantaneous time slice `t_slice` used to evaluate non-autonomous quivers.
    #[cfg(test)]
    #[inline]
    #[must_use]
    pub const fn t_slice(&self) -> f64 {
        self.t_slice
    }

    /// Sets and clamps `t_slice` into `[0.0, total_time]`.
    pub fn set_t_slice(&mut self, t_slice: f64) {
        let hi = if self.total_time.is_finite() && self.total_time > 0.0 {
            self.total_time
        } else {
            10.0
        };
        let clamped = if t_slice.is_finite() {
            t_slice.clamp(0.0, hi)
        } else {
            0.0
        };
        if self.t_slice.to_bits() != clamped.to_bits() {
            self.t_slice = clamped;
            self.bump_revision();
        }
    }

    /// Returns `true` when rendering a 2-state `(y, v)` phase portrait.
    #[inline]
    #[must_use]
    pub const fn is_2d_phase_portrait(&self) -> bool {
        matches!(self.view_mode, PlotViewMode::PhasePortrait) && self.has_second_state()
    }

    pub(crate) fn phase_window_bounds(&self) -> ((f64, f64), (f64, f64)) {
        if self.has_second_state() {
            let yb = series_bounds_with_seed(&self.y_series, self.ic_y0, 2.5);
            let vb = series_bounds_with_seed(&self.v_series, self.ic_v0, 2.5);
            (yb, vb)
        } else {
            let t_hi = if self.total_time.is_finite() && self.total_time > 0.0 {
                self.total_time
            } else {
                10.0
            };
            let yb = series_bounds_with_seed(&self.y_series, self.ic_y0, 2.0);
            ((0.0, t_hi), yb)
        }
    }

    /// Evaluates the instantaneous vector-field derivative at `(x, y)` for `self.t_slice`.
    #[must_use]
    pub fn evaluate_field_vector(
        &self,
        x: f64,
        y: f64,
        custom_spec: Option<&CustomOdeSpec>,
    ) -> [f64; 2] {
        if self.has_second_state() {
            let t = self.t_slice;
            match self.preset {
                OdePreset::HarmonicOscillator => [y, -self.param_k * x],
                OdePreset::Custom => {
                    if let Some(CustomOdeSpec::Coupled {
                        dy,
                        dv,
                        params,
                        param_len,
                    }) = custom_spec
                    {
                        let p = &params[..*param_len];
                        [dy.eval_3d(t, x, y, p), dv.eval_3d(t, x, y, p)]
                    } else {
                        [0.0, 0.0]
                    }
                }
                _ => [0.0, 0.0],
            }
        } else {
            let slope = match self.preset {
                OdePreset::Exponential => self.param_k * y,
                OdePreset::LogisticGrowth => self.param_r * y * (1.0 - y / self.param_cap_k),
                OdePreset::Custom => {
                    if let Some(CustomOdeSpec::Scalar {
                        expr,
                        params,
                        param_len,
                    }) = custom_spec
                    {
                        expr.eval_2d(x, y, &params[..*param_len])
                    } else {
                        0.0
                    }
                }
                OdePreset::HarmonicOscillator => 0.0,
            };
            [1.0, slope]
        }
    }

    fn compute_quiver_signature(&self, x_bounds: (f64, f64), y_bounds: (f64, f64)) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.preset as u8).hash(&mut h);
        self.custom_coupled.hash(&mut h);
        for v in [
            self.t_slice,
            self.param_k,
            self.param_r,
            self.param_cap_k,
            x_bounds.0,
            x_bounds.1,
            y_bounds.0,
            y_bounds.1,
        ] {
            v.to_bits().hash(&mut h);
        }
        if self.preset == OdePreset::Custom {
            if self.custom_coupled {
                self.coupled_bank.raw_text(0).hash(&mut h);
                self.coupled_bank.raw_text(1).hash(&mut h);
                for &v in self.coupled_bank.param_slice() {
                    v.to_bits().hash(&mut h);
                }
            } else {
                self.math_input.raw_text.hash(&mut h);
                for &v in self.math_input.param_slice() {
                    v.to_bits().hash(&mut h);
                }
            }
        }
        h.finish().max(1)
    }

    /// Returns shared `(origins, tips)` quiver vectors for the active phase portrait or slope field.
    pub fn quiver_vectors(&mut self) -> (Arc<Vec<PlotPoint>>, Arc<Vec<PlotPoint>>) {
        let (xb, yb) = self.phase_window_bounds();
        let sig = self.compute_quiver_signature(xb, yb);
        if self.quiver_cache.signature == sig {
            return (
                Arc::clone(&self.quiver_cache.origins),
                Arc::clone(&self.quiver_cache.tips),
            );
        }
        let spec = self.build_custom_ode_spec();
        let (origins, tips) = if self.has_second_state() {
            self.build_2d_phase_quiver(xb, yb, spec.as_ref())
        } else {
            self.build_1d_slope_quiver(xb, yb, spec.as_ref())
        };
        self.quiver_cache = QuiverCache {
            signature: sig,
            origins: Arc::new(origins),
            tips: Arc::new(tips),
        };
        (
            Arc::clone(&self.quiver_cache.origins),
            Arc::clone(&self.quiver_cache.tips),
        )
    }

    fn build_2d_phase_quiver(
        &self,
        xb: (f64, f64),
        yb: (f64, f64),
        spec: Option<&CustomOdeSpec>,
    ) -> (Vec<PlotPoint>, Vec<PlotPoint>) {
        let cell_w = (xb.1 - xb.0) / (QUIVER_GRID_2D as f64);
        let cell_h = (yb.1 - yb.0) / (QUIVER_GRID_2D as f64);
        let cap = QUIVER_GRID_2D * QUIVER_GRID_2D;
        let mut origins = Vec::with_capacity(cap);
        let mut tips = Vec::with_capacity(cap);
        for row in 0..QUIVER_GRID_2D {
            let v = yb.0 + ((row as f64) + 0.5) * cell_h;
            for col in 0..QUIVER_GRID_2D {
                let y = xb.0 + ((col as f64) + 0.5) * cell_w;
                let [dy, dv] = self.evaluate_field_vector(y, v, spec);
                push_normalized_arrow(&mut origins, &mut tips, (y, v), (dy, dv), (cell_w, cell_h));
            }
        }
        (origins, tips)
    }

    fn build_1d_slope_quiver(
        &self,
        xb: (f64, f64),
        yb: (f64, f64),
        spec: Option<&CustomOdeSpec>,
    ) -> (Vec<PlotPoint>, Vec<PlotPoint>) {
        let cell_w = (xb.1 - xb.0) / (SLOPE_COLS_1D as f64);
        let cell_h = (yb.1 - yb.0) / (SLOPE_ROWS_1D as f64);
        let cap = (SLOPE_COLS_1D + 1) * SLOPE_ROWS_1D;
        let mut origins = Vec::with_capacity(cap);
        let mut tips = Vec::with_capacity(cap);
        let slice_t = self.t_slice.clamp(xb.0, xb.1);
        for row in 0..SLOPE_ROWS_1D {
            let y = yb.0 + ((row as f64) + 0.5) * cell_h;
            let [dt_s, dy_s] = self.evaluate_field_vector(slice_t, y, spec);
            push_normalized_arrow(
                &mut origins,
                &mut tips,
                (slice_t, y),
                (dt_s, dy_s),
                (cell_w, cell_h),
            );
            for col in 0..SLOPE_COLS_1D {
                let t = xb.0 + ((col as f64) + 0.5) * cell_w;
                let [dt_g, dy_g] = self.evaluate_field_vector(t, y, spec);
                push_normalized_arrow(
                    &mut origins,
                    &mut tips,
                    (t, y),
                    (dt_g, dy_g),
                    (cell_w, cell_h),
                );
            }
        }
        (origins, tips)
    }

    /// Projects active multi-solver `comparison_manager` snapshots onto decimated 2D `(y, v)` orbits.
    #[must_use]
    pub fn projected_comparison_orbits(&self) -> Vec<(String, Arc<Vec<PlotPoint>>)> {
        self.comparison_manager
            .snapshots()
            .iter()
            .filter(|s| s.visible)
            .filter_map(|s| {
                let pts = project_snapshot_to_phase_orbit(s)?;
                Some((s.display_label(), pts))
            })
            .collect()
    }

    /// Projects visible `snapshot_manager` history and pinned traces onto decimated 2D `(y, v)` orbits.
    #[must_use]
    pub fn projected_snapshot_orbits(&self) -> Vec<(String, Arc<Vec<PlotPoint>>, bool)> {
        let mut out = Vec::new();
        for run in self.snapshot_manager.history_runs() {
            if run.visible {
                if let Some(pts) = project_snapshot_to_phase_orbit(run) {
                    out.push((run.display_label(), pts, true));
                }
            }
        }
        for snap in self.snapshot_manager.snapshots() {
            if snap.visible {
                if let Some(pts) = project_snapshot_to_phase_orbit(snap) {
                    out.push((snap.display_label(), pts, false));
                }
            }
        }
        out
    }
}
