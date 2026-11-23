#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    BifurcationDiagram, CUSTOM_BURN_IN, CUSTOM_ESCAPE_GUARD, CUSTOM_ORBIT_SAMPLES, DEFAULT_X0,
    LYAPUNOV_ITERATIONS,
};
use egui_plot::PlotPoint;
use math_commons::ast::CompiledExpr;
use math_explorer::physics::chaos::logistic::{self, LogisticMap};
use math_explorer::physics::chaos::metrics::logistic_lyapunov;

/// Minimum finite display floor for plotted Lyapunov exponents (clamping superstable `-inf` dips).
pub const LYAPUNOV_PLOT_MIN: f64 = -5.0;
/// Maximum finite display ceiling for plotted Lyapunov exponents.
pub const LYAPUNOV_PLOT_MAX: f64 = 5.0;
/// Finite coordinate clamp applied to custom plot outputs.
pub const CUSTOM_AST_PLOT_CLAMP: f64 = 10.0;

const ORBIT_SERIES_STEPS: usize = 96;
const COBWEB_TRANSIENT_STEPS: usize = 24;
const COBWEB_CYCLE_STEPS: usize = 24;
const CUSTOM_FD_EPS: f64 = 1e-6;

/// Clamps raw Lyapunov exponents (such as superstable `-inf` when `f'(x) = 0`) to finite plot bounds.
#[must_use]
pub fn clamp_lyapunov_value(raw: f64) -> f64 {
    if raw.is_nan() {
        0.0
    } else {
        raw.clamp(LYAPUNOV_PLOT_MIN, LYAPUNOV_PLOT_MAX)
    }
}

fn reset_plot_buf(buf: &mut Vec<PlotPoint>, min_cap: usize) {
    buf.clear();
    if buf.capacity() < min_cap {
        buf.reserve(min_cap - buf.capacity());
    }
}

fn fill_cobweb_staircase<F>(
    out: &mut Vec<PlotPoint>,
    mut x: f64,
    steps: usize,
    include_base: bool,
    escape_limit: f64,
    mut step_fn: F,
) where
    F: FnMut(f64) -> Option<f64>,
{
    reset_plot_buf(out, steps * 2 + 1);
    let x_clamped = x.clamp(-CUSTOM_AST_PLOT_CLAMP, CUSTOM_AST_PLOT_CLAMP);
    out.push(PlotPoint::new(
        x_clamped,
        if include_base { 0.0 } else { x_clamped },
    ));
    for _ in 0..steps {
        let Some(next) = step_fn(x) else {
            break;
        };
        if !next.is_finite() || next.abs() > escape_limit {
            break;
        }
        let cx = x.clamp(-CUSTOM_AST_PLOT_CLAMP, CUSTOM_AST_PLOT_CLAMP);
        let cnext = next.clamp(-CUSTOM_AST_PLOT_CLAMP, CUSTOM_AST_PLOT_CLAMP);
        out.push(PlotPoint::new(cx, cnext));
        out.push(PlotPoint::new(cnext, cnext));
        x = next;
    }
}

fn custom_lyapunov_at(
    compiled: &CompiledExpr,
    params: &[f64],
    r: f64,
    x0: f64,
    n: usize,
) -> Option<f64> {
    let mut x = x0;
    let mut sum_logs = 0.0;
    let steps = n.max(1);
    for _ in 0..steps {
        let f_plus = compiled.eval_2d(r, x + CUSTOM_FD_EPS, params);
        let f_minus = compiled.eval_2d(r, x - CUSTOM_FD_EPS, params);
        let next_x = compiled.eval_2d(r, x, params);
        if !next_x.is_finite()
            || next_x.abs() > CUSTOM_ESCAPE_GUARD
            || !f_plus.is_finite()
            || !f_minus.is_finite()
        {
            return None;
        }
        let deriv = ((f_plus - f_minus) / (2.0 * CUSTOM_FD_EPS)).abs();
        sum_logs += if deriv == 0.0 {
            LYAPUNOV_PLOT_MIN
        } else {
            deriv.ln()
        };
        x = next_x;
    }
    Some(clamp_lyapunov_value(sum_logs / steps as f64))
}

fn fill_custom_sweep(
    compiled: &CompiledExpr,
    params: &[f64],
    (r_min, r_max): (f64, f64),
    steps: usize,
    x0: f64,
    points: &mut Vec<PlotPoint>,
    lyapunov_points: &mut Vec<PlotPoint>,
) {
    let clamped_steps = steps.clamp(2, 2000);
    let span = r_max - r_min;
    if !span.is_finite() || span <= 0.0 {
        points.clear();
        lyapunov_points.clear();
        return;
    }
    reset_plot_buf(points, clamped_steps.saturating_mul(CUSTOM_ORBIT_SAMPLES));
    let lyap_samples = clamped_steps.min(250);
    reset_plot_buf(lyapunov_points, lyap_samples + 1);

    let denom = (clamped_steps - 1) as f64;
    for step_idx in 0..clamped_steps {
        let r = r_min + (step_idx as f64 / denom) * span;
        let mut x = x0;
        let mut bounded = x.is_finite() && x.abs() <= CUSTOM_ESCAPE_GUARD;
        for _ in 0..CUSTOM_BURN_IN {
            if !bounded {
                break;
            }
            x = compiled.eval_2d(r, x, params);
            bounded = x.is_finite() && x.abs() <= CUSTOM_ESCAPE_GUARD;
        }
        if !bounded {
            continue;
        }
        for _ in 0..CUSTOM_ORBIT_SAMPLES {
            x = compiled.eval_2d(r, x, params);
            if !x.is_finite() || x.abs() > CUSTOM_ESCAPE_GUARD {
                break;
            }
            points.push(PlotPoint::new(r, x));
        }
    }

    let dr = span / lyap_samples as f64;
    for idx in 0..=lyap_samples {
        let r = r_min + dr * idx as f64;
        if let Some(lyap) = custom_lyapunov_at(compiled, params, r, x0, 120) {
            lyapunov_points.push(PlotPoint::new(r, lyap));
        }
    }
}

struct CoupledBuffers<'a> {
    orbit_points: &'a mut Vec<PlotPoint>,
    cobweb_curve: &'a mut Vec<PlotPoint>,
    cobweb_diagonal: &'a mut Vec<PlotPoint>,
    cobweb_transient: &'a mut Vec<PlotPoint>,
    cobweb_cycle: &'a mut Vec<PlotPoint>,
}

fn fill_custom_cobweb(
    compiled: &CompiledExpr,
    params: &[f64],
    selected_r: f64,
    x0: f64,
    (x_lo, x_hi): (f64, f64),
    bufs: CoupledBuffers<'_>,
) {
    reset_plot_buf(bufs.cobweb_curve, 129);
    let span_x = (x_hi - x_lo).max(1.0);
    for i in 0..=128 {
        let xv = x_lo + (i as f64 / 128.0) * span_x;
        let yv = compiled.eval_2d(selected_r, xv, params);
        if yv.is_finite() {
            bufs.cobweb_curve.push(PlotPoint::new(
                xv,
                yv.clamp(-CUSTOM_AST_PLOT_CLAMP, CUSTOM_AST_PLOT_CLAMP),
            ));
        }
    }
    reset_plot_buf(bufs.cobweb_diagonal, 2);
    bufs.cobweb_diagonal.push(PlotPoint::new(x_lo, x_lo));
    bufs.cobweb_diagonal.push(PlotPoint::new(x_hi, x_hi));

    fill_cobweb_staircase(
        bufs.cobweb_transient,
        x0,
        COBWEB_TRANSIENT_STEPS,
        true,
        CUSTOM_ESCAPE_GUARD,
        |xv| Some(compiled.eval_2d(selected_r, xv, params)),
    );
    let mut settled = x0;
    for _ in 0..CUSTOM_BURN_IN {
        let nx = compiled.eval_2d(selected_r, settled, params);
        if !nx.is_finite() || nx.abs() > CUSTOM_ESCAPE_GUARD {
            break;
        }
        settled = nx;
    }
    fill_cobweb_staircase(
        bufs.cobweb_cycle,
        settled,
        COBWEB_CYCLE_STEPS,
        false,
        CUSTOM_ESCAPE_GUARD,
        |xv| Some(compiled.eval_2d(selected_r, xv, params)),
    );
}

fn fill_custom_coupled(
    compiled: &CompiledExpr,
    params: &[f64],
    selected_r: f64,
    x0: f64,
    selected_lyapunov: &mut f64,
    bufs: CoupledBuffers<'_>,
) -> bool {
    reset_plot_buf(bufs.orbit_points, ORBIT_SERIES_STEPS + 1);
    let mut x = x0;
    let mut x_lo = 0.0_f64.min(x0.clamp(-CUSTOM_AST_PLOT_CLAMP, CUSTOM_AST_PLOT_CLAMP));
    let mut x_hi = 1.0_f64.max(x0.clamp(-CUSTOM_AST_PLOT_CLAMP, CUSTOM_AST_PLOT_CLAMP));
    bufs.orbit_points.push(PlotPoint::new(
        0.0,
        x.clamp(-CUSTOM_AST_PLOT_CLAMP, CUSTOM_AST_PLOT_CLAMP),
    ));
    let mut escaped = !x.is_finite() || x.abs() > CUSTOM_ESCAPE_GUARD;
    for step in 1..=ORBIT_SERIES_STEPS {
        if escaped {
            break;
        }
        let next = compiled.eval_2d(selected_r, x, params);
        if next.is_finite() && next.abs() <= CUSTOM_ESCAPE_GUARD {
            let cy = next.clamp(-CUSTOM_AST_PLOT_CLAMP, CUSTOM_AST_PLOT_CLAMP);
            x_lo = x_lo.min(cy);
            x_hi = x_hi.max(cy);
            bufs.orbit_points.push(PlotPoint::new(step as f64, cy));
            x = next;
        } else {
            escaped = true;
        }
    }
    *selected_lyapunov = custom_lyapunov_at(compiled, params, selected_r, x0, 200).unwrap_or(0.0);
    fill_custom_cobweb(compiled, params, selected_r, x0, (x_lo, x_hi), bufs);
    !escaped
}

impl BifurcationDiagram {
    pub(crate) fn recompute(&mut self) {
        if !self.r_pinned
            && self.r_min <= self.r_max
            && !(self.r_min..=self.r_max).contains(&self.selected_r)
        {
            self.selected_r = self.selected_r.clamp(self.r_min, self.r_max);
        }
        if self.map_mode.is_custom() {
            self.recompute_custom_compiled();
        } else {
            self.custom_diverged = false;
            self.custom_eval_error = None;
            self.recompute_default_sweep();
            self.recompute_coupled_views();
        }
    }

    pub(crate) fn recompute_coupled_views(&mut self) {
        if self.map_mode.is_custom() {
            let Some(compiled) = self.math_input.compiled() else {
                return;
            };
            let params = self.math_input.param_slice();
            let _ = fill_custom_coupled(
                compiled,
                params,
                self.selected_r,
                self.x0,
                &mut self.selected_lyapunov,
                CoupledBuffers {
                    orbit_points: &mut self.orbit_points,
                    cobweb_curve: &mut self.cobweb_curve,
                    cobweb_diagonal: &mut self.cobweb_diagonal,
                    cobweb_transient: &mut self.cobweb_transient,
                    cobweb_cycle: &mut self.cobweb_cycle,
                },
            );
        } else {
            self.recompute_default_coupled();
        }
    }

    fn recompute_default_coupled(&mut self) {
        let safe_x0 = if self.x0.is_finite() {
            self.x0
        } else {
            DEFAULT_X0
        };
        self.selected_lyapunov = clamp_lyapunov_value(logistic_lyapunov(
            self.selected_r,
            safe_x0,
            LYAPUNOV_ITERATIONS,
        ));
        self.ensure_selected_r_on_default_lyapunov(safe_x0);

        reset_plot_buf(&mut self.cobweb_curve, 129);
        for i in 0..=128 {
            let x = i as f64 / 128.0;
            self.cobweb_curve
                .push(PlotPoint::new(x, self.selected_r * x * (1.0 - x)));
        }
        reset_plot_buf(&mut self.cobweb_diagonal, 2);
        self.cobweb_diagonal.push(PlotPoint::new(0.0, 0.0));
        self.cobweb_diagonal.push(PlotPoint::new(1.0, 1.0));

        let mut trans_map = LogisticMap::new(self.selected_r, safe_x0);
        fill_cobweb_staircase(
            &mut self.cobweb_transient,
            safe_x0,
            COBWEB_TRANSIENT_STEPS,
            true,
            CUSTOM_ESCAPE_GUARD,
            |_| trans_map.next(),
        );

        let mut settled_map = LogisticMap::new(self.selected_r, safe_x0);
        for _ in 0..logistic::TRANSIENT_STEPS {
            let _ = settled_map.next();
        }
        let settled_x = settled_map.state;
        fill_cobweb_staircase(
            &mut self.cobweb_cycle,
            settled_x,
            COBWEB_CYCLE_STEPS,
            false,
            CUSTOM_ESCAPE_GUARD,
            |_| settled_map.next(),
        );

        reset_plot_buf(&mut self.orbit_points, ORBIT_SERIES_STEPS + 1);
        let mut orbit_map = LogisticMap::new(self.selected_r, safe_x0);
        self.orbit_points.push(PlotPoint::new(0.0, safe_x0));
        for step in 1..=ORBIT_SERIES_STEPS {
            if let Some(xn) = orbit_map.next() {
                self.orbit_points.push(PlotPoint::new(
                    step as f64,
                    xn.clamp(-CUSTOM_AST_PLOT_CLAMP, CUSTOM_AST_PLOT_CLAMP),
                ));
            }
        }
    }

    fn recompute_default_sweep(&mut self) {
        let raw_points = logistic::generate_bifurcation_diagram(self.r_min, self.r_max, self.steps);
        self.points = raw_points
            .into_iter()
            .map(|(r, x)| PlotPoint::new(r, x))
            .collect();

        let safe_x0 = if self.x0.is_finite() {
            self.x0
        } else {
            DEFAULT_X0
        };
        let lyap_samples = self.steps.clamp(100, 250);
        let dr = (self.r_max - self.r_min) / lyap_samples as f64;
        self.lyapunov_points = (0..=lyap_samples)
            .map(|i| {
                let r = self.r_min + dr * i as f64;
                let lyap = clamp_lyapunov_value(logistic_lyapunov(r, safe_x0, LYAPUNOV_ITERATIONS));
                PlotPoint::new(r, lyap)
            })
            .collect();
    }

    fn ensure_selected_r_on_default_lyapunov(&mut self, safe_x0: f64) {
        if !(self.r_min..=self.r_max).contains(&self.selected_r) {
            return;
        }
        let exact = clamp_lyapunov_value(logistic_lyapunov(
            self.selected_r,
            safe_x0,
            LYAPUNOV_ITERATIONS,
        ));
        match self.lyapunov_points.binary_search_by(|p| {
            p.x.partial_cmp(&self.selected_r)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(idx) => self.lyapunov_points[idx].y = exact,
            Err(idx) => self
                .lyapunov_points
                .insert(idx, PlotPoint::new(self.selected_r, exact)),
        }
    }

    fn recompute_custom_compiled(&mut self) {
        let Some(compiled) = self.math_input.compiled() else {
            self.custom_diverged = false;
            self.custom_eval_error = None;
            return;
        };
        let params = self.math_input.param_slice();
        fill_custom_sweep(
            compiled,
            params,
            (self.r_min, self.r_max),
            self.steps,
            self.x0,
            &mut self.points,
            &mut self.lyapunov_points,
        );
        let coupled_bounded = fill_custom_coupled(
            compiled,
            params,
            self.selected_r,
            self.x0,
            &mut self.selected_lyapunov,
            CoupledBuffers {
                orbit_points: &mut self.orbit_points,
                cobweb_curve: &mut self.cobweb_curve,
                cobweb_diagonal: &mut self.cobweb_diagonal,
                cobweb_transient: &mut self.cobweb_transient,
                cobweb_cycle: &mut self.cobweb_cycle,
            },
        );
        if !coupled_bounded && self.points.is_empty() {
            self.custom_diverged = true;
            self.custom_eval_error =
                Some("Custom map diverged outside finite bounds [-1e6, 1e6].".to_string());
        } else {
            self.custom_diverged = false;
            self.custom_eval_error = None;
        }
    }

    /// Returns settled post-transient attractor samples for period analysis and verification.
    #[cfg(test)]
    #[must_use]
    pub fn sample_settled_orbit(&self, count: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(count);
        if self.map_mode.is_custom() {
            let Some(compiled) = self.math_input.active_compiled() else {
                return out;
            };
            let params = self.math_input.param_slice();
            let mut x = self.x0;
            for _ in 0..logistic::TRANSIENT_STEPS {
                let nx = compiled.eval_2d(self.selected_r, x, params);
                if !nx.is_finite() || nx.abs() > CUSTOM_ESCAPE_GUARD {
                    return out;
                }
                x = nx;
            }
            for _ in 0..count {
                let nx = compiled.eval_2d(self.selected_r, x, params);
                if nx.is_finite() {
                    x = nx;
                    out.push(x);
                }
            }
        } else {
            let mut map = LogisticMap::new(self.selected_r, self.x0);
            for _ in 0..logistic::TRANSIENT_STEPS {
                let _ = map.next();
            }
            for _ in 0..count {
                if let Some(nx) = map.next() {
                    out.push(nx);
                }
            }
        }
        out
    }
}
