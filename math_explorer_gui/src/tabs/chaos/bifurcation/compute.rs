#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{BifurcationDiagram, DEFAULT_X0, LYAPUNOV_ITERATIONS};
use egui_plot::PlotPoint;
use math_commons::ast::{EvalError, Expr};
use math_explorer::physics::chaos::logistic::{self, LogisticMap};
use math_explorer::physics::chaos::metrics::logistic_lyapunov;
use std::collections::HashMap;

/// Minimum finite display floor for plotted Lyapunov exponents (clamping superstable `-inf` dips).
pub const LYAPUNOV_PLOT_MIN: f64 = -5.0;
/// Maximum finite display ceiling for plotted Lyapunov exponents.
pub const LYAPUNOV_PLOT_MAX: f64 = 5.0;
/// Divergence magnitude threshold for custom `math_commons::ast` 1D recurrences.
pub const CUSTOM_AST_ESCAPE_LIMIT: f64 = 1e4;
/// Finite coordinate clamp applied to custom AST plot outputs.
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

fn eval_ast_step(
    expr: &Expr,
    vars: &mut HashMap<String, f64>,
    r: f64,
    x: f64,
) -> Result<f64, EvalError> {
    if let Some(slot) = vars.get_mut("r") {
        *slot = r;
    }
    if let Some(slot) = vars.get_mut("x") {
        *slot = x;
    }
    expr.eval(vars)
}

fn make_ast_vars(r: f64, x: f64) -> HashMap<String, f64> {
    let mut vars = HashMap::with_capacity(2);
    vars.insert("r".to_string(), r);
    vars.insert("x".to_string(), x);
    vars
}

fn custom_lyapunov_at(expr: &Expr, vars: &mut HashMap<String, f64>, r: f64, x0: f64, n: usize) -> Option<f64> {
    let mut x = x0;
    let mut sum_logs = 0.0;
    for _ in 0..n.max(1) {
        let f_plus = eval_ast_step(expr, vars, r, x + CUSTOM_FD_EPS).ok()?;
        let f_minus = eval_ast_step(expr, vars, r, x - CUSTOM_FD_EPS).ok()?;
        let next_x = eval_ast_step(expr, vars, r, x).ok()?;
        if !next_x.is_finite()
            || next_x.abs() > CUSTOM_AST_ESCAPE_LIMIT
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
    Some(clamp_lyapunov_value(sum_logs / n.max(1) as f64))
}

fn build_cobweb_staircase<F>(mut x: f64, steps: usize, include_base: bool, mut step_fn: F) -> Vec<PlotPoint>
where
    F: FnMut(f64) -> Option<f64>,
{
    let mut pts = Vec::with_capacity(steps * 2 + 1);
    let x_clamped = x.clamp(-CUSTOM_AST_PLOT_CLAMP, CUSTOM_AST_PLOT_CLAMP);
    pts.push(PlotPoint::new(
        x_clamped,
        if include_base { 0.0 } else { x_clamped },
    ));
    for _ in 0..steps {
        let Some(next) = step_fn(x) else {
            break;
        };
        if !next.is_finite() || next.abs() > CUSTOM_AST_ESCAPE_LIMIT {
            break;
        }
        let cx = x.clamp(-CUSTOM_AST_PLOT_CLAMP, CUSTOM_AST_PLOT_CLAMP);
        let cnext = next.clamp(-CUSTOM_AST_PLOT_CLAMP, CUSTOM_AST_PLOT_CLAMP);
        pts.push(PlotPoint::new(cx, cnext));
        pts.push(PlotPoint::new(cnext, cnext));
        x = next;
    }
    pts
}

impl BifurcationDiagram {
    pub(crate) fn recompute(&mut self) {
        if self.use_custom_map {
            self.recompute_custom_ast();
        } else {
            self.custom_diverged = false;
            self.custom_eval_error = None;
            self.recompute_default_sweep();
            self.recompute_coupled_views();
        }
    }

    pub(crate) fn recompute_coupled_views(&mut self) {
        if self.use_custom_map {
            if let Some(expr) = self.math_input.expr().cloned() {
                let mut vars = make_ast_vars(self.selected_r, self.x0);
                let _ = self.recompute_custom_coupled(&expr, &mut vars);
            }
            return;
        }
        let safe_x0 = if self.x0.is_finite() { self.x0 } else { DEFAULT_X0 };
        self.selected_lyapunov = clamp_lyapunov_value(logistic_lyapunov(
            self.selected_r,
            safe_x0,
            LYAPUNOV_ITERATIONS,
        ));
        self.ensure_selected_r_on_default_lyapunov(safe_x0);

        self.cobweb_curve = (0..=128)
            .map(|i| {
                let x = i as f64 / 128.0;
                PlotPoint::new(x, self.selected_r * x * (1.0 - x))
            })
            .collect();
        self.cobweb_diagonal = vec![PlotPoint::new(0.0, 0.0), PlotPoint::new(1.0, 1.0)];

        let mut trans_map = LogisticMap::new(self.selected_r, safe_x0);
        self.cobweb_transient =
            build_cobweb_staircase(safe_x0, COBWEB_TRANSIENT_STEPS, true, |_| trans_map.next());

        let mut settled_map = LogisticMap::new(self.selected_r, safe_x0);
        for _ in 0..logistic::TRANSIENT_STEPS {
            let _ = settled_map.next();
        }
        let settled_x = settled_map.state;
        self.cobweb_cycle =
            build_cobweb_staircase(settled_x, COBWEB_CYCLE_STEPS, false, |_| settled_map.next());

        let mut orbit_map = LogisticMap::new(self.selected_r, safe_x0);
        let mut orbit = Vec::with_capacity(ORBIT_SERIES_STEPS + 1);
        orbit.push(PlotPoint::new(0.0, safe_x0));
        for step in 1..=ORBIT_SERIES_STEPS {
            if let Some(xn) = orbit_map.next() {
                orbit.push(PlotPoint::new(
                    step as f64,
                    xn.clamp(-CUSTOM_AST_PLOT_CLAMP, CUSTOM_AST_PLOT_CLAMP),
                ));
            }
        }
        self.orbit_points = orbit;
    }

    fn recompute_default_sweep(&mut self) {
        let raw_points =
            logistic::generate_bifurcation_diagram(self.r_min, self.r_max, self.steps);
        self.points = raw_points
            .into_iter()
            .map(|(r, x)| PlotPoint::new(r, x))
            .collect();

        let safe_x0 = if self.x0.is_finite() { self.x0 } else { DEFAULT_X0 };
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

    fn recompute_custom_ast(&mut self) {
        let Some(expr) = self.math_input.expr().cloned() else {
            self.custom_diverged = false;
            self.custom_eval_error = None;
            return;
        };
        let mut vars = make_ast_vars(self.selected_r, self.x0);
        if let Err(err) = eval_ast_step(&expr, &mut vars, self.selected_r, self.x0) {
            self.custom_diverged = false;
            self.custom_eval_error = Some(err.to_string());
            return;
        }
        if !self.recompute_custom_coupled(&expr, &mut vars) {
            return;
        }
        self.recompute_custom_sweep(&expr, &mut vars);
    }

    fn recompute_custom_coupled(
        &mut self,
        expr: &Expr,
        vars: &mut HashMap<String, f64>,
    ) -> bool {
        let mut x = self.x0;
        let mut orbit = Vec::with_capacity(ORBIT_SERIES_STEPS + 1);
        orbit.push(PlotPoint::new(
            0.0,
            x.clamp(-CUSTOM_AST_PLOT_CLAMP, CUSTOM_AST_PLOT_CLAMP),
        ));
        for step in 1..=ORBIT_SERIES_STEPS {
            match eval_ast_step(expr, vars, self.selected_r, x) {
                Ok(next) if next.is_finite() && next.abs() <= CUSTOM_AST_ESCAPE_LIMIT => {
                    orbit.push(PlotPoint::new(
                        step as f64,
                        next.clamp(-CUSTOM_AST_PLOT_CLAMP, CUSTOM_AST_PLOT_CLAMP),
                    ));
                    x = next;
                }
                Ok(_) => {
                    self.custom_diverged = true;
                    self.custom_eval_error = Some(
                        "Custom map diverged outside finite bounds [-1e4, 1e4].".to_string(),
                    );
                    return false;
                }
                Err(err) => {
                    self.custom_diverged = false;
                    self.custom_eval_error = Some(err.to_string());
                    return false;
                }
            }
        }
        self.custom_diverged = false;
        self.custom_eval_error = None;
        self.orbit_points = orbit;
        self.selected_lyapunov =
            custom_lyapunov_at(expr, vars, self.selected_r, self.x0, 300).unwrap_or(0.0);
        self.build_custom_cobweb(expr, vars);
        true
    }

    fn build_custom_cobweb(&mut self, expr: &Expr, vars: &mut HashMap<String, f64>) {
        self.cobweb_curve = (0..=128)
            .filter_map(|i| {
                let x = i as f64 / 128.0;
                let y = eval_ast_step(expr, vars, self.selected_r, x).ok()?;
                y.is_finite().then(|| {
                    PlotPoint::new(x, y.clamp(-CUSTOM_AST_PLOT_CLAMP, CUSTOM_AST_PLOT_CLAMP))
                })
            })
            .collect();
        self.cobweb_diagonal = vec![PlotPoint::new(0.0, 0.0), PlotPoint::new(1.0, 1.0)];
        let r = self.selected_r;
        self.cobweb_transient =
            build_cobweb_staircase(self.x0, COBWEB_TRANSIENT_STEPS, true, |xv| {
                eval_ast_step(expr, vars, r, xv).ok()
            });
        let mut x = self.x0;
        for _ in 0..logistic::TRANSIENT_STEPS {
            let Some(next) = eval_ast_step(expr, vars, r, x).ok() else {
                break;
            };
            if !next.is_finite() || next.abs() > CUSTOM_AST_ESCAPE_LIMIT {
                break;
            }
            x = next;
        }
        self.cobweb_cycle = build_cobweb_staircase(x, COBWEB_CYCLE_STEPS, false, |xv| {
            eval_ast_step(expr, vars, r, xv).ok()
        });
    }

    fn recompute_custom_sweep(&mut self, expr: &Expr, vars: &mut HashMap<String, f64>) {
        let sweep_steps = self.steps.clamp(100, 300);
        let dr = (self.r_max - self.r_min) / sweep_steps as f64;
        let mut scatter = Vec::with_capacity(sweep_steps * 32);
        let mut lyap_pts = Vec::with_capacity(sweep_steps + 1);

        for i in 0..=sweep_steps {
            let r = self.r_min + dr * i as f64;
            let mut x = self.x0;
            let mut bounded = true;
            for _ in 0..64 {
                match eval_ast_step(expr, vars, r, x) {
                    Ok(nx) if nx.is_finite() && nx.abs() <= CUSTOM_AST_ESCAPE_LIMIT => x = nx,
                    _ => {
                        bounded = false;
                        break;
                    }
                }
            }
            if !bounded {
                continue;
            }
            for _ in 0..32 {
                if let Ok(nx) = eval_ast_step(expr, vars, r, x) {
                    if nx.is_finite() && nx.abs() <= CUSTOM_AST_ESCAPE_LIMIT {
                        x = nx;
                        scatter.push(PlotPoint::new(
                            r,
                            x.clamp(-CUSTOM_AST_PLOT_CLAMP, CUSTOM_AST_PLOT_CLAMP),
                        ));
                    }
                }
            }
            if let Some(lyap) = custom_lyapunov_at(expr, vars, r, self.x0, 200) {
                lyap_pts.push(PlotPoint::new(r, lyap));
            }
        }
        self.points = scatter;
        self.lyapunov_points = lyap_pts;
    }

    /// Returns settled post-transient attractor samples for period analysis and verification.
    #[cfg(test)]
    #[must_use]
    pub fn sample_settled_orbit(&self, count: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(count);
        if self.use_custom_map {
            let Some(expr) = self.math_input.expr() else {
                return out;
            };
            let mut vars = make_ast_vars(self.selected_r, self.x0);
            let mut x = self.x0;
            for _ in 0..logistic::TRANSIENT_STEPS {
                let Ok(nx) = eval_ast_step(expr, &mut vars, self.selected_r, x) else {
                    return out;
                };
                if !nx.is_finite() || nx.abs() > CUSTOM_AST_ESCAPE_LIMIT {
                    return out;
                }
                x = nx;
            }
            for _ in 0..count {
                if let Ok(nx) = eval_ast_step(expr, &mut vars, self.selected_r, x) {
                    if nx.is_finite() {
                        x = nx;
                        out.push(x);
                    }
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
