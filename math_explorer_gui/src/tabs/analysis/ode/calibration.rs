#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Nonlinear least-squares RK4 trajectory calibration for [`super::OdeSolverTool`]
//! presets (`Exponential`, `HarmonicOscillator`, `LogisticGrowth`) and custom [`DynamicAstOde`].

use super::OdeSolverTool;
use super::systems::DynamicAstOde;
use crate::framework::empirical_csv::{CalibrationReport, EmpiricalPanelOptions};
use crate::framework::recovery::FiniteStateGuard;
use crate::presets::OdePreset;
use crate::web_io::PlatformInstant;
use eframe::egui;
use egui_plot::{MarkerShape, PlotUi, Points};
use math_commons::ast::Expr;
use math_explorer::pure_math::analysis::ode::{ArrayState, OdeSystem, RungeKutta4, Solver, VecState};
use math_explorer::pure_math::analysis::optimization::{
    CalibrationSummary, DEFAULT_SHOOTING_MAX_DT, LevenbergMarquardtSolver,
    LevenbergMarquardtWorkspace, MAX_SUBSTEPS_PER_INTERVAL, ParameterBounds, Rk4ShootingObjective,
};
use std::collections::HashMap;

struct ScalarExpOde {
    k: f64,
}

impl OdeSystem<ArrayState<1>> for ScalarExpOde {
    fn derivative(&self, _t: f64, state: &ArrayState<1>) -> ArrayState<1> {
        ArrayState([self.k * state.0[0]])
    }
}

struct ScalarLogisticOde {
    r: f64,
    cap_k: f64,
}

impl OdeSystem<ArrayState<1>> for ScalarLogisticOde {
    fn derivative(&self, _t: f64, state: &ArrayState<1>) -> ArrayState<1> {
        let y = state.0[0];
        ArrayState([self.r * y * (1.0 - y / self.cap_k)])
    }
}

struct OscArrayOde {
    k: f64,
}

impl OdeSystem<ArrayState<2>> for OscArrayOde {
    fn derivative(&self, _t: f64, state: &ArrayState<2>) -> ArrayState<2> {
        ArrayState([state.0[1], -self.k * state.0[0]])
    }
}

impl OdeSolverTool {
    /// Renders the empirical CSV calibration panel inside the ODE sidebar and triggers fitting.
    pub(super) fn render_empirical_calibration_panel(&mut self, ui: &mut egui::Ui) -> bool {
        let opts = EmpiricalPanelOptions {
            id_salt: "ode_solver_csv",
            x_label: "Time Column t (s)",
            y_label: "Observed State Column y(t)",
            aux_label: None,
            residual_filename: "ode_calibration_residuals.csv",
            fit_button_label: "Fit Parameters",
        };
        if self.empirical_csv.show_panel(ui, &opts) {
            let ctx = ui.ctx().clone();
            return self.calibrate_from_loaded_csv(Some(&ctx)).is_ok();
        }
        false
    }

    /// Renders empirical CSV scatter markers over the active ODE trajectory canvas.
    pub(super) fn render_empirical_overlay(&self, plot_ui: &mut PlotUi<'_>, color: egui::Color32) {
        let Some(ref table) = self.empirical_csv.table else {
            return;
        };
        let pts = table.scatter_points(self.empirical_csv.x_col, self.empirical_csv.y_col);
        if pts.is_empty() {
            return;
        }
        plot_ui.points(
            Points::new(format!("Observed ({})", table.name), pts)
                .radius(4.0_f32)
                .shape(MarkerShape::Circle)
                .color(color),
        );
    }

    /// Runs damped Levenberg-Marquardt + RK4 single-shooting calibration against the active CSV table.
    pub fn calibrate_from_loaded_csv(&mut self, ctx: Option<&egui::Context>) -> Result<(), String> {
        let Some(ref table) = self.empirical_csv.table else {
            return Err("Load an empirical CSV dataset before fitting parameters".to_string());
        };
        let (times, observed) =
            table.extract_xy_sorted(self.empirical_csv.x_col, self.empirical_csv.y_col)?;
        if times[0] < 0.0 {
            let err = "ODE observation timestamps must be non-negative (t >= 0)".to_string();
            self.empirical_csv.record_error(ctx, err.clone());
            return Err(err);
        }
        let start = PlatformInstant::now();
        let outcome = match self.preset {
            OdePreset::Exponential => self.fit_exponential(&times, &observed),
            OdePreset::HarmonicOscillator => self.fit_harmonic(&times, &observed),
            OdePreset::LogisticGrowth => self.fit_logistic(&times, &observed),
            OdePreset::Custom => self.fit_custom_ast(&times, &observed),
        };
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        match outcome {
            Ok((names, summary)) => self.commit_ode_fit(ctx, &names, &summary, &times, &observed, elapsed_ms),
            Err(err) => {
                self.empirical_csv.record_error(ctx, err.clone());
                Err(err)
            }
        }
    }

    fn commit_ode_fit(
        &mut self,
        ctx: Option<&egui::Context>,
        names: &[&str],
        summary: &CalibrationSummary,
        times: &[f64],
        observed: &[f64],
        elapsed_ms: f64,
    ) -> Result<(), String> {
        let snapshot = crate::framework::InteractiveTool::save_recovery_state(self);
        self.apply_fitted_preset_params(names, &summary.parameters, times);
        self.recalculate();
        if FiniteStateGuard::diverged(self) {
            if let Some(prev) = snapshot {
                let _ = self.deserialize_state(&prev, true);
            }
            let err = "Rejected non-finite calibrated ODE state via FiniteStateGuard".to_string();
            self.empirical_csv.record_error(ctx, err.clone());
            return Err(err);
        }
        let report = CalibrationReport::from_summary(names, summary, times, observed, elapsed_ms)?;
        let announce = format!(
            "Calibrated {} in {} iterations (RMSE {:.4}, R² {:.4})",
            self.preset.name(),
            report.iterations,
            report.rmse,
            report.r_squared
        );
        self.empirical_csv.record_fit_success(ctx, report, announce);
        Ok(())
    }

    fn apply_fitted_preset_params(&mut self, names: &[&str], params: &[f64], times: &[f64]) {
        for (&name, &val) in names.iter().zip(params.iter()) {
            match name {
                "k" => self.param_k = val,
                "r" => self.param_r = val,
                "K" => self.param_cap_k = val,
                "y0" => self.ic_y0 = val,
                "v0" => self.ic_v0 = val,
                _ => {}
            }
        }
        if let Some(&t_last) = times.last() {
            if t_last > self.total_time {
                self.total_time = t_last.min(100.0);
            }
        }
    }

    fn fit_exponential(
        &self,
        times: &[f64],
        observed: &[f64],
    ) -> Result<(Vec<&'static str>, CalibrationSummary), String> {
        let max_dt = self.dt.clamp(1e-4, DEFAULT_SHOOTING_MAX_DT);
        let example = ArrayState([self.ic_y0]);
        let mut obj = Rk4ShootingObjective::new(
            ScalarExpOde { k: self.param_k },
            &example,
            times,
            observed,
            max_dt,
            move |sys, p, t0, st| {
                sys.k = p[0];
                st.0[0] = p[1];
                advance_from_zero(sys, t0, max_dt, st, |s| s.0[0].is_finite())
            },
            |st| st.0[0],
            |st| st.0[0].is_finite(),
        )
        .map_err(|e| e.to_string())?;
        let bounds = [
            ParameterBounds::new(-5.0, 5.0).unwrap(),
            ParameterBounds::new(-50.0, 50.0).unwrap(),
        ];
        let mut ws = LevenbergMarquardtWorkspace::new(times.len(), 2).map_err(|e| e.to_string())?;
        let summary = LevenbergMarquardtSolver::default()
            .fit(&mut obj, &[self.param_k, self.ic_y0], &bounds, &mut ws)
            .map_err(|e| e.to_string())?;
        Ok((vec!["k", "y0"], summary))
    }

    fn fit_harmonic(
        &self,
        times: &[f64],
        observed: &[f64],
    ) -> Result<(Vec<&'static str>, CalibrationSummary), String> {
        let max_dt = self.dt.clamp(1e-4, DEFAULT_SHOOTING_MAX_DT);
        let example = ArrayState([self.ic_y0, self.ic_v0]);
        let mut obj = Rk4ShootingObjective::new(
            OscArrayOde { k: self.param_k },
            &example,
            times,
            observed,
            max_dt,
            move |sys, p, t0, st| {
                if p[0] <= 0.0 {
                    return false;
                }
                sys.k = p[0];
                st.0 = [p[1], p[2]];
                advance_from_zero(sys, t0, max_dt, st, |s| s.0.iter().all(|v| v.is_finite()))
            },
            |st| st.0[0],
            |st| st.0.iter().all(|v| v.is_finite()),
        )
        .map_err(|e| e.to_string())?;
        let bounds = [
            ParameterBounds::new(0.05, 10.0).unwrap(),
            ParameterBounds::new(-20.0, 20.0).unwrap(),
            ParameterBounds::new(-20.0, 20.0).unwrap(),
        ];
        let mut ws = LevenbergMarquardtWorkspace::new(times.len(), 3).map_err(|e| e.to_string())?;
        let init = [self.param_k.clamp(0.05, 10.0), self.ic_y0, self.ic_v0];
        let summary = LevenbergMarquardtSolver::default()
            .fit(&mut obj, &init, &bounds, &mut ws)
            .map_err(|e| e.to_string())?;
        Ok((vec!["k", "y0", "v0"], summary))
    }

    fn fit_logistic(
        &self,
        times: &[f64],
        observed: &[f64],
    ) -> Result<(Vec<&'static str>, CalibrationSummary), String> {
        let max_dt = self.dt.clamp(1e-4, DEFAULT_SHOOTING_MAX_DT);
        let example = ArrayState([self.ic_y0]);
        let mut obj = Rk4ShootingObjective::new(
            ScalarLogisticOde {
                r: self.param_r,
                cap_k: self.param_cap_k,
            },
            &example,
            times,
            observed,
            max_dt,
            move |sys, p, t0, st| {
                if p[1] <= 1e-6 {
                    return false;
                }
                sys.r = p[0];
                sys.cap_k = p[1];
                st.0[0] = p[2];
                advance_from_zero(sys, t0, max_dt, st, |s| s.0[0].is_finite())
            },
            |st| st.0[0],
            |st| st.0[0].is_finite(),
        )
        .map_err(|e| e.to_string())?;
        let max_obs = observed.iter().copied().fold(10.0_f64, f64::max);
        let bounds = [
            ParameterBounds::new(0.01, 5.0).unwrap(),
            ParameterBounds::new(0.5, (max_obs * 5.0).max(100.0)).unwrap(),
            ParameterBounds::new(1e-3, (max_obs * 2.0).max(50.0)).unwrap(),
        ];
        let mut ws = LevenbergMarquardtWorkspace::new(times.len(), 3).map_err(|e| e.to_string())?;
        let init = [
            self.param_r.clamp(0.05, 5.0),
            self.param_cap_k.max(1.0),
            observed[0].clamp(1e-2, max_obs * 1.5),
        ];
        let summary = LevenbergMarquardtSolver::default()
            .fit(&mut obj, &init, &bounds, &mut ws)
            .map_err(|e| e.to_string())?;
        Ok((vec!["r", "K", "y0"], summary))
    }

    fn fit_custom_ast(
        &self,
        times: &[f64],
        observed: &[f64],
    ) -> Result<(Vec<&'static str>, CalibrationSummary), String> {
        let Some(expr) = self.math_input.expr().cloned() else {
            return Err("Enter a valid custom differential equation y' = f(t, y) first".into());
        };
        let (names, ast_keys, init, bounds) = self.build_custom_ast_spec(&expr);
        let mut base_map = HashMap::with_capacity(5);
        base_map.insert("k".to_string(), self.param_k);
        base_map.insert("r".to_string(), self.param_r);
        base_map.insert("cap_k".to_string(), self.param_cap_k);
        base_map.insert("K".to_string(), self.param_cap_k);
        let max_dt = self.dt.clamp(1e-4, DEFAULT_SHOOTING_MAX_DT);
        let example = VecState(vec![self.ic_y0]);
        let dyn_ode = DynamicAstOde {
            expr,
            params: base_map,
        };
        let mut obj = Rk4ShootingObjective::new(
            dyn_ode,
            &example,
            times,
            observed,
            max_dt,
            move |sys, p, t0, st| {
                for (idx, &key) in ast_keys.iter().enumerate() {
                    if key == "y0" {
                        st.0[0] = p[idx];
                    } else if key == "cap_k" {
                        if let Some(slot) = sys.params.get_mut("cap_k") {
                            *slot = p[idx];
                        }
                        if let Some(slot) = sys.params.get_mut("K") {
                            *slot = p[idx];
                        }
                    } else if let Some(slot) = sys.params.get_mut(key) {
                        *slot = p[idx];
                    }
                }
                advance_from_zero(sys, t0, max_dt, st, |s| s.0[0].is_finite())
            },
            |st| st.0[0],
            |st| st.0[0].is_finite(),
        )
        .map_err(|e| e.to_string())?;
        let mut ws =
            LevenbergMarquardtWorkspace::new(times.len(), init.len()).map_err(|e| e.to_string())?;
        let summary = LevenbergMarquardtSolver::default()
            .fit(&mut obj, &init, &bounds, &mut ws)
            .map_err(|e| e.to_string())?;
        Ok((names, summary))
    }

    fn build_custom_ast_spec(
        &self,
        expr: &Expr,
    ) -> (
        Vec<&'static str>,
        Vec<&'static str>,
        Vec<f64>,
        Vec<ParameterBounds>,
    ) {
        let mut names = Vec::new();
        let mut keys = Vec::new();
        let mut init = Vec::new();
        let mut bounds = Vec::new();
        if expr_mentions_var(expr, "k") {
            names.push("k");
            keys.push("k");
            init.push(self.param_k);
            bounds.push(ParameterBounds::new(-10.0, 10.0).unwrap());
        }
        if expr_mentions_var(expr, "r") {
            names.push("r");
            keys.push("r");
            init.push(self.param_r);
            bounds.push(ParameterBounds::new(-10.0, 10.0).unwrap());
        }
        if expr_mentions_var(expr, "cap_k") || expr_mentions_var(expr, "K") {
            names.push("K");
            keys.push("cap_k");
            init.push(self.param_cap_k.max(0.5));
            bounds.push(ParameterBounds::new(0.1, 500.0).unwrap());
        }
        names.push("y0");
        keys.push("y0");
        init.push(self.ic_y0);
        bounds.push(ParameterBounds::new(-50.0, 50.0).unwrap());
        (names, keys, init, bounds)
    }
}

fn expr_mentions_var(expr: &Expr, target: &str) -> bool {
    let mut stack = vec![expr];
    while let Some(node) = stack.pop() {
        match node {
            Expr::Number(_) => {}
            Expr::Variable(v) => {
                if v == target {
                    return true;
                }
            }
            Expr::Unary { expr: inner, .. } => stack.push(inner),
            Expr::Binary { left, right, .. } => {
                stack.push(left);
                stack.push(right);
            }
            Expr::Function { args, .. } => stack.extend(args.iter()),
        }
    }
    false
}

fn advance_from_zero<S, St, F>(
    sys: &S,
    t0: f64,
    max_dt: f64,
    state: &mut St,
    is_finite: F,
) -> bool
where
    S: OdeSystem<St>,
    St: math_explorer::pure_math::analysis::ode::VectorOperations,
    F: Fn(&St) -> bool,
{
    if t0 <= 0.0 {
        return is_finite(state);
    }
    let steps = ((t0 / max_dt).ceil() as usize).clamp(1, MAX_SUBSTEPS_PER_INTERVAL);
    let dt = t0 / (steps as f64);
    let mut rk4 = RungeKutta4::new(state);
    let mut t = 0.0;
    for _ in 0..steps {
        rk4.step(sys, t, state, dt);
        if !is_finite(state) {
            return false;
        }
        t += dt;
    }
    true
}
