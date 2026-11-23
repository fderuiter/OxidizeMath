#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Nonlinear least-squares RK4 trajectory calibration for [`super::OdeSolverTool`]
//! presets (`Exponential`, `HarmonicOscillator`, `LogisticGrowth`) and custom [`CustomOdeSpec`].

#[path = "calibration_models.rs"]
mod models;

use super::OdeSolverTool;
use super::systems::CustomOdeSpec;
use crate::framework::empirical_csv::{CalibrationReport, EmpiricalPanelOptions};
use crate::framework::math_input::MAX_SYNTHESIZED_PARAMS;
use crate::framework::recovery::FiniteStateGuard;
use crate::presets::OdePreset;
use crate::web_io::PlatformInstant;
use eframe::egui;
use egui_plot::{MarkerShape, PlotUi, Points};
use math_commons::ast::CompiledExpr;
use math_explorer::pure_math::analysis::ode::ArrayState;
use math_explorer::pure_math::analysis::optimization::{
    CalibrationSummary, DEFAULT_SHOOTING_MAX_DT, ParameterBounds,
};
use models::{
    CompiledCoupledOde, CompiledScalarOde, OscArrayOde, ScalarExpOde, ScalarLogisticOde,
    advance_from_zero, solve_shooting,
};

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
            Ok((names, summary)) => {
                let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
                self.commit_ode_fit(ctx, &name_refs, &summary, &times, &observed, elapsed_ms)
            }
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
        let mut scalar_map = self.math_input.params_map();
        let mut coupled_map = self.coupled_bank.params_map();
        let is_coupled_custom = self.preset == OdePreset::Custom && self.custom_coupled;

        for (&name, &val) in names.iter().zip(params.iter()) {
            if is_coupled_custom && !matches!(name, "y0" | "v0") {
                coupled_map.insert(name.to_string(), val);
                continue;
            }
            match name {
                "k" => self.param_k = val,
                "r" => self.param_r = val,
                "K" | "cap_k" => self.param_cap_k = val,
                "y0" => self.ic_y0 = val,
                "v0" => self.ic_v0 = val,
                other => {
                    scalar_map.insert(other.to_string(), val);
                }
            }
        }
        if self.preset == OdePreset::Custom {
            if self.custom_coupled {
                self.coupled_bank.set_params_map(&coupled_map);
            } else {
                self.math_input.set_params_map(&scalar_map);
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
    ) -> Result<(Vec<String>, CalibrationSummary), String> {
        let max_dt = self.dt.clamp(1e-4, DEFAULT_SHOOTING_MAX_DT);
        let bounds = [
            ParameterBounds::new(-5.0, 5.0).unwrap(),
            ParameterBounds::new(-50.0, 50.0).unwrap(),
        ];
        let summary = solve_shooting(
            ScalarExpOde { k: self.param_k },
            &ArrayState([self.ic_y0]),
            times,
            observed,
            max_dt,
            &[self.param_k, self.ic_y0],
            &bounds,
            move |sys, p, t0, st| {
                sys.k = p[0];
                st.0[0] = p[1];
                advance_from_zero(sys, t0, max_dt, st, |s| s.0[0].is_finite())
            },
            |st| st.0[0],
            |st| st.0[0].is_finite(),
        )?;
        Ok((vec!["k".to_string(), "y0".to_string()], summary))
    }

    fn fit_harmonic(
        &self,
        times: &[f64],
        observed: &[f64],
    ) -> Result<(Vec<String>, CalibrationSummary), String> {
        let max_dt = self.dt.clamp(1e-4, DEFAULT_SHOOTING_MAX_DT);
        let bounds = [
            ParameterBounds::new(0.05, 10.0).unwrap(),
            ParameterBounds::new(-20.0, 20.0).unwrap(),
            ParameterBounds::new(-20.0, 20.0).unwrap(),
        ];
        let init = [self.param_k.clamp(0.05, 10.0), self.ic_y0, self.ic_v0];
        let summary = solve_shooting(
            OscArrayOde { k: self.param_k },
            &ArrayState([self.ic_y0, self.ic_v0]),
            times,
            observed,
            max_dt,
            &init,
            &bounds,
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
        )?;
        Ok((
            vec!["k".to_string(), "y0".to_string(), "v0".to_string()],
            summary,
        ))
    }

    fn fit_logistic(
        &self,
        times: &[f64],
        observed: &[f64],
    ) -> Result<(Vec<String>, CalibrationSummary), String> {
        let max_dt = self.dt.clamp(1e-4, DEFAULT_SHOOTING_MAX_DT);
        let max_obs = observed.iter().copied().fold(10.0_f64, f64::max);
        let bounds = [
            ParameterBounds::new(0.01, 5.0).unwrap(),
            ParameterBounds::new(0.5, (max_obs * 5.0).max(100.0)).unwrap(),
            ParameterBounds::new(1e-3, (max_obs * 2.0).max(50.0)).unwrap(),
        ];
        let init = [
            self.param_r.clamp(0.05, 5.0),
            self.param_cap_k.max(1.0),
            observed[0].clamp(1e-2, max_obs * 1.5),
        ];
        let summary = solve_shooting(
            ScalarLogisticOde {
                r: self.param_r,
                cap_k: self.param_cap_k,
            },
            &ArrayState([self.ic_y0]),
            times,
            observed,
            max_dt,
            &init,
            &bounds,
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
        )?;
        Ok((
            vec!["r".to_string(), "K".to_string(), "y0".to_string()],
            summary,
        ))
    }

    fn fit_custom_ast(
        &self,
        times: &[f64],
        observed: &[f64],
    ) -> Result<(Vec<String>, CalibrationSummary), String> {
        if self.custom_coupled {
            if !self.coupled_bank.is_valid() {
                return Err("Enter valid coupled differential equations (y', v') first".into());
            }
        } else if !self.math_input.is_valid() {
            return Err("Enter a valid custom differential equation y' = f(t, y) first".into());
        }
        match self.build_custom_ode_spec() {
            Some(CustomOdeSpec::Scalar {
                expr,
                params,
                param_len,
            }) => self.fit_compiled_scalar(expr, params, param_len, times, observed),
            Some(CustomOdeSpec::Coupled {
                dy,
                dv,
                params,
                param_len,
            }) => self.fit_compiled_coupled(dy, dv, params, param_len, times, observed),
            None => Err("Enter a valid custom differential equation y' = f(t, y) first".into()),
        }
    }

    fn fit_compiled_scalar(
        &self,
        expr: CompiledExpr,
        params: [f64; MAX_SYNTHESIZED_PARAMS],
        param_len: usize,
        times: &[f64],
        observed: &[f64],
    ) -> Result<(Vec<String>, CalibrationSummary), String> {
        let mut names = Vec::with_capacity(param_len + 1);
        let mut init = Vec::with_capacity(param_len + 1);
        let mut bounds = Vec::with_capacity(param_len + 1);
        for (idx, raw) in expr.param_names().iter().take(param_len).enumerate() {
            if raw == "cap_k" || raw == "K" {
                names.push("K".to_string());
                init.push(params[idx].clamp(0.5, 500.0));
                bounds.push(ParameterBounds::new(0.1, 500.0).unwrap());
            } else {
                names.push(raw.clone());
                init.push(params[idx].clamp(-10.0, 10.0));
                bounds.push(ParameterBounds::new(-10.0, 10.0).unwrap());
            }
        }
        names.push("y0".to_string());
        init.push(self.ic_y0.clamp(-50.0, 50.0));
        bounds.push(ParameterBounds::new(-50.0, 50.0).unwrap());

        let max_dt = self.dt.clamp(1e-4, DEFAULT_SHOOTING_MAX_DT);
        let dyn_ode = CompiledScalarOde {
            expr,
            params,
            param_len,
        };
        let summary = solve_shooting(
            dyn_ode,
            &ArrayState([self.ic_y0]),
            times,
            observed,
            max_dt,
            &init,
            &bounds,
            move |sys, p, t0, st| {
                sys.params[..param_len].copy_from_slice(&p[..param_len]);
                st.0[0] = p[param_len];
                advance_from_zero(sys, t0, max_dt, st, |s| s.0[0].is_finite())
            },
            |st| st.0[0],
            |st| st.0[0].is_finite(),
        )?;
        Ok((names, summary))
    }

    fn fit_compiled_coupled(
        &self,
        dy: CompiledExpr,
        dv: CompiledExpr,
        params: [f64; MAX_SYNTHESIZED_PARAMS],
        param_len: usize,
        times: &[f64],
        observed: &[f64],
    ) -> Result<(Vec<String>, CalibrationSummary), String> {
        let mut names = Vec::with_capacity(param_len + 2);
        let mut init = Vec::with_capacity(param_len + 2);
        let mut bounds = Vec::with_capacity(param_len + 2);
        for (idx, raw) in self.coupled_bank.param_names().iter().take(param_len).enumerate() {
            names.push(raw.clone());
            init.push(params[idx].clamp(-10.0, 10.0));
            bounds.push(ParameterBounds::new(-10.0, 10.0).unwrap());
        }
        names.extend(["y0".to_string(), "v0".to_string()]);
        init.extend([self.ic_y0.clamp(-50.0, 50.0), self.ic_v0.clamp(-50.0, 50.0)]);
        bounds.extend([
            ParameterBounds::new(-50.0, 50.0).unwrap(),
            ParameterBounds::new(-50.0, 50.0).unwrap(),
        ]);

        let max_dt = self.dt.clamp(1e-4, DEFAULT_SHOOTING_MAX_DT);
        let dyn_ode = CompiledCoupledOde {
            dy,
            dv,
            params,
            param_len,
        };
        let summary = solve_shooting(
            dyn_ode,
            &ArrayState([self.ic_y0, self.ic_v0]),
            times,
            observed,
            max_dt,
            &init,
            &bounds,
            move |sys, p, t0, st| {
                sys.params[..param_len].copy_from_slice(&p[..param_len]);
                st.0 = [p[param_len], p[param_len + 1]];
                advance_from_zero(sys, t0, max_dt, st, |s| s.0.iter().all(|v| v.is_finite()))
            },
            |st| st.0[0],
            |st| st.0.iter().all(|v| v.is_finite()),
        )?;
        Ok((names, summary))
    }
}
