//! Empirical outbreak CSV ingestion, RK4 single-shooting calibration (`beta`, `gamma`, `i0`),
//! and propagated $R_0 = \beta / \gamma$ 95% confidence intervals for [`super::sir::SirTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::sir::SirTool;
use crate::framework::empirical_csv::{CalibrationReport, EmpiricalPanelOptions};
use crate::web_io::PlatformInstant;
use eframe::egui;
use egui_plot::{MarkerShape, PlotUi, Points};
use math_explorer::epidemiology::compartmental::SIRState;
use math_explorer::pure_math::analysis::ode::{OdeSystem, RungeKutta4, Solver};
use math_explorer::pure_math::analysis::optimization::{
    CalibrationSummary, LevenbergMarquardtSolver, LevenbergMarquardtWorkspace,
    MAX_SUBSTEPS_PER_INTERVAL, ParameterBounds, Rk4ShootingObjective,
};

struct SirShootingSystem {
    n: f64,
    beta: f64,
    gamma: f64,
}

impl OdeSystem<SIRState> for SirShootingSystem {
    fn derivative(&self, _t: f64, y: &SIRState) -> SIRState {
        let inf = self.beta * y.s * y.i / self.n.max(1.0);
        let rec = self.gamma * y.i;
        SIRState {
            s: -inf,
            i: inf - rec,
            r: rec,
        }
    }
}

/// Compartment observable targeted during outbreak CSV calibration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SirObservedCompartment {
    /// Active infectious cases $I(t)$ (default epidemiological case curve).
    #[default]
    Infected,
    /// Cumulative recovered/removed individuals $R(t)$.
    Recovered,
    /// Remaining susceptible population $S(t)$.
    Susceptible,
}

impl SirObservedCompartment {
    fn label(self) -> &'static str {
        match self {
            Self::Infected => "Infected I(t)",
            Self::Recovered => "Recovered R(t)",
            Self::Susceptible => "Susceptible S(t)",
        }
    }

    fn extract(self, state: &SIRState) -> f64 {
        match self {
            Self::Infected => state.i,
            Self::Recovered => state.r,
            Self::Susceptible => state.s,
        }
    }
}

impl SirTool {
    /// Renders compartment target selector, CSV column mapping, `Fit Parameters`,
    /// $R_0$ 95% confidence intervals, and `Download Residuals CSV`.
    pub(super) fn render_sir_calibration_panel(&mut self, ui: &mut egui::Ui) -> bool {
        if self.empirical_csv.table.is_some() {
            ui.horizontal_wrapped(|ui| {
                ui.label("Observed Compartment:");
                egui::ComboBox::from_id_salt("sir_obs_compartment")
                    .selected_text(self.observed_compartment.label())
                    .show_ui(ui, |ui| {
                        for comp in [
                            SirObservedCompartment::Infected,
                            SirObservedCompartment::Recovered,
                            SirObservedCompartment::Susceptible,
                        ] {
                            ui.selectable_value(
                                &mut self.observed_compartment,
                                comp,
                                comp.label(),
                            );
                        }
                    });
            });
        }
        let opts = EmpiricalPanelOptions {
            id_salt: "sir_outbreak_csv",
            x_label: "Time Column (days)",
            y_label: "Observed Cases Column",
            aux_label: None,
            residual_filename: "sir_outbreak_residuals.csv",
            fit_button_label: "Fit Parameters",
        };
        if self.empirical_csv.show_panel(ui, &opts) {
            let ctx = ui.ctx().clone();
            return self.calibrate_from_loaded_csv(Some(&ctx)).is_ok();
        }
        false
    }

    /// Renders empirical outbreak scatter points onto the SIR plot canvas.
    pub(super) fn render_empirical_overlay(&self, plot_ui: &mut PlotUi<'_>, color: egui::Color32) {
        let Some(ref table) = self.empirical_csv.table else {
            return;
        };
        let pts = table.scatter_points(self.empirical_csv.x_col, self.empirical_csv.y_col);
        if pts.is_empty() {
            return;
        }
        plot_ui.points(
            Points::new(
                format!("Observed {} ({})", self.observed_compartment.label(), table.name),
                pts,
            )
            .radius(4.0_f32)
            .shape(MarkerShape::Diamond)
            .color(color),
        );
    }

    /// Calibrates `beta`, `gamma`, and `i0` against the loaded outbreak CSV dataset via RK4 shooting.
    pub fn calibrate_from_loaded_csv(&mut self, ctx: Option<&egui::Context>) -> Result<(), String> {
        let Some(ref table) = self.empirical_csv.table else {
            return Err("Load an outbreak CSV dataset before fitting SIR parameters".to_string());
        };
        let (times, observed) =
            table.extract_xy_sorted(self.empirical_csv.x_col, self.empirical_csv.y_col)?;
        if times.len() < 3 {
            let err = "SIR calibration requires at least 3 timepoints to fit beta, gamma, and I0"
                .to_string();
            self.empirical_csv.record_error(ctx, err.clone());
            return Err(err);
        }
        if times[0] < 0.0 || observed.iter().any(|&v| v < 0.0) {
            let err = "SIR outbreak timestamps and case counts must be non-negative".to_string();
            self.empirical_csv.record_error(ctx, err.clone());
            return Err(err);
        }
        let max_obs = observed.iter().copied().fold(0.0_f64, f64::max);
        if max_obs > self.n {
            self.n = (max_obs * 1.25).clamp(100.0, 100_000.0);
        }

        let start = PlatformInstant::now();
        let (summary, cov) = match self.solve_sir_shooting(&times, &observed) {
            Ok(pair) => pair,
            Err(err) => {
                self.empirical_csv.record_error(ctx, err.clone());
                return Err(err);
            }
        };
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.commit_sir_fit(ctx, &summary, &cov, &times, &observed, elapsed_ms)
    }

    fn solve_sir_shooting(
        &self,
        times: &[f64],
        observed: &[f64],
    ) -> Result<(CalibrationSummary, Vec<f64>), String> {
        let pop_n = self.n.clamp(100.0, 100_000.0);
        let comp = self.observed_compartment;
        let max_dt = 0.05_f64;
        let example = SIRState {
            s: pop_n - self.i0.clamp(1.0, pop_n * 0.5),
            i: self.i0.clamp(1.0, pop_n * 0.5),
            r: 0.0,
        };
        let mut obj = Rk4ShootingObjective::new(
            SirShootingSystem {
                n: pop_n,
                beta: self.beta.clamp(1e-3, 5.0),
                gamma: self.gamma.clamp(1e-3, 1.0),
            },
            &example,
            times,
            observed,
            max_dt,
            move |sys, p, t0, st| {
                let (beta, gamma, i0) = (p[0], p[1], p[2]);
                if beta <= 0.0 || gamma <= 0.0 || i0 < 1.0 || i0 >= pop_n {
                    return false;
                }
                sys.beta = beta;
                sys.gamma = gamma;
                *st = SIRState {
                    s: pop_n - i0,
                    i: i0,
                    r: 0.0,
                };
                advance_sir_from_zero(sys, t0, max_dt, st)
            },
            move |st| comp.extract(st),
            |st| st.s.is_finite() && st.i.is_finite() && st.r.is_finite(),
        )
        .map_err(|e| e.to_string())?;

        let max_i0 = (pop_n * 0.5).max(2.0);
        let bounds = [
            ParameterBounds::new(1e-3, 5.0).unwrap(),
            ParameterBounds::new(1e-3, 1.0).unwrap(),
            ParameterBounds::new(1.0, max_i0).unwrap(),
        ];
        let seed_i0 = if comp == SirObservedCompartment::Infected && times[0] == 0.0 {
            observed[0].clamp(1.0, max_i0)
        } else {
            self.i0.clamp(1.0, max_i0)
        };
        let init = [
            self.beta.clamp(0.05, 4.5),
            self.gamma.clamp(0.02, 0.9),
            seed_i0,
        ];
        let mut ws = LevenbergMarquardtWorkspace::new(times.len(), 3).map_err(|e| e.to_string())?;
        let summary = LevenbergMarquardtSolver::default()
            .fit(&mut obj, &init, &bounds, &mut ws)
            .map_err(|e| e.to_string())?;
        Ok((summary, ws.covariance().to_vec()))
    }

    fn commit_sir_fit(
        &mut self,
        ctx: Option<&egui::Context>,
        summary: &CalibrationSummary,
        cov: &[f64],
        times: &[f64],
        observed: &[f64],
        elapsed_ms: f64,
    ) -> Result<(), String> {
        let (beta, gamma, i0) = (
            summary.parameters[0],
            summary.parameters[1],
            summary.parameters[2],
        );
        if !beta.is_finite() || !gamma.is_finite() || !i0.is_finite() || gamma <= 0.0 {
            let err = "Rejected non-finite calibrated SIR state".to_string();
            self.empirical_csv.record_error(ctx, err.clone());
            return Err(err);
        }
        self.beta = beta.clamp(0.0, 5.0);
        self.gamma = gamma.clamp(1e-4, 1.0);
        self.i0 = i0.clamp(1.0, self.n * 0.5);
        if let Some(&t_last) = times.last() {
            if t_last > self.duration {
                self.duration = t_last.clamp(10.0, 365.0);
            }
        }
        self.recalculate_from_calibration();

        let mut report = CalibrationReport::from_summary(
            &["beta", "gamma", "I0"],
            summary,
            times,
            observed,
            elapsed_ms,
        )?;
        let (r0, r0_ci) = compute_r0_confidence_interval(self.beta, self.gamma, summary, cov);
        report.derived_metrics.push((
            "R₀ = β/γ (95% CI)".to_string(),
            format!("{r0:.3} ± {r0_ci:.3}"),
        ));
        let announce = format!(
            "Calibrated SIR outbreak: R0 {:.2} ± {:.2}, RMSE {:.3}, R² {:.4}",
            r0, r0_ci, report.rmse, report.r_squared
        );
        self.empirical_csv.record_fit_success(ctx, report, announce);
        Ok(())
    }
}

fn compute_r0_confidence_interval(
    beta: f64,
    gamma: f64,
    summary: &CalibrationSummary,
    cov: &[f64],
) -> (f64, f64) {
    let r0 = beta / gamma.max(1e-12);
    if cov.len() < 9 || gamma <= 1e-12 {
        return (r0, 0.0);
    }
    let var_beta = cov[0];
    let cov_bg = cov[1];
    let var_gamma = cov[4];
    let db = 1.0 / gamma;
    let dg = -beta / (gamma * gamma);
    let var_r0 = (db * db * var_beta + 2.0 * db * dg * cov_bg + dg * dg * var_gamma).max(0.0);
    let se_r0 = var_r0.sqrt();
    let t_mult = if summary.standard_errors[0] > 1e-15 {
        summary.ci_95_half_widths[0] / summary.standard_errors[0]
    } else {
        1.96
    };
    (r0, (t_mult * se_r0).max(0.0))
}

fn advance_sir_from_zero<S: OdeSystem<SIRState>>(
    sys: &S,
    t0: f64,
    max_dt: f64,
    state: &mut SIRState,
) -> bool {
    if t0 <= 0.0 {
        return state.s.is_finite() && state.i.is_finite() && state.r.is_finite();
    }
    let steps = ((t0 / max_dt).ceil() as usize).clamp(1, MAX_SUBSTEPS_PER_INTERVAL);
    let dt = t0 / (steps as f64);
    let mut rk4 = RungeKutta4::new(state);
    let mut t = 0.0;
    for _ in 0..steps {
        rk4.step(sys, t, state, dt);
        if !state.s.is_finite() || !state.i.is_finite() || !state.r.is_finite() {
            return false;
        }
        t += dt;
    }
    true
}
