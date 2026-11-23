#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{AttractorMode, AttractorPlotter, SolverExecutionState, DEFAULT_FLOW_EXPRS};
use crate::framework::recovery::DiagnosticReport;
use crate::framework::{Camera3D, InteractiveTool};
use std::collections::BTreeMap;

impl AttractorPlotter {
    pub(super) fn build_diagnostic_report(&self) -> Option<DiagnosticReport> {
        if let Some(report) = self.worker.diagnostic_report() {
            return Some(report.clone());
        }
        let step_dt = if self.dt.is_finite() && self.dt > 0.0 {
            self.dt
        } else {
            0.01
        };
        if let SolverExecutionState::Diverged {
            invalid_vec,
            diverged_at_step,
        } = self.execution_state
        {
            let var = if !invalid_vec.x.is_finite() {
                "x"
            } else if !invalid_vec.y.is_finite() {
                "y"
            } else if !invalid_vec.z.is_finite() {
                "z"
            } else {
                "lorenz_state"
            };
            return Some(DiagnosticReport::new(
                diverged_at_step as f64 * step_dt,
                diverged_at_step,
                var,
                format!(
                    "Non-finite attractor state [{:.2}, {:.2}, {:.2}] at step {}",
                    invalid_vec.x, invalid_vec.y, invalid_vec.z, diverged_at_step
                ),
            ));
        }
        if !self.has_finite_parameters() {
            return Some(DiagnosticReport::new(
                self.step_count as f64 * step_dt,
                self.step_count,
                "parameters",
                "Non-finite attractor parameter or state detected.",
            ));
        }
        None
    }
}

const fn default_annotations_enabled() -> bool {
    true
}

#[derive(serde::Serialize)]
struct AttractorSaveState {
    sigma: f64,
    rho: f64,
    beta: f64,
    dt: f64,
    simulation_speed: usize,
    execution_state: SolverExecutionState,
    show_annotations: bool,
    camera: Camera3D,
    #[serde(skip_serializing_if = "AttractorMode::is_lorenz")]
    mode: AttractorMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    custom_exprs: Option<[String; 3]>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    custom_params: BTreeMap<String, f64>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AttractorState {
    sigma: f64,
    rho: f64,
    beta: f64,
    dt: f64,
    simulation_speed: usize,
    #[serde(default)]
    execution_state: Option<SolverExecutionState>,
    #[serde(default = "default_annotations_enabled")]
    show_annotations: bool,
    #[serde(default)]
    paused: Option<bool>,
    #[serde(default)]
    camera: Option<Camera3D>,
    #[serde(default)]
    pitch: Option<f32>,
    #[serde(default)]
    yaw: Option<f32>,
    #[serde(default)]
    zoom: Option<f32>,
    #[serde(default)]
    mode: AttractorMode,
    #[serde(default)]
    custom_exprs: Option<[String; 3]>,
    #[serde(default)]
    custom_params: BTreeMap<String, f64>,
}

impl AttractorPlotter {
    pub(super) fn serialize_state(&self) -> Option<String> {
        let expr_vec = self.flow_bank.expressions();
        let nondefault_exprs = expr_vec
            .iter()
            .zip(DEFAULT_FLOW_EXPRS.iter())
            .any(|(a, b)| a != b);
        let custom_exprs = (self.mode == AttractorMode::Custom || nondefault_exprs).then(|| {
            [
                expr_vec[0].clone(),
                expr_vec[1].clone(),
                expr_vec[2].clone(),
            ]
        });
        let custom_params = if self.mode == AttractorMode::Custom {
            self.flow_bank.params_map()
        } else {
            BTreeMap::new()
        };
        serde_json::to_string(&AttractorSaveState {
            sigma: self.system.sigma,
            rho: self.system.rho,
            beta: self.system.beta,
            dt: self.dt,
            simulation_speed: self.simulation_speed,
            execution_state: self.execution_state,
            show_annotations: self.show_annotations,
            camera: self.camera,
            mode: self.mode,
            custom_exprs,
            custom_params,
        })
        .ok()
    }

    pub(super) fn deserialize_state(&mut self, state: &str) -> Result<(), String> {
        <Self as crate::framework::InteractiveTool>::state_bounds(self).validate_json(state)?;
        let s: AttractorState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid AttractorPlotter state: {e}"))?;
        if !s.sigma.is_finite()
            || !s.rho.is_finite()
            || !s.beta.is_finite()
            || !s.dt.is_finite()
            || s.dt <= 0.0
            || !(1..=super::MAX_SIMULATION_SPEED).contains(&s.simulation_speed)
            || s.custom_params.values().any(|v| !v.is_finite())
        {
            return Err(format!(
                "AttractorPlotter parameters must be finite with dt > 0 and simulation_speed in 1..={}",
                super::MAX_SIMULATION_SPEED
            ));
        }
        let camera = Camera3D::from_preset_fields(
            s.camera,
            s.pitch,
            s.yaw,
            s.zoom,
            Some(Camera3D::new(0.0, 0.0, 1.0)),
        )?;

        self.mode = s.mode;
        self.apply_custom_flow_state(s.custom_exprs.as_ref(), &s.custom_params);
        self.system.sigma = s.sigma;
        self.system.rho = s.rho;
        self.system.beta = s.beta;
        self.dt = s.dt;
        self.simulation_speed = s.simulation_speed;
        self.show_annotations = s.show_annotations;
        self.camera.apply_pose(camera);
        self.dragged_callout = None;
        self.reset();
        let _ = self.worker.update();
        let execution_state = s.execution_state.unwrap_or(if s.paused == Some(true) {
            SolverExecutionState::Paused
        } else {
            SolverExecutionState::Ready
        });
        self.execution_state = execution_state;
        if let SolverExecutionState::Diverged {
            invalid_vec,
            diverged_at_step,
        } = execution_state
        {
            self.system.state.vec = invalid_vec;
            self.step_count = diverged_at_step;
        }
        Ok(())
    }

    fn apply_custom_flow_state(
        &mut self,
        custom_exprs: Option<&[String; 3]>,
        custom_params: &BTreeMap<String, f64>,
    ) {
        if let Some(exprs) = custom_exprs {
            if !custom_params.is_empty() {
                self.flow_bank.load_state(exprs, custom_params);
            } else {
                self.flow_bank.set_expressions(exprs);
            }
        } else {
            self.flow_bank = super::default_flow_bank();
            if !custom_params.is_empty() {
                self.flow_bank.set_params_map(custom_params);
            }
        }
    }
}
