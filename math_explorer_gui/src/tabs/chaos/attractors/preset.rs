#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{AttractorPlotter, SolverExecutionState};

const fn default_annotations_enabled() -> bool {
    true
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AttractorState {
    sigma: f64,
    rho: f64,
    beta: f64,
    dt: f64,
    simulation_speed: usize,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    execution_state: Option<SolverExecutionState>,
    #[serde(default = "default_annotations_enabled")]
    show_annotations: bool,
    #[serde(skip_serializing, default)]
    paused: Option<bool>,
}

impl AttractorPlotter {
    pub(super) fn serialize_state(&self) -> Option<String> {
        serde_json::to_string(&AttractorState {
            sigma: self.system.sigma,
            rho: self.system.rho,
            beta: self.system.beta,
            dt: self.dt,
            simulation_speed: self.simulation_speed,
            execution_state: Some(self.execution_state),
            show_annotations: self.show_annotations,
            paused: None,
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
        {
            return Err(format!(
                "AttractorPlotter parameters must be finite with dt > 0 and simulation_speed in 1..={}",
                super::MAX_SIMULATION_SPEED
            ));
        }
        self.system.sigma = s.sigma;
        self.system.rho = s.rho;
        self.system.beta = s.beta;
        self.dt = s.dt;
        self.simulation_speed = s.simulation_speed;
        self.show_annotations = s.show_annotations;
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
}
