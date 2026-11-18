#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{AttractorPlotter, SolverExecutionState};

const fn default_annotations_enabled() -> bool {
    true
}

#[derive(serde::Serialize)]
struct SerializedAttractorState {
    sigma: f64,
    rho: f64,
    beta: f64,
    dt: f64,
    simulation_speed: usize,
    execution_state: SolverExecutionState,
    #[serde(default = "default_annotations_enabled")]
    show_annotations: bool,
}

#[derive(serde::Deserialize)]
struct DeserializedAttractorState {
    sigma: f64,
    rho: f64,
    beta: f64,
    dt: f64,
    simulation_speed: usize,
    #[serde(default)]
    execution_state: Option<SolverExecutionState>,
    #[serde(default = "default_annotations_enabled")]
    show_annotations: bool,
}

impl AttractorPlotter {
    pub(super) fn serialize_state(&self) -> Option<String> {
        serde_json::to_string(&SerializedAttractorState {
            sigma: self.system.sigma,
            rho: self.system.rho,
            beta: self.system.beta,
            dt: self.dt,
            simulation_speed: self.simulation_speed,
            execution_state: self.execution_state,
            show_annotations: self.show_annotations,
        })
        .ok()
    }

    pub(super) fn deserialize_state(&mut self, state: &str) -> Result<(), String> {
        let s: DeserializedAttractorState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid AttractorPlotter state: {e}"))?;
        if !s.sigma.is_finite() || !s.rho.is_finite() || !s.beta.is_finite() || !s.dt.is_finite() {
            return Err("AttractorPlotter parameters must be finite".to_string());
        }
        self.system.sigma = s.sigma;
        self.system.rho = s.rho;
        self.system.beta = s.beta;
        self.dt = s.dt;
        self.simulation_speed = s.simulation_speed;
        self.show_annotations = s.show_annotations;
        self.reset();
        let execution_state = s.execution_state.unwrap_or_else(|| {
            let legacy_paused = serde_json::from_str::<serde_json::Value>(state)
                .ok()
                .and_then(|v| v.get("paused").and_then(|p| p.as_bool()))
                .unwrap_or(false);
            if legacy_paused {
                SolverExecutionState::Paused
            } else {
                SolverExecutionState::Ready
            }
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
