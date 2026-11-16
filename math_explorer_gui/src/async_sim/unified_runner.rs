#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::unified::UnifiedModel;
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[allow(missing_docs)]
pub struct UnifiedSimRunner<M: UnifiedModel> {
    model: M,
    params: Arc<RwLock<HashMap<String, f64>>>,
    steps_per_frame: usize,
}

impl<M: UnifiedModel> UnifiedSimRunner<M> {
    #[allow(missing_docs)]
    pub fn new(params: Arc<RwLock<HashMap<String, f64>>>) -> Self {
        let initial_params = params.read().unwrap().clone();
        Self {
            model: M::new(&initial_params),
            params,
            steps_per_frame: 5,
        }
    }
}

impl<M: UnifiedModel> SimulationRunner for UnifiedSimRunner<M> {
    fn process_command(&mut self, cmd: SimCommand) {
        let current_params = self.params.read().unwrap().clone();
        match cmd {
            SimCommand::SetSpeed(speed) => self.steps_per_frame = speed,
            SimCommand::Reset => {
                self.model = M::new(&current_params);
            }
            _ => self.model.process_command(cmd, &current_params),
        }
    }

    fn step(&mut self) {
        let current_params = self.params.read().unwrap().clone();
        self.model.step(&current_params);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        self.model.get_snapshot()
    }

    fn get_steps_per_frame(&self) -> usize {
        self.steps_per_frame
    }

    fn diverged(&self) -> bool {
        self.model.diverged()
    }

    fn diagnostic_report(&self) -> Option<crate::async_sim::recovery::DiagnosticReport> {
        self.model.diagnostic_report()
    }
}
