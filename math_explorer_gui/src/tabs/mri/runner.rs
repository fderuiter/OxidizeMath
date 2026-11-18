#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::DiagnosticReport;
use math_explorer::physics::mri::bloch::BlochSimulator;
use math_explorer::pure_math::analysis::ode::{RungeKutta4, TimeStepper};
use nalgebra::Vector3;
use std::collections::VecDeque;
use std::sync::Arc;

#[allow(missing_docs)]
pub struct MriSharedParams {
    pub t1: f64,
    pub t2: f64,
    pub b_field: Vector3<f64>,
    pub time_scale: f64,
}

#[derive(Clone)]
#[allow(missing_docs)]
pub struct MriSnapshot {
    pub current_time: f64,
    pub magnetization: Vector3<f64>,
    pub history: VecDeque<(f64, Vector3<f64>)>,
}

#[allow(missing_docs)]
pub struct MriRunner {
    pub simulator: BlochSimulator,
    pub solver: RungeKutta4<Vector3<f64>>,
    pub params: Arc<std::sync::RwLock<MriSharedParams>>,
    pub current_time: f64,
    pub history: VecDeque<(f64, Vector3<f64>)>,
    pub max_history: usize,
}

impl MriRunner {
    fn sync_params_and_get_scale(&mut self) -> f64 {
        let (t1, t2, b_field, time_scale) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.t1, p.t2, p.b_field, p.time_scale)
        };
        self.simulator.t1 = t1;
        self.simulator.t2 = t2;
        self.simulator.b_field = b_field;
        time_scale
    }

    fn advance_bloch_ticks(&mut self, ticks: usize) {
        let dt = 0.001;
        for _ in 0..ticks {
            TimeStepper::step_with(&mut self.simulator, &mut self.solver, dt);
            self.current_time += dt;

            if self.history.len() >= self.max_history {
                self.history.pop_front();
            }
            self.history
                .push_back((self.current_time, self.simulator.magnetization));
        }
    }
}

impl SimulationRunner for MriRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        if let SimCommand::Reset = cmd {
            self.simulator.magnetization = Vector3::new(self.simulator.m0, 0.0, 0.0);
            self.current_time = 0.0;
            self.history.clear();
        }
    }

    fn step(&mut self) {
        let time_scale = self.sync_params_and_get_scale();
        let steps = time_scale.ceil().max(1.0) as usize;
        self.advance_bloch_ticks(steps);
    }

    fn step_once(&mut self) {
        let _ = self.sync_params_and_get_scale();
        self.advance_bloch_ticks(1);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let snapshot = MriSnapshot {
            current_time: self.current_time,
            magnetization: self.simulator.magnetization,
            history: self.history.clone(),
        };
        StateSnapshot {
            width: 0,
            height: 0,
            pixels: Arc::new(std::sync::RwLock::new(Vec::new())),
            custom_data: vec![
                self.current_time,
                self.simulator.magnetization.x,
                self.simulator.magnetization.y,
                self.simulator.magnetization.z,
            ],
            structured_data: Some(Box::new(snapshot)),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn is_finite_state(&self) -> bool {
        let m = self.simulator.magnetization;
        self.current_time.is_finite() && m.x.is_finite() && m.y.is_finite() && m.z.is_finite()
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        if self.is_finite_state() {
            None
        } else {
            Some(DiagnosticReport::new(
                self.current_time,
                self.history.len(),
                "magnetization",
                "Non-finite Bloch magnetization vector encountered",
            ))
        }
    }
}
