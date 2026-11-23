//! Synchronized explicit `RungeKutta4` comparator twin stepping and energy-drift history.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::simulation::{push_capped_ring, push_sliding_trail};
use super::{
    GhostComparator, HamiltonianChaosLab, OrbitSlot, MAX_DRIFT_SAMPLES, MAX_GHOST_SECTION_POINTS,
    MAX_TRAIL_POINTS,
};
use math_explorer::pure_math::analysis::ode::{
    HamiltonianSystem, OdeSystem, Solver, VectorOperations,
};

impl HamiltonianChaosLab {
    pub(super) fn init_ghost_from_slot(&mut self, slot: &OrbitSlot) {
        self.ghost = GhostComparator {
            active: true,
            initial_energy: slot.initial_energy,
            elapsed_time: 0.0,
            step_count: 0,
            state4: slot.state4,
            prev4: slot.prev4,
            deriv4: slot.deriv4,
            prev_deriv4: slot.prev_deriv4,
            state12: slot.state12,
            section_points: Vec::with_capacity(128),
            section_write_idx: 0,
            trail: Vec::with_capacity(MAX_TRAIL_POINTS),
            symplectic_drift: Vec::with_capacity(MAX_DRIFT_SAMPLES),
            rk4_drift: Vec::with_capacity(MAX_DRIFT_SAMPLES),
            symplectic_err_pct: 0.0,
            rk4_err_pct: 0.0,
        };
    }

    pub(super) fn step_ghost_4d_henon(&mut self, t: f64, dt: f64) {
        let Some(lead) = self.orbits.last() else {
            return;
        };
        if !self.ghost.active {
            return;
        }
        self.ghost.prev4.copy_from(&self.ghost.state4);
        self.ghost.prev_deriv4.copy_from(&self.ghost.deriv4);
        self.rk4_4.step(&self.henon, t, &mut self.ghost.state4, dt);
        if !self.ghost.state4.is_finite() {
            self.ghost.state4.copy_from(&self.ghost.prev4);
            return;
        }
        self.henon
            .derivative_in_place(t + dt, &self.ghost.state4, &mut self.ghost.deriv4);
        if let Some(hit) = self.poincare4.detect_crossing(
            t,
            dt,
            &self.ghost.prev4,
            &self.ghost.state4,
            &self.ghost.prev_deriv4,
            &self.ghost.deriv4,
        ) {
            push_capped_ring(
                &mut self.ghost.section_points,
                &mut self.ghost.section_write_idx,
                MAX_GHOST_SECTION_POINTS,
                [hit.q_sec, hit.p_sec],
            );
        }
        let rk4_err = self
            .henon
            .relative_energy_error(&self.ghost.state4, self.ghost.initial_energy)
            * 100.0;
        let pos = [self.ghost.state4.0[0], self.ghost.state4.0[1]];
        self.record_ghost_metrics(dt, lead.last_rel_err_pct, rk4_err, pos);
    }

    pub(super) fn step_ghost_4d_pendulum(&mut self, t: f64, dt: f64) {
        let Some(lead) = self.orbits.last() else {
            return;
        };
        if !self.ghost.active {
            return;
        }
        self.ghost.prev4.copy_from(&self.ghost.state4);
        self.ghost.prev_deriv4.copy_from(&self.ghost.deriv4);
        self.rk4_4
            .step(&self.pendulum, t, &mut self.ghost.state4, dt);
        self.pendulum
            .derivative_in_place(t + dt, &self.ghost.state4, &mut self.ghost.deriv4);
        if let Some(hit) = self.poincare4.detect_crossing(
            t,
            dt,
            &self.ghost.prev4,
            &self.ghost.state4,
            &self.ghost.prev_deriv4,
            &self.ghost.deriv4,
        ) {
            push_capped_ring(
                &mut self.ghost.section_points,
                &mut self.ghost.section_write_idx,
                MAX_GHOST_SECTION_POINTS,
                [hit.q_sec, hit.p_sec],
            );
        }
        let rk4_err = self
            .pendulum
            .relative_energy_error(&self.ghost.state4, self.ghost.initial_energy)
            * 100.0;
        let (_, bob2) = self.pendulum.cartesian_positions(&self.ghost.state4);
        self.record_ghost_metrics(dt, lead.last_rel_err_pct, rk4_err, bob2);
    }

    pub(super) fn step_ghost_12d_three_body(&mut self, t: f64, dt: f64) {
        let Some(lead) = self.orbits.last() else {
            return;
        };
        if !self.ghost.active {
            return;
        }
        self.rk4_12
            .step(&self.three_body, t, &mut self.ghost.state12, dt);
        let rk4_err = self
            .three_body
            .relative_energy_error(&self.ghost.state12, self.ghost.initial_energy)
            * 100.0;
        let pos = [self.ghost.state12.0[0], self.ghost.state12.0[1]];
        self.record_ghost_metrics(dt, lead.last_rel_err_pct, rk4_err, pos);
    }

    fn record_ghost_metrics(&mut self, dt: f64, sym_err: f64, rk4_err: f64, pos: [f64; 2]) {
        self.ghost.elapsed_time += dt;
        self.ghost.step_count += 1;
        self.ghost.symplectic_err_pct = sym_err;
        self.ghost.rk4_err_pct = rk4_err;
        push_sliding_trail(&mut self.ghost.trail, MAX_TRAIL_POINTS, pos);
        if self.ghost.step_count.is_multiple_of(4) || self.ghost.symplectic_drift.is_empty() {
            let time = self.ghost.elapsed_time;
            push_sliding_trail(
                &mut self.ghost.symplectic_drift,
                MAX_DRIFT_SAMPLES,
                [time, sym_err],
            );
            push_sliding_trail(
                &mut self.ghost.rk4_drift,
                MAX_DRIFT_SAMPLES,
                [time, rk4_err],
            );
        }
    }
}
