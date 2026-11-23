//! Ensemble fan seeding, 3-body perturbation helpers, and Chirikov standard map stepping.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::simulation::{push_capped_ring, push_sliding_trail, WARMUP_STEPS_ON_SEED};
use super::{HamiltonianChaosTool, HamiltonianModelKind, MAX_SECTION_POINTS, MAX_TRAIL_POINTS};
use math_explorer::physics::chaos::hamiltonian::{seed_double_pendulum_fan, ChirikovStandardMap};
use math_explorer::pure_math::analysis::ode::ArrayState;

const PENDULUM_FAN_COUNT: usize = 8;

impl HamiltonianChaosTool {
    /// Launches a zero-allocation ensemble fan of nearby double-pendulum trajectories around `cursor_qp`.
    pub fn seed_pendulum_fan(&mut self) -> Result<usize, &'static str> {
        if self.model != HamiltonianModelKind::DoublePendulum {
            return Err("Pendulum fan requires Double Pendulum mode.");
        }
        let mut fan_states = [ArrayState::<4>::zeros(); PENDULUM_FAN_COUNT];
        let written = seed_double_pendulum_fan(
            &self.pendulum,
            self.energy,
            self.cursor_qp[0],
            self.cursor_qp[1],
            self.pendulum_fan_spread,
            &mut fan_states,
        );
        if written == 0 {
            return Err("Cursor lies outside the Double Pendulum energy shell.");
        }
        self.clear_orbits();
        for s in fan_states.iter().take(written) {
            let _ = self.seed_orbit_no_warmup([s.0[1], s.0[3]]);
        }
        self.step_batch(WARMUP_STEPS_ON_SEED);
        self.state_rev = self.state_rev.wrapping_add(1);
        Ok(written)
    }

    /// Re-seeds the Figure-8 choreography with the zero-linear-momentum perturbation slider.
    pub fn apply_figure_eight_perturbation(&mut self) {
        if self.model != HamiltonianModelKind::FigureEightThreeBody {
            return;
        }
        self.clear_orbits();
        let _ = self.seed_orbit_no_warmup([0.0, 0.0]);
        if self.figure_eight_perturbation.abs() > 1e-9 {
            let _ = self.seed_orbit_no_warmup([0.0, self.figure_eight_perturbation]);
        }
        self.step_batch(WARMUP_STEPS_ON_SEED);
        self.state_rev = self.state_rev.wrapping_add(1);
    }

    pub(super) fn step_chirikov_once(&mut self) {
        self.chirikov = ChirikovStandardMap::new(self.chirikov_k);
        for slot in &mut self.orbits {
            let det_err =
                (self.chirikov.jacobian_determinant(slot.map_state.0[0]) - 1.0).abs() * 100.0;
            self.chirikov.step_in_place(&mut slot.map_state);
            let qp = [slot.map_state.0[0], slot.map_state.0[1]];
            push_capped_ring(
                &mut slot.section_points,
                &mut slot.section_write_idx,
                MAX_SECTION_POINTS,
                qp,
            );
            slot.last_rel_err_pct = det_err;
            push_sliding_trail(
                &mut slot.trail,
                MAX_TRAIL_POINTS,
                [qp[0].cos(), qp[0].sin()],
            );
        }
        self.step_ghost_chirikov();
    }
}
