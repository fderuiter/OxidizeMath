//! Ensemble fan seeding, Lyapunov estimation, 3-body perturbation, and Chirikov stepping.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::simulation::{push_capped_ring, push_sliding_trail, WARMUP_STEPS_ON_SEED};
use super::{HamiltonianChaosTool, HamiltonianModelKind, MAX_SECTION_POINTS, MAX_TRAIL_POINTS};
use math_explorer::physics::chaos::hamiltonian::{
    perturb_figure_eight_momentum, seed_double_pendulum_fan, ChirikovStandardMap,
};
use math_explorer::physics::chaos::metrics::{array_lyapunov_exponent, lyapunov_exponent_4d};
use math_explorer::pure_math::analysis::ode::ArrayState;

const PENDULUM_FAN_COUNT: usize = 8;

impl HamiltonianChaosTool {
    /// Recomputes the finite-time largest Lyapunov exponent $\lambda_{\max}$ for the lead orbit.
    pub(crate) fn refresh_lyapunov_estimate(&mut self) {
        let Some(lead) = self.orbits.last() else {
            self.lyapunov_estimate = 0.0;
            return;
        };
        let s2 = lead.map_state;
        let s4 = lead.state4;
        let s12 = lead.state12;
        self.lyapunov_estimate = match self.model {
            HamiltonianModelKind::HenonHeiles => {
                lyapunov_exponent_4d(&self.henon, &mut self.mid4, s4, 0.025, 24, 0.25)
                    .unwrap_or(0.0)
            }
            HamiltonianModelKind::DoublePendulum => {
                lyapunov_exponent_4d(&self.pendulum, &mut self.gl4_4, s4, 0.02, 24, 0.2)
                    .unwrap_or(0.0)
            }
            HamiltonianModelKind::FigureEightThreeBody => {
                array_lyapunov_exponent(&self.three_body, &mut self.mid12, s12, 0.015, 16, 0.15)
                    .unwrap_or(0.0)
            }
            HamiltonianModelKind::ChirikovStandardMap => {
                self.chirikov = ChirikovStandardMap::new(self.chirikov_k);
                self.chirikov.finite_time_lyapunov(s2, 64)
            }
        };
    }

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
        self.refresh_lyapunov_estimate();
        self.step_batch(WARMUP_STEPS_ON_SEED);
        self.state_rev = self.state_rev.wrapping_add(1);
        Ok(written)
    }

    pub(super) fn seed_three_body_state(
        &self,
        qp: [f64; 2],
    ) -> Result<ArrayState<12>, &'static str> {
        if !(-0.45..=0.45).contains(&qp[0]) || !(-0.45..=0.45).contains(&qp[1]) {
            return Err("Perturbation seed lies outside bounded 3-body domain [-0.45, 0.45].");
        }
        let d = self.three_body_perturbation.clamp(-0.20, 0.20);
        let dpx = (qp[1] + d).clamp(-0.5, 0.5);
        let mut state12 = perturb_figure_eight_momentum(dpx, -0.5 * dpx);
        state12.0[1] += qp[0];
        state12.0[3] -= qp[0];
        Ok(state12)
    }

    /// Re-seeds the Figure-8 choreography with the zero-linear-momentum perturbation slider.
    pub fn apply_figure_eight_perturbation(&mut self) {
        if self.model != HamiltonianModelKind::FigureEightThreeBody {
            return;
        }
        if self.orbits.is_empty() {
            let _ = self.seed_orbit_no_warmup([0.0, 0.0]);
            let _ = self.seed_orbit_no_warmup([0.035, -0.02]);
        }
        self.rebuild_existing_seeds();
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
