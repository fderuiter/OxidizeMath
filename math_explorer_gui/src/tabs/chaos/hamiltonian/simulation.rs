//! Zero-allocation symplectic stepping and cubic-Hermite Poincaré section capture.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    GhostComparator, HamiltonianChaosLab, HamiltonianModelKind, OrbitSlot, SymplecticSolverChoice,
    FRAME_BUDGET_MS, MAX_ORBITS, MAX_SECTION_POINTS, MAX_TRAIL_POINTS,
};
use crate::web_io::PlatformInstant;
use math_explorer::physics::astrophysics::nbody::FigureEightThreeBody;
use math_explorer::physics::chaos::poincare::PoincareSection;
use math_explorer::pure_math::analysis::ode::{
    ArrayState, HamiltonianSystem, OdeSystem, Solver, SymplecticMethod, VectorOperations,
};
use std::f64::consts::PI;

const WARMUP_STEPS_ON_SEED: usize = 320;

pub(super) fn push_capped_ring(
    buf: &mut Vec<[f64; 2]>,
    write_idx: &mut usize,
    cap: usize,
    pt: [f64; 2],
) {
    if !pt[0].is_finite() || !pt[1].is_finite() {
        return;
    }
    if buf.len() < cap {
        buf.push(pt);
    } else if cap > 0 {
        let idx = *write_idx % cap;
        buf[idx] = pt;
        *write_idx = (*write_idx + 1) % cap;
    }
}

pub(super) fn push_sliding_trail(buf: &mut Vec<[f64; 2]>, cap: usize, pt: [f64; 2]) {
    if !pt[0].is_finite() || !pt[1].is_finite() {
        return;
    }
    if buf.len() >= cap {
        buf.remove(0);
    }
    buf.push(pt);
}

impl HamiltonianChaosLab {
    /// Allowed energy slider range `(min, max, default)` for the active model.
    #[must_use]
    pub const fn energy_bounds(&self) -> (f64, f64, f64) {
        match self.model {
            HamiltonianModelKind::HenonHeiles => (0.02, 0.1665, 0.125),
            HamiltonianModelKind::DoublePendulum => (4.0, 52.0, 26.0),
            HamiltonianModelKind::FigureEightThreeBody => (0.5, 2.5, 1.287),
        }
    }

    /// Phase-space viewport `[q_min, q_max, p_min, p_max]` for the Poincaré canvas.
    #[must_use]
    pub const fn poincare_bounds(&self) -> [f64; 4] {
        match self.model {
            HamiltonianModelKind::HenonHeiles => [-0.62, 0.88, -0.62, 0.62],
            HamiltonianModelKind::DoublePendulum => [-PI, PI, -8.5, 8.5],
            HamiltonianModelKind::FigureEightThreeBody => [-1.25, 1.25, -1.25, 1.25],
        }
    }

    /// Clamps `dt`, `energy`, `steps_per_frame`, and `cursor_qp` into valid finite bounds.
    pub fn clamp_model_parameters(&mut self) {
        if !self.dt.is_finite() {
            self.dt = 0.015;
        }
        self.dt = self.dt.clamp(0.0005, 0.1);
        let (e_min, e_max, e_def) = self.energy_bounds();
        if !self.energy.is_finite() {
            self.energy = e_def;
        }
        self.energy = self.energy.clamp(e_min, e_max);
        self.steps_per_frame = self.steps_per_frame.clamp(1, 128);
        let [q_min, q_max, p_min, p_max] = self.poincare_bounds();
        let q = if self.cursor_qp[0].is_finite() {
            self.cursor_qp[0].clamp(q_min, q_max)
        } else {
            0.0
        };
        let p = if self.cursor_qp[1].is_finite() {
            self.cursor_qp[1].clamp(p_min, p_max)
        } else {
            0.0
        };
        self.cursor_qp = [q, p];
    }

    /// Checks that all active orbit state buffers remain finite.
    #[must_use]
    pub fn orbits_are_finite(&self) -> bool {
        self.orbits.iter().all(|slot| match self.model {
            HamiltonianModelKind::FigureEightThreeBody => slot.state12.is_finite(),
            _ => slot.state4.is_finite(),
        })
    }

    /// Switches to `new_model`, resetting default energy, solver, and showcase orbits.
    pub fn switch_model(&mut self, new_model: HamiltonianModelKind) {
        self.model = new_model;
        let (_, _, e_def) = self.energy_bounds();
        self.energy = e_def;
        self.solver = match new_model {
            HamiltonianModelKind::HenonHeiles => SymplecticSolverChoice::Yoshida4,
            HamiltonianModelKind::DoublePendulum => SymplecticSolverChoice::GaussLegendre4,
            HamiltonianModelKind::FigureEightThreeBody => SymplecticSolverChoice::Yoshida4,
        };
        self.dt = match new_model {
            HamiltonianModelKind::HenonHeiles => 0.02,
            HamiltonianModelKind::DoublePendulum => 0.02,
            HamiltonianModelKind::FigureEightThreeBody => 0.01,
        };
        self.cursor_qp = [0.0, 0.0];
        self.seed_default_showcase();
        self.state_rev = self.state_rev.wrapping_add(1);
    }

    /// Populates curated default showcase orbits for the active Hamiltonian model.
    pub fn seed_default_showcase(&mut self) {
        self.clear_orbits();
        let seeds: &[[f64; 2]] = match self.model {
            HamiltonianModelKind::HenonHeiles => &[
                [0.10, 0.0],
                [0.30, 0.0],
                [-0.18, 0.10],
                [0.0, 0.30],
                [-0.27, 0.0],
            ],
            HamiltonianModelKind::DoublePendulum => &[[0.25, 0.0], [1.15, 0.0], [-0.95, 1.2]],
            HamiltonianModelKind::FigureEightThreeBody => &[[0.0, 0.0], [0.035, -0.02]],
        };
        for &qp in seeds {
            let _ = self.seed_orbit_no_warmup(qp);
        }
        self.step_batch(WARMUP_STEPS_ON_SEED);
    }

    /// Clears all seeded orbits and resets the `RungeKutta4` comparator.
    pub fn clear_orbits(&mut self) {
        self.orbits.clear();
        self.ghost = GhostComparator::default();
        self.sim_time = 0.0;
        self.poincare4 = match self.model {
            HamiltonianModelKind::DoublePendulum => PoincareSection::<4>::double_pendulum(),
            _ => PoincareSection::<4>::henon_heiles(),
        };
        self.poincare12 = PoincareSection::<12>::figure_eight();
        self.state_rev = self.state_rev.wrapping_add(1);
    }

    /// Reconstructs and seeds a canonical orbit at `(q, p)` with an immediate warm-up burst.
    pub fn seed_orbit_at(&mut self, qp: [f64; 2]) -> Result<usize, &'static str> {
        let idx = self.seed_orbit_no_warmup(qp)?;
        self.step_batch(WARMUP_STEPS_ON_SEED);
        self.state_rev = self.state_rev.wrapping_add(1);
        Ok(idx)
    }

    pub(crate) fn seed_orbit_no_warmup(&mut self, qp: [f64; 2]) -> Result<usize, &'static str> {
        if !qp[0].is_finite() || !qp[1].is_finite() {
            return Err("Seed coordinates must be finite.");
        }
        let mut slot = self.construct_orbit_slot(qp, self.orbits.len())?;
        if self.model != HamiltonianModelKind::FigureEightThreeBody {
            slot.section_points.push(qp);
        } else {
            slot.section_points
                .push([slot.state12.0[0], slot.state12.0[6]]);
        }
        if self.orbits.len() >= MAX_ORBITS {
            self.orbits.remove(0);
            for (idx, existing) in self.orbits.iter_mut().enumerate() {
                existing.color_index = idx;
            }
        }
        slot.color_index = self.orbits.len();
        self.init_ghost_from_slot(&slot);
        self.orbits.push(slot);
        Ok(self.orbits.len() - 1)
    }

    fn construct_orbit_slot(
        &self,
        qp: [f64; 2],
        color_index: usize,
    ) -> Result<OrbitSlot, &'static str> {
        let mut state4 = ArrayState::<4>::zeros();
        let mut deriv4 = ArrayState::<4>::zeros();
        let mut state12 = ArrayState::<12>::zeros();
        let mut deriv12 = ArrayState::<12>::zeros();
        let initial_energy = match self.model {
            HamiltonianModelKind::HenonHeiles => {
                state4 = self
                    .henon
                    .seed_on_poincare_section(self.energy, qp[0], qp[1])
                    .ok_or("Point lies outside the Hénon-Heiles Hill energy shell.")?;
                self.henon.derivative_in_place(0.0, &state4, &mut deriv4);
                self.henon.hamiltonian(&state4)
            }
            HamiltonianModelKind::DoublePendulum => {
                state4 = self
                    .pendulum
                    .seed_on_poincare_section(self.energy, qp[0], qp[1])
                    .ok_or("Point lies outside the Double Pendulum energy shell.")?;
                self.pendulum.derivative_in_place(0.0, &state4, &mut deriv4);
                self.pendulum.hamiltonian(&state4)
            }
            HamiltonianModelKind::FigureEightThreeBody => {
                state12 = FigureEightThreeBody::figure_eight_initial_state();
                let dy = qp[0].clamp(-0.45, 0.45);
                let dpy = qp[1].clamp(-0.45, 0.45);
                state12.0[1] += dy;
                state12.0[3] -= dy;
                state12.0[7] += dpy;
                state12.0[9] -= dpy;
                self.three_body
                    .derivative_in_place(0.0, &state12, &mut deriv12);
                self.three_body.hamiltonian(&state12)
            }
        };
        Ok(OrbitSlot {
            seed_qp: qp,
            color_index,
            initial_energy,
            last_rel_err_pct: 0.0,
            prev4: state4,
            state4,
            prev_deriv4: deriv4,
            deriv4,
            prev12: state12,
            state12,
            prev_deriv12: deriv12,
            deriv12,
            section_points: Vec::with_capacity(256),
            section_write_idx: 0,
            trail: Vec::with_capacity(MAX_TRAIL_POINTS),
        })
    }

    /// Advances the ensemble within the 60 FPS [`FRAME_BUDGET_MS`] wall-clock guard.
    pub fn step_frame_with_budget(&mut self) {
        let start = PlatformInstant::now();
        let target = self.steps_per_frame.clamp(1, 128);
        for _ in 0..target {
            self.step_once();
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            if elapsed_ms >= FRAME_BUDGET_MS {
                self.last_frame_ms = elapsed_ms;
                return;
            }
        }
        self.last_frame_ms = start.elapsed().as_secs_f64() * 1000.0;
    }

    /// Advances all seeded orbits and the `RungeKutta4` comparator by `count` steps.
    pub fn step_batch(&mut self, count: usize) {
        let bounded = count.min(4_096);
        for _ in 0..bounded {
            self.step_once();
        }
    }

    fn step_once(&mut self) {
        if self.orbits.is_empty() {
            return;
        }
        let dt = self.dt.clamp(0.0005, 0.1);
        let t = self.sim_time;
        match self.model {
            HamiltonianModelKind::HenonHeiles => self.step_henon_once(t, dt),
            HamiltonianModelKind::DoublePendulum => self.step_pendulum_once(t, dt),
            HamiltonianModelKind::FigureEightThreeBody => self.step_three_body_once(t, dt),
        }
        self.sim_time += dt;
    }

    fn step_henon_once(&mut self, t: f64, dt: f64) {
        for slot in &mut self.orbits {
            slot.prev4.copy_from(&slot.state4);
            slot.prev_deriv4.copy_from(&slot.deriv4);
            match self.solver.to_method() {
                SymplecticMethod::StormerVerlet => {
                    self.verlet2.step_separable(&self.henon, &mut slot.state4, dt);
                }
                SymplecticMethod::Yoshida4 => {
                    self.yoshida2.step_separable(&self.henon, &mut slot.state4, dt);
                }
                SymplecticMethod::GaussLegendre4 => {
                    self.gl4_4.step(&self.henon, t, &mut slot.state4, dt);
                }
            }
            if !slot.state4.is_finite() {
                slot.state4.copy_from(&slot.prev4);
                continue;
            }
            self.henon.derivative_in_place(t + dt, &slot.state4, &mut slot.deriv4);
            if let Some(hit) = self.poincare4.detect_crossing(
                t,
                dt,
                &slot.prev4,
                &slot.state4,
                &slot.prev_deriv4,
                &slot.deriv4,
            ) {
                push_capped_ring(
                    &mut slot.section_points,
                    &mut slot.section_write_idx,
                    MAX_SECTION_POINTS,
                    [hit.q_sec, hit.p_sec],
                );
            }
            slot.last_rel_err_pct =
                self.henon.relative_energy_error(&slot.state4, slot.initial_energy) * 100.0;
            push_sliding_trail(
                &mut slot.trail,
                MAX_TRAIL_POINTS,
                [slot.state4.0[0], slot.state4.0[1]],
            );
        }
        self.step_ghost_4d_henon(t, dt);
    }

    fn step_pendulum_once(&mut self, t: f64, dt: f64) {
        for slot in &mut self.orbits {
            slot.prev4.copy_from(&slot.state4);
            slot.prev_deriv4.copy_from(&slot.deriv4);
            self.gl4_4.step(&self.pendulum, t, &mut slot.state4, dt);
            self.pendulum
                .derivative_in_place(t + dt, &slot.state4, &mut slot.deriv4);
            if let Some(hit) = self.poincare4.detect_crossing(
                t,
                dt,
                &slot.prev4,
                &slot.state4,
                &slot.prev_deriv4,
                &slot.deriv4,
            ) {
                push_capped_ring(
                    &mut slot.section_points,
                    &mut slot.section_write_idx,
                    MAX_SECTION_POINTS,
                    [hit.q_sec, hit.p_sec],
                );
            }
            slot.last_rel_err_pct = self
                .pendulum
                .relative_energy_error(&slot.state4, slot.initial_energy)
                * 100.0;
            let (_, bob2) = self.pendulum.cartesian_positions(&slot.state4);
            push_sliding_trail(&mut slot.trail, MAX_TRAIL_POINTS, bob2);
        }
        self.step_ghost_4d_pendulum(t, dt);
    }

    fn step_three_body_once(&mut self, t: f64, dt: f64) {
        for slot in &mut self.orbits {
            slot.prev12.copy_from(&slot.state12);
            slot.prev_deriv12.copy_from(&slot.deriv12);
            match self.solver {
                SymplecticSolverChoice::StormerVerlet => {
                    self.verlet6
                        .step_separable(&self.three_body, &mut slot.state12, dt);
                }
                SymplecticSolverChoice::Yoshida4 => {
                    self.yoshida6
                        .step_separable(&self.three_body, &mut slot.state12, dt);
                }
                SymplecticSolverChoice::GaussLegendre4 => {
                    self.gl4_12
                        .step(&self.three_body, t, &mut slot.state12, dt);
                }
            }
            self.three_body
                .derivative_in_place(t + dt, &slot.state12, &mut slot.deriv12);
            if let Some(hit) = self.poincare12.detect_crossing(
                t,
                dt,
                &slot.prev12,
                &slot.state12,
                &slot.prev_deriv12,
                &slot.deriv12,
            ) {
                push_capped_ring(
                    &mut slot.section_points,
                    &mut slot.section_write_idx,
                    MAX_SECTION_POINTS,
                    [hit.q_sec, hit.p_sec],
                );
            }
            slot.last_rel_err_pct = self
                .three_body
                .relative_energy_error(&slot.state12, slot.initial_energy)
                * 100.0;
            push_sliding_trail(
                &mut slot.trail,
                MAX_TRAIL_POINTS,
                [slot.state12.0[0], slot.state12.0[1]],
            );
        }
        self.step_ghost_12d_three_body(t, dt);
    }
}
