//! Counterfactual shadow twin and Hahn `180°` spin-echo refocusing for [`super::tool::MriTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::runner::MriRunner;
use crate::framework::{
    l2_distance_3d, mri_transverse_echo_gain, sanitize_perturbation_delta, DivergenceSnapshot,
    DomainInterventionDelta, SeparationTracker,
};
use math_explorer::physics::mri::bloch::{rotating_frame_effective_field, BlochSimulator};
use math_explorer::physics::mri::proton;
use math_explorer::pure_math::analysis::ode::{RungeKutta4, TimeStepper};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Fixed Bloch RK4 time step (`1 ms`) used by `MriRunner`.
pub const MRI_STEP_DT: f64 = 0.001;

/// Number of off-resonance spin isochromats in the stack-allocated Hahn spin-echo ensemble.
const ISOCHROMAT_COUNT: usize = 9;

/// Reversible static field inhomogeneity spread `ΔB_z` (Tesla) corresponding to `~18 rad/s` Larmor spread.
const SPIN_ECHO_BZ_SPREAD_TESLA: f64 = 6.8e-8;

/// Compact serializable fork descriptor for `MriTool` (`< 300` bytes JSON).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MriForkDescriptor {
    /// Step index `k > 0` (`t_k = k * 0.001` s) where the fork occurred.
    pub fork_step: usize,
    /// Number of lockstep `1 ms` steps evolved after `fork_step`.
    #[serde(default)]
    pub post_fork_steps: usize,
    /// Frozen baseline `T1` on `Branch A`.
    pub baseline_t1: f64,
    /// Frozen baseline `T2` on `Branch A`.
    pub baseline_t2: f64,
    /// Frozen baseline `Bx` on `Branch A`.
    pub baseline_bx: f64,
    /// Frozen baseline `By` on `Branch A`.
    pub baseline_by: f64,
    /// Frozen baseline `Bz` on `Branch A`.
    pub baseline_bz: f64,
    /// State perturbation `δx` applied to `Mx(t_k)` on `Branch B`.
    pub delta_x: f64,
    /// Whether a `180°` Hahn spin-echo refocusing pulse was applied on `Branch B` at `t_k`.
    #[serde(default)]
    pub refocus_180: bool,
}

/// Stack-allocated reversible off-resonance spin ensemble for zero-allocation `180°` spin-echo simulation.
#[derive(Clone, Debug)]
pub struct MriSpinEchoEnsemble {
    /// Unperturbed dephasing isochromats on `Branch A`.
    pub spins_a: [Vector3<f64>; ISOCHROMAT_COUNT],
    /// `180°`-refocused isochromats on `Branch B`.
    pub spins_b: [Vector3<f64>; ISOCHROMAT_COUNT],
    /// Per-isochromat static `δB_z` offsets in Tesla.
    pub bz_offsets: [f64; ISOCHROMAT_COUNT],
}

impl MriSpinEchoEnsemble {
    /// Seeds a 9-isochromat fan around `base_m` at echo time `tau = fork_time` and applies a `180°` pulse to `spins_b`.
    #[must_use]
    pub fn from_mid_decay(base_m: Vector3<f64>, fork_time: f64) -> Self {
        let m_xy = base_m.x.hypot(base_m.y);
        let base_phase = base_m.y.atan2(base_m.x);
        let gamma = proton::GYROMAGNETIC_RATIO;
        let tau = fork_time.clamp(0.02, 1.0);

        let mut bz_offsets = [0.0; ISOCHROMAT_COUNT];
        let half = (ISOCHROMAT_COUNT / 2) as isize;
        for (idx, slot) in bz_offsets.iter_mut().enumerate() {
            let rel = (idx as isize - half) as f64 / (half as f64);
            *slot = rel * SPIN_ECHO_BZ_SPREAD_TESLA;
        }
        let env = m_xy.clamp(0.05, 1.0);

        let mut spins_a = [Vector3::zeros(); ISOCHROMAT_COUNT];
        let mut spins_b = [Vector3::zeros(); ISOCHROMAT_COUNT];
        for idx in 0..ISOCHROMAT_COUNT {
            // Under dM/dt = M x (gamma * B), Larmor phase evolves as dphi/dt = -gamma * B_z.
            // At t = tau, off-resonance spin idx has accumulated relative phase -dphi;
            // the 180° CPMG pulse reflects relative phase (-dphi -> +dphi) so spins_b refocus at 2*tau.
            let dphi = gamma * bz_offsets[idx] * tau;
            let phi_a = base_phase - dphi;
            let phi_b = base_phase + dphi;
            spins_a[idx] = Vector3::new(env * phi_a.cos(), env * phi_a.sin(), base_m.z);
            spins_b[idx] = Vector3::new(env * phi_b.cos(), env * phi_b.sin(), -base_m.z);
        }
        Self {
            spins_a,
            spins_b,
            bz_offsets,
        }
    }

    /// Mean macroscopic magnetization across an isochromat array.
    #[must_use]
    pub fn mean_magnetization(spins: &[Vector3<f64>; ISOCHROMAT_COUNT]) -> Vector3<f64> {
        let mut sum = Vector3::zeros();
        for s in spins {
            sum += *s;
        }
        sum / (ISOCHROMAT_COUNT as f64)
    }
}

/// Pre-allocated counterfactual shadow twin state for `MriRunner` and `MriTool`.
pub struct MriTwinState {
    /// Frozen baseline `BlochSimulator` (`Branch A`).
    pub simulator_a: BlochSimulator,
    /// Pre-allocated RK4 scratch state for `Branch A` (zero heap allocation in `step()`).
    pub solver_a: RungeKutta4<Vector3<f64>>,
    /// Pre-allocated trajectory `(t, M_A)` history for `Branch A`.
    pub history_a: VecDeque<(f64, Vector3<f64>)>,
    /// Step index `k` where the fork was anchored.
    pub fork_step: usize,
    /// Magnetization `M(t_k)` at the fork point for plot callouts.
    pub fork_m: Vector3<f64>,
    /// State perturbation `δx` applied to `Mx(t_k)` on `Branch B`.
    pub delta_x: f64,
    /// Optional stack-allocated Hahn spin-echo ensemble when a `180°` pulse is active.
    pub spin_echo: Option<MriSpinEchoEnsemble>,
    /// Live Euclidean separation and logarithmic slope tracker.
    pub tracker: SeparationTracker,
}

impl Clone for MriTwinState {
    fn clone(&self) -> Self {
        let mut simulator_a =
            BlochSimulator::new(self.simulator_a.magnetization, self.simulator_a.m0);
        simulator_a.set_relaxation(self.simulator_a.t1, self.simulator_a.t2);
        simulator_a.set_b_field(self.simulator_a.b_field);
        Self {
            solver_a: RungeKutta4::new(&simulator_a.magnetization),
            simulator_a,
            history_a: self.history_a.clone(),
            fork_step: self.fork_step,
            fork_m: self.fork_m,
            delta_x: self.delta_x,
            spin_echo: self.spin_echo.clone(),
            tracker: self.tracker,
        }
    }
}

impl MriTwinState {
    /// Clones `(step_count, current_time, simulator, history)` into `Branch A` and perturbs `simulator_b`.
    pub fn fork_from(
        fork_step: usize,
        current_time: f64,
        simulator_b: &mut BlochSimulator,
        history_b: &VecDeque<(f64, Vector3<f64>)>,
        max_history: usize,
        delta_x: f64,
        refocus_180: bool,
    ) -> Self {
        let clean_delta = sanitize_perturbation_delta(delta_x);
        let fork_m = simulator_b.magnetization;
        let mut simulator_a = BlochSimulator::new(fork_m, simulator_b.m0);
        simulator_a.set_relaxation(simulator_b.t1, simulator_b.t2);
        simulator_a.set_b_field(simulator_b.b_field);

        let mut history_a = VecDeque::with_capacity(max_history);
        for &item in history_b {
            if history_a.len() >= max_history {
                history_a.pop_front();
            }
            history_a.push_back(item);
        }

        simulator_b.magnetization.x += clean_delta;
        let spin_echo = refocus_180.then(|| {
            let ens = MriSpinEchoEnsemble::from_mid_decay(simulator_b.magnetization, current_time);
            simulator_a.magnetization = MriSpinEchoEnsemble::mean_magnetization(&ens.spins_a);
            simulator_b.magnetization = MriSpinEchoEnsemble::mean_magnetization(&ens.spins_b);
            ens
        });

        let ma = simulator_a.magnetization;
        let mb = simulator_b.magnetization;
        let init_l2 = l2_distance_3d([ma.x, ma.y, ma.z], [mb.x, mb.y, mb.z]);

        Self {
            simulator_a,
            solver_a: RungeKutta4::new(&ma),
            history_a,
            fork_step,
            fork_m,
            delta_x: clean_delta,
            spin_echo,
            tracker: SeparationTracker::new(current_time, init_l2),
        }
    }

    /// Applies a `180°` refocusing pulse at the current state if already forked, or marks spin-echo active.
    pub fn apply_refocus_180(&mut self, current_time: f64, simulator_b: &mut BlochSimulator) {
        let ens = MriSpinEchoEnsemble::from_mid_decay(simulator_b.magnetization, current_time);
        self.simulator_a.magnetization = MriSpinEchoEnsemble::mean_magnetization(&ens.spins_a);
        simulator_b.magnetization = MriSpinEchoEnsemble::mean_magnetization(&ens.spins_b);
        self.spin_echo = Some(ens);
        let ma = self.simulator_a.magnetization;
        let mb = simulator_b.magnetization;
        self.tracker.observe(
            current_time,
            l2_distance_3d([ma.x, ma.y, ma.z], [mb.x, mb.y, mb.z]),
        );
    }

    /// Advances `Branch A` by one `dt` increment in the rotating frame with zero heap allocations.
    pub fn step_branch_a(&mut self, next_time: f64, dt: f64, max_history: usize) {
        if let Some(ref mut ens) = self.spin_echo {
            for idx in 0..ISOCHROMAT_COUNT {
                self.simulator_a.magnetization = ens.spins_a[idx];
                step_bloch_rotating_frame(
                    &mut self.simulator_a,
                    &mut self.solver_a,
                    dt,
                    ens.bz_offsets[idx],
                );
                ens.spins_a[idx] = self.simulator_a.magnetization;
            }
            self.simulator_a.magnetization = MriSpinEchoEnsemble::mean_magnetization(&ens.spins_a);
        } else {
            step_bloch_rotating_frame(&mut self.simulator_a, &mut self.solver_a, dt, 0.0);
        }
        let m = self.simulator_a.magnetization;
        if m.x.is_finite() && m.y.is_finite() && m.z.is_finite() {
            if self.history_a.len() >= max_history {
                self.history_a.pop_front();
            }
            self.history_a.push_back((next_time, m));
        }
    }

    /// Advances `Branch B` (`simulator_b`) by one `dt` increment (handling spin-echo ensemble when active).
    pub fn step_branch_b_spin_echo(
        &mut self,
        simulator_b: &mut BlochSimulator,
        solver_b: &mut RungeKutta4<Vector3<f64>>,
        dt: f64,
    ) -> bool {
        let Some(ref mut ens) = self.spin_echo else {
            return false;
        };
        for idx in 0..ISOCHROMAT_COUNT {
            simulator_b.magnetization = ens.spins_b[idx];
            step_bloch_rotating_frame(simulator_b, solver_b, dt, ens.bz_offsets[idx]);
            ens.spins_b[idx] = simulator_b.magnetization;
        }
        simulator_b.magnetization = MriSpinEchoEnsemble::mean_magnetization(&ens.spins_b);
        true
    }

    /// Records separation after a lockstep tick.
    pub fn observe_lockstep(&mut self, current_time: f64, mb: Vector3<f64>) {
        let ma = self.simulator_a.magnetization;
        let l2 = l2_distance_3d([ma.x, ma.y, ma.z], [mb.x, mb.y, mb.z]);
        self.tracker.observe(current_time, l2);
    }

    /// Produces a [`DivergenceSnapshot`] including `ΔM_xy` transverse echo gain.
    #[must_use]
    pub fn snapshot(&self, current_time: f64, mb: Vector3<f64>) -> DivergenceSnapshot {
        let ma = self.simulator_a.magnetization;
        let echo_gain = mri_transverse_echo_gain([ma.x, ma.y, ma.z], [mb.x, mb.y, mb.z]);
        DivergenceSnapshot {
            fork_step: self.fork_step,
            fork_time: self.tracker.fork_time(),
            current_time,
            delta_x: self.delta_x,
            l2_separation: self.tracker.l2_separation(),
            log_separation_slope: self.tracker.log_separation_slope(),
            peak_separation: self.tracker.peak_separation(),
            primary_delta: Some(DomainInterventionDelta::new(
                "ΔM_xy",
                format!("{echo_gain:+.4}"),
                echo_gain,
                "Transverse echo recovery gain ||M_xy,B|| - ||M_xy,A|| relative to unperturbed decay",
            )),
            secondary_delta: None,
        }
    }

    /// Builds a compact descriptor for preset/URL serialization.
    #[must_use]
    pub fn to_descriptor(&self, total_steps: usize) -> MriForkDescriptor {
        MriForkDescriptor {
            fork_step: self.fork_step,
            post_fork_steps: total_steps.saturating_sub(self.fork_step),
            baseline_t1: self.simulator_a.t1,
            baseline_t2: self.simulator_a.t2,
            baseline_bx: self.simulator_a.b_field.x,
            baseline_by: self.simulator_a.b_field.y,
            baseline_bz: self.simulator_a.b_field.z,
            delta_x: self.delta_x,
            refocus_180: self.spin_echo.is_some(),
        }
    }
}

impl MriRunner {
    pub(super) fn current_step_index(&self) -> usize {
        (self.current_time / MRI_STEP_DT).round().max(0.0) as usize
    }

    pub(super) fn handle_custom_command(&mut self, cmd: &str) {
        if let Some(rest) = cmd.strip_prefix("fork:") {
            let step = self.current_step_index();
            if step > 0 {
                let delta = rest.parse::<f64>().unwrap_or(0.0);
                let _ = self.sync_params_and_get_scale();
                self.twin = Some(MriTwinState::fork_from(
                    step,
                    self.current_time,
                    &mut self.simulator,
                    &self.history,
                    self.max_history,
                    delta,
                    false,
                ));
            }
        } else if cmd == "refocus_180" {
            let step = self.current_step_index();
            let _ = self.sync_params_and_get_scale();
            if let Some(ref mut twin) = self.twin {
                twin.apply_refocus_180(self.current_time, &mut self.simulator);
            } else if step > 0 {
                self.twin = Some(MriTwinState::fork_from(
                    step,
                    self.current_time,
                    &mut self.simulator,
                    &self.history,
                    self.max_history,
                    0.0,
                    true,
                ));
            }
        } else if cmd == "swap" {
            self.swap_twin_branches();
        } else if cmd == "clear_fork" {
            self.twin = None;
        }
    }

    pub(super) fn swap_twin_branches(&mut self) {
        if let Some(ref mut twin) = self.twin {
            std::mem::swap(
                &mut self.simulator.magnetization,
                &mut twin.simulator_a.magnetization,
            );
            std::mem::swap(&mut self.simulator.t1, &mut twin.simulator_a.t1);
            std::mem::swap(&mut self.simulator.t2, &mut twin.simulator_a.t2);
            std::mem::swap(&mut self.simulator.b_field, &mut twin.simulator_a.b_field);
            std::mem::swap(&mut self.history, &mut twin.history_a);
            if let Some(ref mut ens) = twin.spin_echo {
                std::mem::swap(&mut ens.spins_a, &mut ens.spins_b);
            }
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.t1 = self.simulator.t1;
            p.t2 = self.simulator.t2;
            p.b_field = self.simulator.b_field;
        }
    }
}

/// Steps `sim` by `dt` in the Larmor rotating frame of `sim.b_field.z` (plus optional `bz_offset`),
/// subdividing large off-resonance phase increments with zero heap allocations.
pub(crate) fn step_bloch_rotating_frame(
    sim: &mut BlochSimulator,
    solver: &mut RungeKutta4<Vector3<f64>>,
    dt: f64,
    bz_offset: f64,
) {
    let b_lab = sim.b_field;
    let omega_rf = proton::larmor_frequency(b_lab.z);
    let b1_amp = b_lab.x.hypot(b_lab.y);
    let phase_rad = b_lab.y.atan2(b_lab.x);
    let mut b_rot = rotating_frame_effective_field(b_lab.z, omega_rf, b1_amp, phase_rad);
    b_rot.z += bz_offset;
    let max_phase_step = proton::GYROMAGNETIC_RATIO * b_rot.norm() * dt;
    let sub_steps = if max_phase_step.is_finite() && max_phase_step > 0.25 {
        (max_phase_step / 0.25).ceil().min(64.0) as usize
    } else {
        1
    };
    let sub_dt = dt / (sub_steps as f64);
    sim.b_field = b_rot;
    for _ in 0..sub_steps {
        TimeStepper::step_with(sim, solver, sub_dt);
    }
    sim.b_field = b_lab;
}
