//! Zero-allocation NMR free-induction-decay (`T1`/`T2`) and RF pulse sonification helpers for [`crate::tabs::mri::tool::MriTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::pulse::MRI_PULSE_SAMPLES;
use super::tool::MriTool;
use crate::audio::mappings::{mri_bloch_frame, mri_pulse_playhead_frame};
use crate::audio::SonificationFrame;
use math_explorer::physics::mri::bloch::{rotating_frame_effective_field, BlochSimulator};
use math_explorer::physics::mri::proton;
use math_explorer::pure_math::analysis::ode::{RungeKutta4, TimeStepper};

impl MriTool {
    pub(super) fn render_wav_frames(&self, frame_count: usize) -> Vec<SonificationFrame> {
        let count = frame_count.max(1);
        if !crate::framework::InteractiveTool::has_finite_parameters(self) {
            return vec![SonificationFrame::silent(); count];
        }
        if !self.pulse_config.mode.is_continuous() {
            let mut out = Vec::with_capacity(count);
            for idx in 0..count {
                let sample_idx = (idx * MRI_PULSE_SAMPLES / count).min(MRI_PULSE_SAMPLES - 1);
                let t_s = self.pulse_workspace.times[sample_idx];
                let m_sample = self.pulse_workspace.trajectory[sample_idx];
                let disp = self.pulse_workspace.dispersion[sample_idx];
                out.push(mri_pulse_playhead_frame(
                    self.pulse_config.mode,
                    t_s,
                    m_sample,
                    disp,
                    self.simulator.m0,
                    self.simulator.b_field.z,
                    &self.pulse_config,
                ));
            }
            return out;
        }

        let b_lab = self.simulator.b_field;
        let omega_rf = proton::larmor_frequency(b_lab.z);
        let b1_amp = b_lab.x.hypot(b_lab.y);
        let phase_rad = b_lab.y.atan2(b_lab.x);
        let b_rot = rotating_frame_effective_field(b_lab.z, omega_rf, b1_amp, phase_rad);

        let mut local_sim = BlochSimulator::new(self.simulator.magnetization, self.simulator.m0);
        local_sim.set_relaxation(self.simulator.t1, self.simulator.t2);
        local_sim.set_b_field(b_rot);
        let mut local_solver = RungeKutta4::new(&local_sim.magnetization);
        let dt = 0.002 * self.time_scale.clamp(1.0, 20.0);
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(mri_bloch_frame(
                local_sim.magnetization,
                local_sim.m0,
                b_lab,
                self.pulse_config.detuning_hz,
            ));
            TimeStepper::step_with(&mut local_sim, &mut local_solver, dt);
        }
        out
    }
}
