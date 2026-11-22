//! Zero-allocation NMR free-induction-decay (`T1`/`T2`) bell synthesis for [`super::tool::MriTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::tool::MriTool;
use crate::audio::SonificationFrame;
use math_explorer::physics::mri::bloch::BlochSimulator;
use math_explorer::pure_math::analysis::ode::{RungeKutta4, TimeStepper};

/// Maps Bloch magnetization `(Mx, My, Mz)`, static/RF field `B`, and `T1`/`T2` relaxation times
/// into a zero-allocation NMR free-induction-decay [`SonificationFrame`].
#[inline]
#[must_use]
pub fn bloch_fid_sonification_frame(sim: &BlochSimulator, faulted: bool) -> SonificationFrame {
    if faulted {
        return SonificationFrame::silent();
    }
    let m = sim.magnetization;
    let b = sim.b_field;
    if !m.x.is_finite()
        || !m.y.is_finite()
        || !m.z.is_finite()
        || !b.x.is_finite()
        || !b.y.is_finite()
        || !b.z.is_finite()
        || !sim.t1.is_finite()
        || !sim.t2.is_finite()
        || !sim.m0.is_finite()
        || sim.m0 <= 0.0
    {
        return SonificationFrame::silent();
    }

    let m0 = sim.m0;
    let m_xy = (m.x.hypot(m.y) / m0).clamp(0.0, 1.5) as f32;
    let mz_norm = (((m.z / m0) + 1.0) * 0.5).clamp(0.0, 1.0) as f32;
    let b_mag = b.norm().clamp(0.0, 75.0) as f32;

    let carrier_hz = (180.0 + b_mag * 28.0).clamp(110.0, 2_200.0);
    let bell_decay_rate = ((1.0 / sim.t2.clamp(0.01, 2.0)) as f32).clamp(0.5, 25.0);
    let t1_factor = ((1.0 / sim.t1.clamp(0.1, 5.0)) as f32).clamp(0.2, 10.0);
    let amplitude = (0.14 + 0.80 * m_xy).clamp(0.0, 0.95);
    let pan = (m.x / m0).clamp(-1.0, 1.0) as f32;
    let filter_cutoff_hz = (600.0 + 4_200.0 * m_xy + 1_100.0 * mz_norm).clamp(350.0, 6_800.0);

    SonificationFrame {
        active: true,
        carrier_hz,
        amplitude,
        pan,
        fm_modulator_hz: carrier_hz * 2.01,
        fm_index_hz: 6.0 + 24.0 * m_xy * (t1_factor * 0.25).clamp(0.2, 2.0),
        filter_cutoff_hz,
        filter_q: 2.2,
        bell_decay_rate,
        ..SonificationFrame::silent()
    }
}

impl MriTool {
    pub(super) fn render_wav_frames(&self, frame_count: usize) -> Vec<SonificationFrame> {
        let count = frame_count.max(1);
        if !crate::framework::InteractiveTool::has_finite_parameters(self) {
            return vec![SonificationFrame::silent(); count];
        }
        let mut local_sim = BlochSimulator::new(self.simulator.magnetization, self.simulator.m0);
        local_sim.set_relaxation(self.simulator.t1, self.simulator.t2);
        local_sim.set_b_field(self.simulator.b_field);
        let mut local_solver = RungeKutta4::new(&local_sim.magnetization);
        let dt = 0.01 * self.time_scale.clamp(1.0, 20.0);
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            TimeStepper::step_with(&mut local_sim, &mut local_solver, dt);
            out.push(bloch_fid_sonification_frame(&local_sim, false));
        }
        out
    }
}
