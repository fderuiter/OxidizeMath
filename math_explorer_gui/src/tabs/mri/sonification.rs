//! Zero-allocation NMR free-induction-decay (`T1`/`T2`) bell synthesis for [`crate::tabs::mri::tool::MriTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::tool::MriTool;
use crate::audio::{SonificationFrame, VoiceSpec, WaveformPreset};
use math_explorer::physics::mri::bloch::BlochSimulator;
use math_explorer::pure_math::analysis::ode::{RungeKutta4, TimeStepper};

const FID_BELL_RATIOS: [f32; 3] = [1.0, 2.756, 5.404];
const FID_BELL_WEIGHTS: [f32; 3] = [0.65, 0.23, 0.12];

/// Maps Bloch magnetization `(Mx, My, Mz)`, static/RF field `B`, and `T1`/`T2` relaxation times
/// into a zero-allocation polyphonic NMR free-induction-decay [`SonificationFrame`].
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

    let carrier_hz = (180.0 + b_mag * 28.0).clamp(110.0, 1_400.0);
    let t2_damping = ((1.0 / sim.t2.clamp(0.01, 2.0)) as f32 * 0.04).clamp(0.02, 0.65);
    let t1_factor = ((1.0 / sim.t1.clamp(0.1, 5.0)) as f32).clamp(0.2, 10.0);
    let pan = (m.x / m0).clamp(-1.0, 1.0) as f32;
    let filter_cutoff_hz = (600.0 + 4_200.0 * m_xy + 1_100.0 * mz_norm).clamp(350.0, 6_800.0);
    let fm_index = (0.15 + 1.6 * m_xy * (t1_factor * 0.25).clamp(0.2, 2.0)).clamp(0.0, 4.0);

    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::FourierWarm;
    frame.active_voices = 3;
    frame.master_gain = (0.18 + 0.72 * m_xy).clamp(0.0, 0.90);

    for (idx, (&ratio, &weight)) in FID_BELL_RATIOS
        .iter()
        .zip(FID_BELL_WEIGHTS.iter())
        .enumerate()
    {
        let partial_decay = 1.0 / (1.0 + (idx as f32) * t2_damping);
        frame.voices[idx] = VoiceSpec {
            freq_hz: (carrier_hz * ratio).clamp(80.0, 7_800.0),
            gain: (weight * partial_decay * (0.28 + 0.72 * m_xy)).clamp(0.0, 0.95),
            pan: (pan * (1.0 - 0.2 * idx as f32)).clamp(-1.0, 1.0),
            fm_mod_hz: (carrier_hz * 2.01).clamp(40.0, 3_800.0),
            fm_index: if idx == 0 { fm_index } else { fm_index * 0.35 },
            filter_cutoff_hz,
            transient_impulse: 0.0,
        };
    }
    frame.sanitize()
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
