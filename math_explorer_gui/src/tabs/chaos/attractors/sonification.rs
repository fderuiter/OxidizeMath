//! Zero-allocation Lorenz strange-attractor stereo pan and two-operator FM sonification
//! for [`crate::tabs::chaos::attractors::AttractorPlotter`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::AttractorPlotter;
use crate::audio::SonificationFrame;
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState, LorenzSystem};

#[inline]
pub(super) fn lorenz_sonification_frame(sys: &LorenzSystem, diverged: bool) -> SonificationFrame {
    if diverged {
        return SonificationFrame::silent();
    }
    let v = sys.state.vec;
    if !v.x.is_finite()
        || !v.y.is_finite()
        || !v.z.is_finite()
        || !sys.sigma.is_finite()
        || !sys.rho.is_finite()
        || !sys.beta.is_finite()
    {
        return SonificationFrame::silent();
    }

    let pan = (v.x / 22.0).clamp(-1.0, 1.0) as f32;
    let z_norm = (v.z / 50.0).clamp(0.0, 1.5) as f32;
    let carrier_hz = 130.0 + z_norm * 360.0;
    let y_abs = (v.y.abs() / 28.0).clamp(0.0, 1.5) as f32;
    let shear = ((v.x - v.y).abs() / 30.0).clamp(0.0, 1.5) as f32;
    let fm_modulator_hz = carrier_hz * (0.5 + 1.5 * y_abs);
    let fm_index_hz = 18.0 + 260.0 * shear;
    let radius = ((v.x * v.x + v.y * v.y + v.z * v.z).sqrt() / 55.0).clamp(0.0, 1.2) as f32;
    let filter_cutoff_hz = 550.0 + 4_200.0 * radius;
    let amplitude = (0.32 + 0.45 * radius).clamp(0.0, 0.88);

    SonificationFrame {
        active: true,
        carrier_hz,
        amplitude,
        pan,
        fm_modulator_hz,
        fm_index_hz,
        filter_cutoff_hz,
        filter_q: 1.8,
        ..SonificationFrame::silent()
    }
}

impl AttractorPlotter {
    pub(super) fn render_wav_frames(&self, frame_count: usize) -> Vec<SonificationFrame> {
        let count = frame_count.max(1);
        if self.diverged() || !self.is_finite_state() {
            return vec![SonificationFrame::silent(); count];
        }
        let v = self.system.state.vec;
        let mut sys = LorenzBuilder::new()
            .sigma(self.system.sigma)
            .rho(self.system.rho)
            .beta(self.system.beta)
            .build(LorenzState::new(v.x, v.y, v.z));
        let dt = self.dt.clamp(0.001, 0.05);
        let substeps = self.simulation_speed.clamp(1, super::MAX_SIMULATION_SPEED);
        let custom_job = (self.mode == super::AttractorMode::Custom)
            .then(|| super::CompiledFlowJob::from_bank(&self.flow_bank));
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let mut bad = false;
            for _ in 0..substeps {
                if let Some(ref job) = custom_job {
                    sys.state.vec = job.rk4_step(sys.state.vec, dt);
                } else {
                    sys.step(dt);
                }
                let sv = sys.state.vec;
                if !sv.x.is_finite() || !sv.y.is_finite() || !sv.z.is_finite() {
                    bad = true;
                    break;
                }
            }
            out.push(lorenz_sonification_frame(&sys, bad));
        }
        out
    }
}
