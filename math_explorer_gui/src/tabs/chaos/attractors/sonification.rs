//! Zero-allocation 3D chaotic attractor stereo pan and FM sonification helpers
//! for [`crate::tabs::chaos::attractors::AttractorPlotter`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{AttractorMode, AttractorPlotter, CompiledFlowJob};
use crate::audio::SonificationFrame;
use crate::audio::mappings::attractor_frame;
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState};

impl AttractorPlotter {
    #[inline]
    pub(super) fn current_sonification_frame(&self) -> SonificationFrame {
        if self.diverged() || !self.is_finite_state() {
            return SonificationFrame::silent();
        }
        let v = self.system.state.vec;
        let deriv = match self.mode {
            AttractorMode::Lorenz => [
                self.system.sigma * (v.y - v.x),
                v.x * (self.system.rho - v.z) - v.y,
                v.x * v.y - self.system.beta * v.z,
            ],
            AttractorMode::Custom => {
                let compiled = self.flow_bank.active_compiled();
                let (params, len) = self.flow_bank.inline_params();
                let p = &params[..len];
                [
                    compiled[0].eval_3d(v.x, v.y, v.z, p),
                    compiled[1].eval_3d(v.x, v.y, v.z, p),
                    compiled[2].eval_3d(v.x, v.y, v.z, p),
                ]
            }
        };
        attractor_frame([v.x, v.y, v.z], deriv, false)
    }

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
        let custom_job = (self.mode == AttractorMode::Custom)
            .then(|| CompiledFlowJob::from_bank(&self.flow_bank));
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
            let sv = sys.state.vec;
            let deriv = if bad {
                [0.0, 0.0, 0.0]
            } else if let Some(ref job) = custom_job {
                let rhs = job.eval_rhs(sv);
                [rhs.x, rhs.y, rhs.z]
            } else {
                [
                    sys.sigma * (sv.y - sv.x),
                    sv.x * (sys.rho - sv.z) - sv.y,
                    sv.x * sv.y - sys.beta * sv.z,
                ]
            };
            out.push(attractor_frame([sv.x, sv.y, sv.z], deriv, bad));
        }
        out
    }
}
