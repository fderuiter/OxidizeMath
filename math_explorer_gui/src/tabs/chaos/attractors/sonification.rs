//! Zero-allocation 3D chaotic attractor stereo pan and FM sonification helpers
//! for [`crate::tabs::chaos::attractors::AttractorPlotter`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{AttractorMode, AttractorPlotter, AttractorSystemKind, CompiledFlowJob};
use crate::audio::mappings::attractor_frame;
use crate::audio::SonificationFrame;
use math_explorer::physics::chaos::double_pendulum::{DoublePendulumBuilder, DoublePendulumState};
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState};
use math_explorer::physics::chaos::rossler::{RosslerBuilder, RosslerState};

impl AttractorPlotter {
    #[inline]
    pub(super) fn current_sonification_frame(&self) -> SonificationFrame {
        if self.diverged() || !self.is_finite_state() {
            return SonificationFrame::silent();
        }
        let v = self.system.state.vec;
        let deriv = match self.mode {
            AttractorMode::Lorenz => match self.system_kind {
                AttractorSystemKind::Lorenz => [
                    self.system.sigma * (v.y - v.x),
                    v.x * (self.system.rho - v.z) - v.y,
                    v.x * v.y - self.system.beta * v.z,
                ],
                AttractorSystemKind::Rossler => [
                    -v.y - v.z,
                    v.x + self.extra_params.rossler_a * v.y,
                    self.extra_params.rossler_b + v.z * (v.x - self.extra_params.rossler_c),
                ],
                AttractorSystemKind::DoublePendulum => {
                    [v.z, 0.0, -self.extra_params.pendulum_g * v.x.sin()]
                }
            },
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
        let dt = self.dt.clamp(0.001, 0.05);
        let substeps = self.simulation_speed.clamp(1, super::MAX_SIMULATION_SPEED);
        if self.mode == AttractorMode::Custom {
            return self.render_custom_wav_frames(count, dt, substeps);
        }
        match self.system_kind {
            AttractorSystemKind::Lorenz => self.render_lorenz_wav_frames(count, dt, substeps),
            AttractorSystemKind::Rossler => self.render_rossler_wav_frames(count, dt, substeps),
            AttractorSystemKind::DoublePendulum => {
                self.render_pendulum_wav_frames(count, dt, substeps)
            }
        }
    }

    fn render_custom_wav_frames(
        &self,
        count: usize,
        dt: f64,
        substeps: usize,
    ) -> Vec<SonificationFrame> {
        let job = CompiledFlowJob::from_bank(&self.flow_bank);
        let mut pos = self.system.state.vec;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let mut bad = false;
            for _ in 0..substeps {
                pos = job.rk4_step(pos, dt);
                if !pos.x.is_finite() || !pos.y.is_finite() || !pos.z.is_finite() {
                    bad = true;
                    break;
                }
            }
            let deriv = if bad {
                [0.0; 3]
            } else {
                let rhs = job.eval_rhs(pos);
                [rhs.x, rhs.y, rhs.z]
            };
            out.push(attractor_frame([pos.x, pos.y, pos.z], deriv, bad));
        }
        out
    }

    fn render_lorenz_wav_frames(
        &self,
        count: usize,
        dt: f64,
        substeps: usize,
    ) -> Vec<SonificationFrame> {
        let v = self.system.state.vec;
        let mut sys = LorenzBuilder::new()
            .sigma(self.system.sigma)
            .rho(self.system.rho)
            .beta(self.system.beta)
            .build(LorenzState::new(v.x, v.y, v.z));
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let mut bad = false;
            for _ in 0..substeps {
                sys.step(dt);
                let sv = sys.state.vec;
                if !sv.x.is_finite() || !sv.y.is_finite() || !sv.z.is_finite() {
                    bad = true;
                    break;
                }
            }
            let sv = sys.state.vec;
            let deriv = if bad {
                [0.0; 3]
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

    fn render_rossler_wav_frames(
        &self,
        count: usize,
        dt: f64,
        substeps: usize,
    ) -> Vec<SonificationFrame> {
        let v = self.system.state.vec;
        let mut sys = RosslerBuilder::new()
            .a(self.extra_params.rossler_a)
            .b(self.extra_params.rossler_b)
            .c(self.extra_params.rossler_c)
            .build(RosslerState::new(v.x, v.y, v.z));
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let mut bad = false;
            for _ in 0..substeps {
                sys.step(dt);
                let sv = sys.state.vec;
                if !sv.x.is_finite() || !sv.y.is_finite() || !sv.z.is_finite() {
                    bad = true;
                    break;
                }
            }
            let sv = sys.state.vec;
            let deriv = if bad {
                [0.0; 3]
            } else {
                [
                    -sv.y - sv.z,
                    sv.x + sys.a * sv.y,
                    sys.b + sv.z * (sv.x - sys.c),
                ]
            };
            out.push(attractor_frame([sv.x, sv.y, sv.z], deriv, bad));
        }
        out
    }

    fn render_pendulum_wav_frames(
        &self,
        count: usize,
        dt: f64,
        substeps: usize,
    ) -> Vec<SonificationFrame> {
        let p4 = self.timeline.current().map_or(
            [
                self.system.state.vec.x,
                self.system.state.vec.y,
                self.system.state.vec.z,
                0.0,
            ],
            |f| f.primary,
        );
        let mut sys = DoublePendulumBuilder::new()
            .m1(self.extra_params.pendulum_m1)
            .m2(self.extra_params.pendulum_m2)
            .l1(self.extra_params.pendulum_l1)
            .l2(self.extra_params.pendulum_l2)
            .g(self.extra_params.pendulum_g)
            .build(DoublePendulumState::new(p4[0], p4[1], p4[2], p4[3]));
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let mut bad = false;
            for _ in 0..substeps {
                sys.step(dt);
                if sys.state.state.0.iter().any(|c| !c.is_finite()) {
                    bad = true;
                    break;
                }
            }
            let sv = sys.state.phase_point_3d();
            let deriv = if bad {
                [0.0; 3]
            } else {
                [sv.z, sys.state.state.0[3], -sys.g * sv.x.sin()]
            };
            out.push(attractor_frame([sv.x, sv.y, sv.z], deriv, bad));
        }
        out
    }
}
