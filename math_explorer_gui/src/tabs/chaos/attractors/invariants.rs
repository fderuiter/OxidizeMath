#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::AttractorPlotter;
use crate::tabs::geometry_topology::vietoris_rips::{
    handoff::{request_open_vietoris_rips, submit_chaos_landmark_handoff, ChaosLandmarkHandoff},
    MAX_CHAOS_LANDMARKS,
};
use eframe::egui;
use math_explorer::physics::chaos::fractals::correlation_dimension;
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState};
use math_explorer::physics::chaos::metrics::lorenz_lyapunov;
use math_explorer::pure_math::analysis::ode::RungeKutta4;
use math_explorer::pure_math::statistics::tda::{farthest_point_landmarks, Point3D};
use nalgebra::Vector3;

#[derive(Default, Clone, Copy)]
pub(crate) struct CachedAttractorInvariants {
    param_key: Option<(u64, u64, u64, u64)>,
    lyapunov: Option<f64>,
    corr_key: Option<(u64, usize)>,
    correlation_sum: Option<f64>,
    correlation_dim: Option<f64>,
}

impl AttractorPlotter {
    fn flow_config_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.mode as u8).hash(&mut h);
        self.system.sigma.to_bits().hash(&mut h);
        self.system.rho.to_bits().hash(&mut h);
        self.system.beta.to_bits().hash(&mut h);
        self.dt.to_bits().hash(&mut h);
        if self.mode == super::AttractorMode::Custom {
            for bar in self.flow_bank.bars() {
                bar.raw_text.hash(&mut h);
            }
            for &pv in self.flow_bank.param_slice() {
                pv.to_bits().hash(&mut h);
            }
        }
        h.finish()
    }

    /// Computes and caches the largest Lyapunov exponent `λ₁` via Wolf's algorithm.
    pub fn current_lyapunov_exponent(&mut self) -> Option<f64> {
        if self.mode != super::AttractorMode::Lorenz || !self.is_finite_state() || self.diverged() {
            return None;
        }
        let key = (
            self.system.sigma.to_bits(),
            self.system.rho.to_bits(),
            self.system.beta.to_bits(),
            self.dt.to_bits(),
        );
        if self.invariants_cache.param_key != Some(key) {
            let step = self.dt.clamp(0.002, 0.02);
            let probe = LorenzBuilder::new()
                .sigma(self.system.sigma)
                .rho(self.system.rho)
                .beta(self.system.beta)
                .build(LorenzState::new(1.0, 1.0, 1.0));
            let init = Vector3::new(1.0, 1.0, 1.0);
            let mut rk4 = RungeKutta4::new(&init);
            self.invariants_cache.lyapunov =
                lorenz_lyapunov(&probe, &mut rk4, init, step, 60, 0.25)
                    .ok()
                    .filter(|v| v.is_finite());
            self.invariants_cache.param_key = Some(key);
        }
        self.invariants_cache.lyapunov
    }

    /// Computes and caches Grassberger-Procaccia `(C(ε), D₂)` invariants on the 3D trajectory.
    pub fn current_correlation_invariants(&mut self) -> Option<(f64, f64)> {
        if !self.is_finite_state() || self.diverged() {
            return None;
        }
        let key = (self.flow_config_hash(), self.step_count / 25);
        if self.invariants_cache.corr_key != Some(key) {
            let sample = self.collect_or_seed_trajectory(600);
            let c1 = correlation_dimension(&sample, 2.5);
            let c2 = correlation_dimension(&sample, 7.5);
            let d2 = if c1 > 0.0 && c2 > c1 {
                ((c2.ln() - c1.ln()) / (7.5_f64 / 2.5_f64).ln()).clamp(0.0, 3.0)
            } else {
                c2
            };
            self.invariants_cache.correlation_sum = c2.is_finite().then_some(c2);
            self.invariants_cache.correlation_dim = d2.is_finite().then_some(d2);
            self.invariants_cache.corr_key = Some(key);
        }
        Some((
            self.invariants_cache.correlation_sum?,
            self.invariants_cache.correlation_dim?,
        ))
    }

    fn seed_custom_trajectory(&self, dt: f64, fallback_steps: usize) -> Vec<Vector3<f64>> {
        let job = super::CompiledFlowJob::from_bank(&self.flow_bank);
        let mut pos = Vector3::new(1.0, 1.0, 1.0);
        for _ in 0..400 {
            let next = job.rk4_step(pos, dt);
            if !(next.x.is_finite() && next.y.is_finite() && next.z.is_finite()) {
                break;
            }
            pos = next;
        }
        let mut out = Vec::with_capacity(fallback_steps);
        for _ in 0..fallback_steps {
            pos = job.rk4_step(pos, dt);
            if pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite() {
                out.push(pos);
            } else {
                break;
            }
        }
        if out.is_empty() {
            out.push(Vector3::new(1.0, 1.0, 1.0));
        }
        out
    }

    fn collect_or_seed_trajectory(&self, fallback_steps: usize) -> Vec<Vector3<f64>> {
        let finite_hist: Vec<Vector3<f64>> = self
            .history
            .iter()
            .copied()
            .filter(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite())
            .collect();
        if finite_hist.len() >= 32 {
            return finite_hist;
        }
        let dt = self.dt.clamp(0.001, 0.05);
        if self.mode == super::AttractorMode::Custom {
            return self.seed_custom_trajectory(dt, fallback_steps);
        }
        let mut sim = LorenzBuilder::new()
            .sigma(self.system.sigma)
            .rho(self.system.rho)
            .beta(self.system.beta)
            .build(LorenzState::new(1.0, 1.0, 1.0));
        let mut rk4 = RungeKutta4::new(&sim.state.vec);
        for _ in 0..400 {
            sim.step_with(&mut rk4, dt);
        }
        let mut out = Vec::with_capacity(fallback_steps);
        for _ in 0..fallback_steps {
            sim.step_with(&mut rk4, dt);
            let v = sim.state.vec;
            if v.x.is_finite() && v.y.is_finite() && v.z.is_finite() {
                out.push(v);
            } else {
                break;
            }
        }
        if out.is_empty() {
            out.push(Vector3::new(1.0, 1.0, 1.0));
        }
        out
    }

    /// Extracts up to `MAX_CHAOS_LANDMARKS` 3D landmarks and sends them to `VietorisRipsTool`.
    pub fn send_landmarks_to_vietoris_rips(
        &self,
        ctx: Option<&egui::Context>,
    ) -> Result<ChaosLandmarkHandoff, String> {
        if !self.is_finite_state() || self.diverged() {
            return Err("Cannot export landmarks from non-finite attractor state".to_string());
        }
        let traj = self.collect_or_seed_trajectory(1200);
        let pts: Vec<Point3D> = traj
            .into_iter()
            .map(|v| Point3D::new(v.x, v.y, v.z))
            .collect();
        let landmarks =
            farthest_point_landmarks(&pts, MAX_CHAOS_LANDMARKS).map_err(|e| format!("{e:?}"))?;
        let payload = ChaosLandmarkHandoff {
            points: landmarks,
            sigma: self.system.sigma,
            rho: self.system.rho,
            beta: self.system.beta,
            dt: self.dt,
        };
        submit_chaos_landmark_handoff(payload.clone());
        if let Some(c) = ctx {
            request_open_vietoris_rips(c);
        }
        Ok(payload)
    }
}
