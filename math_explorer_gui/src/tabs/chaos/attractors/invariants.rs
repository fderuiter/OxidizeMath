#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::AttractorPlotter;
use crate::tabs::geometry_topology::vietoris_rips::{
    handoff::{request_open_vietoris_rips, submit_chaos_landmark_handoff, ChaosLandmarkHandoff},
    MAX_CHAOS_LANDMARKS,
};
use eframe::egui;
use math_explorer::physics::chaos::fractals::correlation_dimension_in_place;
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState};
use math_explorer::physics::chaos::metrics::lorenz_lyapunov;
use math_explorer::pure_math::analysis::ode::RungeKutta4;
use math_explorer::pure_math::statistics::tda::{farthest_point_landmarks, Point3D};
use nalgebra::Vector3;

#[derive(Default, Clone)]
pub(crate) struct CachedAttractorInvariants {
    param_key: Option<u64>,
    lyapunov: Option<f64>,
    corr_key: Option<(u64, usize)>,
    correlation_sum: Option<f64>,
    correlation_dim: Option<f64>,
    scratch: Vec<Vector3<f64>>,
}

impl CachedAttractorInvariants {
    pub(crate) fn invalidate(&mut self) {
        self.param_key = None;
        self.corr_key = None;
    }
}

impl AttractorPlotter {
    pub(super) fn flow_config_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.mode as u8).hash(&mut h);
        self.system_kind.hash(&mut h);
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
        } else if self.system_kind != super::AttractorSystemKind::Lorenz {
            for v in [
                self.extra_params.rossler_a,
                self.extra_params.rossler_b,
                self.extra_params.rossler_c,
                self.extra_params.pendulum_m1,
                self.extra_params.pendulum_m2,
                self.extra_params.pendulum_l1,
                self.extra_params.pendulum_l2,
                self.extra_params.pendulum_g,
            ] {
                v.to_bits().hash(&mut h);
            }
        }
        h.finish()
    }

    /// Computes and caches the largest Lyapunov exponent `λ₁` via Wolf's algorithm.
    pub fn current_lyapunov_exponent(&mut self) -> Option<f64> {
        if !self.is_finite_state() || self.diverged() {
            return None;
        }
        let key = self.flow_config_hash();
        if self.invariants_cache.param_key != Some(key) {
            let step = self.dt.clamp(0.002, 0.02);
            let mut init = Vector3::new(1.0, 1.0, 1.0);
            let mut rk4 = RungeKutta4::new(&init);
            self.invariants_cache.lyapunov = match self.mode {
                super::AttractorMode::Lorenz => {
                    let probe = LorenzBuilder::new()
                        .sigma(self.system.sigma)
                        .rho(self.system.rho)
                        .beta(self.system.beta)
                        .build(LorenzState::new(1.0, 1.0, 1.0));
                    lorenz_lyapunov(&probe, &mut rk4, init, step, 60, 0.25)
                        .ok()
                        .filter(|v| v.is_finite())
                }
                super::AttractorMode::Custom => {
                    let job = super::CompiledFlowJob::from_bank(&self.flow_bank);
                    for _ in 0..200 {
                        let next = job.rk4_step(init, step);
                        if !(next.x.is_finite() && next.y.is_finite() && next.z.is_finite())
                            || next.norm() > 1e6
                        {
                            break;
                        }
                        init = next;
                    }
                    lorenz_lyapunov(&job, &mut rk4, init, step, 60, 0.25)
                        .ok()
                        .filter(|v| v.is_finite())
                }
            };
            self.invariants_cache.param_key = Some(key);
        }
        self.invariants_cache.lyapunov
    }

    /// Computes and caches Grassberger-Procaccia `(C(ε), D₂)` invariants via zero-alloc in-place sorting.
    pub fn current_correlation_invariants(&mut self) -> Option<(f64, f64)> {
        if !self.is_finite_state() || self.diverged() {
            return None;
        }
        let key = (self.flow_config_hash(), self.step_count / 25);
        if self.invariants_cache.corr_key != Some(key) {
            let mut scratch = std::mem::take(&mut self.invariants_cache.scratch);
            self.fill_or_seed_trajectory(&mut scratch, 600);
            let c1 = correlation_dimension_in_place(&mut scratch, 2.5);
            let c2 = correlation_dimension_in_place(&mut scratch, 7.5);
            self.invariants_cache.scratch = scratch;
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

    fn seed_custom_trajectory(&self, out: &mut Vec<Vector3<f64>>, dt: f64, fallback_steps: usize) {
        let job = super::CompiledFlowJob::from_bank(&self.flow_bank);
        let mut pos = Vector3::new(1.0, 1.0, 1.0);
        for _ in 0..400 {
            let next = job.rk4_step(pos, dt);
            if !(next.x.is_finite() && next.y.is_finite() && next.z.is_finite()) {
                break;
            }
            pos = next;
        }
        out.clear();
        out.reserve(fallback_steps);
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
    }

    pub(super) fn collect_or_seed_trajectory(&self, fallback_steps: usize) -> Vec<Vector3<f64>> {
        let mut out = Vec::with_capacity(fallback_steps);
        self.fill_or_seed_trajectory(&mut out, fallback_steps);
        out
    }

    fn fill_or_seed_trajectory(&self, out: &mut Vec<Vector3<f64>>, fallback_steps: usize) {
        out.clear();
        out.extend(
            self.history
                .iter()
                .copied()
                .filter(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite()),
        );
        if out.len() >= 32 {
            return;
        }
        let dt = self.dt.clamp(0.001, 0.05);
        if self.mode == super::AttractorMode::Custom {
            self.seed_custom_trajectory(out, dt, fallback_steps);
            return;
        }
        let mut sim = LorenzBuilder::new()
            .sigma(self.system.sigma)
            .rho(self.system.rho)
            .beta(self.system.beta)
            .build(LorenzState::new(1.0, 1.0, 1.0));
        let mut rk4 = RungeKutta4::new(&Vector3::zeros());
        for _ in 0..400 {
            sim.step_with(&mut rk4, dt);
        }
        out.clear();
        out.reserve(fallback_steps);
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
    }

    /// Extracts up to `MAX_CHAOS_LANDMARKS` 3D landmarks and sends them to `VietorisRipsTool`.
    pub fn send_landmarks_to_vietoris_rips(
        &self,
        ctx: Option<&egui::Context>,
    ) -> Result<ChaosLandmarkHandoff, String> {
        if !self.is_finite_state() || self.diverged() {
            return Err("Cannot export landmarks from non-finite attractor state".to_string());
        }
        let mut traj = Vec::with_capacity(1200);
        self.fill_or_seed_trajectory(&mut traj, 1200);
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
        if let Some(c) = ctx {
            submit_chaos_landmark_handoff(c, payload.clone());
            request_open_vietoris_rips(c);
        } else if let Some(bus) = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .sim_bus
            .clone()
        {
            let raw_pts = payload.points.iter().map(|p| [p.x, p.y, p.z]).collect();
            let _ = bus.submit_vr_handoff(
                raw_pts,
                payload.sigma,
                payload.rho,
                payload.beta,
                payload.dt,
                false,
            );
        }
        Ok(payload)
    }

    pub(super) fn draw_telemetry_and_handoff(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let labels = if self.mode == super::AttractorMode::Custom {
            ["x₀", "y₀", "z₀"]
        } else {
            self.system_kind.coord_labels()
        };
        let state = self.system.state.vec;
        ui.label(format!(
            "Primary ({}, {}, {}): ({:.2}, {:.2}, {:.2})",
            labels[0], labels[1], labels[2], state.x, state.y, state.z
        ));
        if self.shadow_enabled {
            if let Some(curr) = self.timeline.current() {
                let sh = curr.shadow_vec3();
                ui.label(format!("Shadow: ({:.2}, {:.2}, {:.2})", sh.x, sh.y, sh.z));
            }
        }
        if self.mode == super::AttractorMode::Custom
            || self.system_kind == super::AttractorSystemKind::Lorenz
        {
            if let Some(lyap) = self.current_lyapunov_exponent() {
                let prefix = if self.mode == super::AttractorMode::Lorenz {
                    "Lorenz Lyapunov"
                } else {
                    "Flow Lyapunov"
                };
                ui.label(format!("{prefix} λ₁: {lyap:+.3}"));
            }
        }
        if let Some((c_eps, d2)) = self.current_correlation_invariants() {
            ui.label(format!("Correlation C(ε): {c_eps:.3} (D₂ ≈ {d2:.2})"));
        }
        ui.separator();
        if ui
            .add_enabled(
                !self.diverged() && self.is_finite_state(),
                egui::Button::new("🕸 Send 3D Landmarks to Vietoris-Rips"),
            )
            .clicked()
        {
            let _ = self.send_landmarks_to_vietoris_rips(Some(ctx));
        }
    }
}
