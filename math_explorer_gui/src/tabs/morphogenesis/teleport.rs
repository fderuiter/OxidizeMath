//! Atlas teleport, washout re-seeding, and finite-state recovery for [`super::MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::{classify_gray_scott, GrayScottRegime};
use super::cartographer::{grid_u_variance, is_pattern_forming_regime, WASHOUT_VARIANCE_THRESHOLD};
use super::runner::MorphoSnapshotData;
use super::state::{
    ensure_nucleated_for_kinetics, initialize_schnakenberg_system, MorphogenesisKinetics,
    DEFAULT_GRAY_SCOTT_F, DEFAULT_GRAY_SCOTT_K,
};
use super::MorphogenesisTool;
use crate::async_sim::atlas::AtlasTeleportEvent;
use crate::async_sim::{SimCommand, SimWorkerState};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use math_explorer::biology::morphogenesis::FusedEulerSolver;
use std::hash::{Hash, Hasher};

impl MorphogenesisTool {
    pub(crate) fn non_axis_context_sig(&self) -> u64 {
        let (xk, yk) = self.atlas.axis_keys();
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        let mut h = std::collections::hash_map::DefaultHasher::new();
        p.kinetics.hash(&mut h);
        p.auto_dt.hash(&mut h);
        for (name, val) in [
            ("a", p.a),
            ("b", p.b),
            ("f", p.f),
            ("k", p.k),
            ("d_u", p.d_u),
            ("d_v", p.d_v),
            ("dt", p.dt),
        ] {
            if name != xk && name != yk {
                name.hash(&mut h);
                val.to_bits().hash(&mut h);
            }
        }
        h.finish()
    }

    pub(crate) fn active_axis_values(&self) -> (f64, f64) {
        let (xk, yk) = self.atlas.axis_keys();
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        let get = |k: &str| match k {
            "a" => p.a,
            "b" => p.b,
            "f" => p.f,
            "k" => p.k,
            "d_u" => p.d_u,
            "d_v" => p.d_v,
            "dt" => p.dt,
            _ => 0.0,
        };
        (get(xk), get(yk))
    }

    /// Returns `true` when the latest activator concentration grid is uniform (`Var(u) < 1e-8`).
    #[must_use]
    pub fn is_active_grid_washed_out(&self) -> bool {
        if let Some(sr) = &self.step_runner {
            let v = grid_u_variance(sr.system.u());
            return v.is_finite() && v < WASHOUT_VARIANCE_THRESHOLD;
        }
        if let Some(snap) = self.worker.latest_snapshot() {
            let u_slice = snap.structured_data.as_ref().and_then(|d| {
                d.downcast_ref::<MorphoSnapshotData>()
                    .map(|m| m.u_data.as_slice())
                    .or_else(|| d.downcast_ref::<Vec<f64>>().map(Vec::as_slice))
            });
            if let Some(u_data) = u_slice {
                let v = grid_u_variance(u_data);
                return v.is_finite() && v < WASHOUT_VARIANCE_THRESHOLD;
            }
        }
        false
    }

    fn reseed_gray_scott_if_needed(&mut self, f: f64, k: f64, d_u: f64, d_v: f64) -> bool {
        if classify_gray_scott(f, k, d_u, d_v) == GrayScottRegime::UniformStable
            || !self.is_active_grid_washed_out()
        {
            return false;
        }
        if let Some(sr) = &mut self.step_runner {
            sr.gray_scott.f = f;
            sr.gray_scott.k = k;
            sr.system.diffusion_coeffs = [d_u, d_v];
            ensure_nucleated_for_kinetics(
                &mut sr.system,
                MorphogenesisKinetics::GrayScott,
                self.width,
                self.height,
            );
            sr.enforce_mask_concentrations();
        }
        let _ = self
            .worker
            .send_command(SimCommand::Custom("ensure_nucleated".to_string()));
        self.dirty = true;
        true
    }

    /// Re-seeds deterministic spatial noise if the active grid washed out and current parameters form patterns.
    pub fn reseed_if_washed_out_entering_pattern(&mut self) -> bool {
        let p = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if p.kinetics == MorphogenesisKinetics::GrayScott {
            return self.reseed_gray_scott_if_needed(p.f, p.k, p.d_u, p.d_v);
        }
        let eff_dt = if p.auto_dt {
            FusedEulerSolver::cfl_substeps_2d(
                p.dt,
                1.0,
                p.d_u,
                p.d_v,
                super::runner::MAX_CFL_SUBSTEPS,
            )
            .1
        } else {
            p.dt
        };
        if !is_pattern_forming_regime(p.a, p.b, p.d_u, p.d_v, eff_dt)
            || !self.is_active_grid_washed_out()
        {
            return false;
        }
        if let Some(sr) = &mut self.step_runner {
            sr.system.kinetics.a = p.a;
            sr.system.kinetics.b = p.b;
            sr.system.diffusion_coeffs = [p.d_u, p.d_v];
            initialize_schnakenberg_system(&mut sr.system, self.width, self.height);
            sr.enforce_mask_concentrations();
        }
        let _ = self
            .worker
            .send_command(SimCommand::Custom("reinit".to_string()));
        self.dirty = true;
        true
    }

    /// Applies an [`AtlasTeleportEvent`] to live morphogenesis parameters with CFL recovery and washout re-seeding.
    pub fn apply_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        let was_faulted = self.worker.state() == SimWorkerState::Faulted;
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
                match key {
                    "a" => p.a = val,
                    "b" => p.b = val,
                    "f" => p.f = val,
                    "k" => p.k = val,
                    "d_u" => p.d_u = val,
                    "d_v" => p.d_v = val,
                    "dt" => p.dt = val,
                    _ => {}
                }
            }
            if was_faulted
                && ev.x_key != "dt"
                && ev.y_key != "dt"
                && 4.0 * p.d_u.max(p.d_v) * p.dt >= 1.0
            {
                p.dt = (0.2 / p.d_u.max(p.d_v).max(1e-6)).clamp(1e-4, 0.1);
            }
        }
        if was_faulted {
            self.reset_system();
            return;
        }
        let _ = self.reseed_if_washed_out_entering_pattern();
        self.pending_brush_frames = 2;
        self.dirty = true;
    }
}

impl FiniteStateGuard for MorphogenesisTool {
    fn is_finite_state(&self) -> bool {
        self.worker.state() != SimWorkerState::Faulted && self.worker.diagnostic_report().is_none()
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.worker.diagnostic_report()
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }

    fn auto_scale_step_size(&mut self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.dt = (p.dt * 0.5).max(1e-5);
        drop(p);
        self.reset_system();
    }

    fn clamp_parameters(&mut self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.a = p.a.clamp(0.0, 1.0);
        p.b = p.b.clamp(0.0, 2.0);
        p.f = p.f.clamp(0.0, 0.2);
        p.k = p.k.clamp(0.0, 0.2);
        p.d_u = p.d_u.clamp(0.01, 5.0);
        p.d_v = p.d_v.clamp(0.01, 200.0);
        p.dt = p.dt.clamp(1e-4, 2.0);
        p.auto_dt = true;
        drop(p);
        self.reset_system();
    }

    fn reset_parameters(&mut self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.kinetics = MorphogenesisKinetics::Schnakenberg;
        p.a = 0.1;
        p.b = 0.9;
        p.f = DEFAULT_GRAY_SCOTT_F;
        p.k = DEFAULT_GRAY_SCOTT_K;
        p.d_u = 1.0;
        p.d_v = 100.0;
        p.dt = 0.05;
        p.auto_dt = true;
        drop(p);
        self.reset_system();
    }
}
