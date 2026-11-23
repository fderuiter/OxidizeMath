//! Mode-aware atlas context, washout detection, auto-reseed, and teleportation bridge
//! for [`super::MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::{
    grid_u_variance, is_pattern_forming_regime, MorphogenesisCartographer, MORPHO_ATLAS_AXIS_PAIRS,
    WASHOUT_VARIANCE_THRESHOLD,
};
use super::gray_scott_atlas::{
    is_gray_scott_pattern_viable, is_gray_scott_v_washed_out, GrayScottCartographer,
    GRAY_SCOTT_ATLAS_AXIS_PAIRS,
};
use super::MorphogenesisTool;
use crate::async_sim::atlas::{AtlasTeleportEvent, RegimeCartographer};
use crate::async_sim::{SimCommand, SimWorkerState};
use crate::presets::MorphogenesisKineticsMode;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

impl MorphogenesisTool {
    pub(crate) fn sync_atlas_options_for_mode(&mut self, mode: MorphogenesisKineticsMode) {
        let pairs = match mode {
            MorphogenesisKineticsMode::Schnakenberg => MORPHO_ATLAS_AXIS_PAIRS.to_vec(),
            MorphogenesisKineticsMode::GrayScott => GRAY_SCOTT_ATLAS_AXIS_PAIRS.to_vec(),
        };
        self.atlas.replace_axis_pair_options(pairs);
    }

    pub(crate) fn build_active_morpho_cartographer(
        &self,
        keys: (&'static str, &'static str),
    ) -> Arc<dyn RegimeCartographer> {
        let mode = self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .mode;
        let mut theory = self.theory.clone();
        theory.mode = mode;
        let theory_params =
            scientific_metadata::theory::TheoryDescribable::theory_parameters(&theory);
        let (a, b, d_u, d_v, dt) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v, p.dt)
        };
        match mode {
            MorphogenesisKineticsMode::Schnakenberg => Arc::new(MorphogenesisCartographer::new(
                &theory_params,
                keys,
                a,
                b,
                d_u,
                d_v,
                dt,
            )),
            MorphogenesisKineticsMode::GrayScott => {
                let (f, k) = {
                    let ks = self
                        .kinetics_state
                        .read()
                        .unwrap_or_else(|e| e.into_inner());
                    (ks.f, ks.k)
                };
                Arc::new(GrayScottCartographer::new(
                    &theory_params,
                    keys,
                    f,
                    k,
                    d_u,
                    d_v,
                    dt,
                ))
            }
        }
    }

    pub(crate) fn non_axis_context_sig(&self) -> u64 {
        let (xk, yk) = self.atlas.axis_keys();
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        let ks = self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let mut h = std::collections::hash_map::DefaultHasher::new();
        ks.mode.hash(&mut h);
        for (name, val) in [
            ("a", p.a),
            ("b", p.b),
            ("f", ks.f),
            ("k", ks.k),
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
        let ks = self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let get = |k: &str| match k {
            "a" => p.a,
            "b" => p.b,
            "f" => ks.f,
            "k" => ks.k,
            "d_u" => p.d_u,
            "d_v" => p.d_v,
            "dt" => p.dt,
            _ => 0.0,
        };
        (get(xk), get(yk))
    }

    /// Returns `true` when the latest active concentration grid (`u` or Gray-Scott `v`) is washed out.
    #[must_use]
    pub fn is_active_grid_washed_out(&self) -> bool {
        let mode = self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .mode;
        if let Some(sr) = &self.step_runner {
            return match mode {
                MorphogenesisKineticsMode::Schnakenberg => {
                    let v = grid_u_variance(sr.system.u());
                    v.is_finite() && v < WASHOUT_VARIANCE_THRESHOLD
                }
                MorphogenesisKineticsMode::GrayScott => is_gray_scott_v_washed_out(sr.system.v()),
            };
        }
        if let Some(snap) = self.worker.latest_snapshot() {
            let u_slice = snap.structured_data.as_ref().and_then(|d| {
                d.downcast_ref::<super::runner::MorphoSnapshotData>()
                    .map(|m| m.u_data.as_slice())
                    .or_else(|| d.downcast_ref::<Vec<f64>>().map(Vec::as_slice))
            });
            if let Some(u_data) = u_slice {
                return match mode {
                    MorphogenesisKineticsMode::Schnakenberg => {
                        let v = grid_u_variance(u_data);
                        v.is_finite() && v < WASHOUT_VARIANCE_THRESHOLD
                    }
                    MorphogenesisKineticsMode::GrayScott => is_gray_scott_v_washed_out(u_data),
                };
            }
        }
        false
    }

    pub(crate) fn reseed_gray_scott_if_washed_out(&mut self) -> bool {
        let (f, k) = {
            let ks = self
                .kinetics_state
                .read()
                .unwrap_or_else(|e| e.into_inner());
            (ks.f, ks.k)
        };
        let (d_u, d_v, dt) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.d_u, p.d_v, p.dt)
        };
        if !is_gray_scott_pattern_viable(f, k, d_u, d_v, dt) || !self.is_active_grid_washed_out() {
            return false;
        }
        if let Some(sr) = &mut self.step_runner {
            sr.system.kinetics.mode = MorphogenesisKineticsMode::GrayScott;
            sr.system.kinetics.f = f;
            sr.system.kinetics.k = k;
            sr.system.diffusion_coeffs = [d_u, d_v];
            Self::initialize_adapter_system(&mut sr.system, self.width, self.height);
            sr.enforce_mask_concentrations();
        }
        let _ = self
            .worker
            .send_command(SimCommand::Custom("reinit".to_string()));
        self.dirty = true;
        true
    }

    /// Re-seeds deterministic spatial perturbations or Gray-Scott catalyst patches if washed out.
    pub fn reseed_if_washed_out_entering_pattern(&mut self) -> bool {
        let mode = self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .mode;
        if mode == MorphogenesisKineticsMode::GrayScott {
            return self.reseed_gray_scott_if_washed_out();
        }
        let (a, b, d_u, d_v, dt) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v, p.dt)
        };
        if !is_pattern_forming_regime(a, b, d_u, d_v, dt) || !self.is_active_grid_washed_out() {
            return false;
        }
        if let Some(sr) = &mut self.step_runner {
            sr.system.kinetics.mode = MorphogenesisKineticsMode::Schnakenberg;
            sr.system.kinetics.a = a;
            sr.system.kinetics.b = b;
            sr.system.kinetics.spatial.a = a;
            sr.system.kinetics.spatial.b = b;
            sr.system.diffusion_coeffs = [d_u, d_v];
            Self::initialize_adapter_system(&mut sr.system, self.width, self.height);
            sr.enforce_mask_concentrations();
        }
        let _ = self
            .worker
            .send_command(SimCommand::Custom("reinit".to_string()));
        self.dirty = true;
        true
    }

    pub(crate) fn apply_gray_scott_teleport(&mut self, ev: AtlasTeleportEvent, was_faulted: bool) {
        self.theory.mode = MorphogenesisKineticsMode::GrayScott;
        {
            let mut ks = self
                .kinetics_state
                .write()
                .unwrap_or_else(|e| e.into_inner());
            ks.mode = MorphogenesisKineticsMode::GrayScott;
            for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
                match key {
                    "f" => ks.f = val.clamp(0.0, 0.12),
                    "k" => ks.k = val.clamp(0.0, 0.12),
                    _ => {}
                }
            }
        }
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
                match key {
                    "d_u" => p.d_u = val.clamp(0.01, 1.0),
                    "d_v" => p.d_v = val.clamp(0.005, 1.0),
                    "dt" => p.dt = val.clamp(1e-4, 2.0),
                    _ => {}
                }
            }
            if was_faulted
                && ev.x_key != "dt"
                && ev.y_key != "dt"
                && 8.0 * p.d_u.max(p.d_v) * p.dt >= 1.95
            {
                p.dt = (0.2 / p.d_u.max(p.d_v).max(1e-6)).clamp(1e-4, 1.5);
            }
        }
        if was_faulted {
            self.reset_system();
            return;
        }
        let _ = self.reseed_if_washed_out_entering_pattern();
        self.dirty = true;
    }

    /// Applies an [`AtlasTeleportEvent`] while preserving the active kinetics mode (`Schnakenberg` or `GrayScott`).
    pub fn apply_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        let was_faulted = self.worker.state() == SimWorkerState::Faulted;
        let is_gray_scott = self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .mode
            == MorphogenesisKineticsMode::GrayScott
            || matches!(ev.x_key, "f" | "k")
            || matches!(ev.y_key, "f" | "k");
        if is_gray_scott {
            self.apply_gray_scott_teleport(ev, was_faulted);
            return;
        }
        self.theory.mode = MorphogenesisKineticsMode::Schnakenberg;
        self.kinetics_state
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .mode = MorphogenesisKineticsMode::Schnakenberg;
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
                match key {
                    "a" => p.a = val,
                    "b" => p.b = val,
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
        self.dirty = true;
    }
}
