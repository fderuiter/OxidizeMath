//! Mode-aware atlas cartographer dispatch, teleportation, washout re-seeding, and
//! finite-state recovery for [`super::MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::cartographer::{
    grid_u_variance, is_pattern_forming_regime, MorphogenesisCartographer, MORPHO_ATLAS_AXIS_PAIRS,
    WASHOUT_VARIANCE_THRESHOLD,
};
use super::gray_scott_atlas::{
    is_gray_scott_pattern_viable, is_gray_scott_v_washed_out, GrayScottCartographer,
    GRAY_SCOTT_ATLAS_AXIS_PAIRS,
};
use super::runner::MorphoSnapshotData;
use super::state::{
    initialize_gray_scott_system, initialize_schnakenberg_system, MorphogenesisKinetics,
    DEFAULT_GRAY_SCOTT_F, DEFAULT_GRAY_SCOTT_K,
};
use super::MorphogenesisTool;
use crate::async_sim::atlas::{AtlasTeleportEvent, RegimeCartographer};
use crate::async_sim::{SimCommand, SimWorkerState};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use math_explorer::biology::morphogenesis::FusedEulerSolver;
use scientific_metadata::theory::TheoryDescribable;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

/// Shared Gray-Scott mode and `(f, k)` parameters synchronized with [`MorphogenesisSharedParams`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MorphogenesisKineticsState {
    /// Selected reaction kinetics regime.
    pub mode: MorphogenesisKinetics,
    /// Gray-Scott substrate feed rate `f`.
    pub f: f64,
    /// Gray-Scott autocatalyst kill rate `k`.
    pub k: f64,
}

impl Default for MorphogenesisKineticsState {
    fn default() -> Self {
        Self {
            mode: MorphogenesisKinetics::Schnakenberg,
            f: DEFAULT_GRAY_SCOTT_F,
            k: DEFAULT_GRAY_SCOTT_K,
        }
    }
}

/// Thread-safe parameter block shared between [`super::MorphogenesisTool`] and [`super::MorphogenesisRunner`].
#[derive(Clone, Copy, Debug)]
pub struct MorphogenesisSharedParams {
    /// Active reaction kinetics family (`Schnakenberg` or `GrayScott`).
    pub kinetics: MorphogenesisKinetics,
    /// Schnakenberg activator production rate `a`.
    pub a: f64,
    /// Schnakenberg inhibitor production rate `b`.
    pub b: f64,
    /// Gray-Scott dimensionless substrate feed rate `F`.
    pub f: f64,
    /// Gray-Scott dimensionless autocatalyst kill rate `k`.
    pub k: f64,
    /// Species `u` diffusion coefficient `D_u`.
    pub d_u: f64,
    /// Species `v` diffusion coefficient `D_v`.
    pub d_v: f64,
    /// Target integration step `dt` (substepped within CFL bound when `auto_dt` is `true`).
    pub dt: f64,
    /// Enables automatic 2D CFL time-step clamping (`dt <= 0.8 * dx^2 / (4 * max(D_u, D_v))`).
    pub auto_dt: bool,
    /// Whether high-contrast accessible colormap tokens are active.
    pub high_contrast: bool,
}

impl Default for MorphogenesisSharedParams {
    fn default() -> Self {
        Self {
            kinetics: MorphogenesisKinetics::Schnakenberg,
            a: 0.1,
            b: 0.9,
            f: DEFAULT_GRAY_SCOTT_F,
            k: DEFAULT_GRAY_SCOTT_K,
            d_u: 1.0,
            d_v: 100.0,
            dt: 0.05,
            auto_dt: true,
            high_contrast: false,
        }
    }
}

pub(super) fn reconcile_kinetics_state(
    params: &RwLock<MorphogenesisSharedParams>,
    ks_arc: &RwLock<MorphogenesisKineticsState>,
    last_synced_ks: &mut MorphogenesisKineticsState,
    p: &mut MorphogenesisSharedParams,
) {
    let ks = *ks_arc.read().unwrap_or_else(|e| e.into_inner());
    let ks_changed = ks.mode != last_synced_ks.mode
        || ks.f.to_bits() != last_synced_ks.f.to_bits()
        || ks.k.to_bits() != last_synced_ks.k.to_bits();
    let p_changed = p.kinetics != last_synced_ks.mode
        || p.f.to_bits() != last_synced_ks.f.to_bits()
        || p.k.to_bits() != last_synced_ks.k.to_bits();
    if ks_changed {
        p.kinetics = ks.mode;
        p.f = ks.f;
        p.k = ks.k;
        if let Ok(mut wg) = params.try_write() {
            wg.kinetics = ks.mode;
            wg.f = ks.f;
            wg.k = ks.k;
        }
        *last_synced_ks = ks;
    } else if p_changed {
        let next_ks = MorphogenesisKineticsState {
            mode: p.kinetics,
            f: p.f,
            k: p.k,
        };
        if let Ok(mut wg) = ks_arc.try_write() {
            *wg = next_ks;
        }
        *last_synced_ks = next_ks;
    }
}

impl MorphogenesisTool {
    pub(crate) fn effective_kinetics_triplet(&self) -> (MorphogenesisKinetics, f64, f64) {
        let p = *self.params.read().unwrap_or_else(|e| e.into_inner());
        let ks = *self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let def = MorphogenesisKineticsState::default();
        if ks != def
            && p.kinetics == def.mode
            && p.f.to_bits() == def.f.to_bits()
            && p.k.to_bits() == def.k.to_bits()
        {
            (ks.mode, ks.f, ks.k)
        } else {
            (p.kinetics, p.f, p.k)
        }
    }

    pub(crate) fn sync_atlas_options_for_mode(&mut self, mode: MorphogenesisKinetics) {
        let pairs = match mode {
            MorphogenesisKinetics::Schnakenberg => MORPHO_ATLAS_AXIS_PAIRS.to_vec(),
            MorphogenesisKinetics::GrayScott => GRAY_SCOTT_ATLAS_AXIS_PAIRS.to_vec(),
        };
        self.atlas.replace_axis_pair_options(pairs);
    }

    pub(crate) fn build_active_morpho_cartographer(
        &self,
        keys: (&'static str, &'static str),
    ) -> Arc<dyn RegimeCartographer> {
        let (mode, f, k) = self.effective_kinetics_triplet();
        let theory = self.theory.clone().with_mode(mode);
        let theory_params = theory.theory_parameters();
        let p = *self.params.read().unwrap_or_else(|e| e.into_inner());
        match mode {
            MorphogenesisKinetics::Schnakenberg => Arc::new(
                MorphogenesisCartographer::new(&theory_params, keys, p.a, p.b, p.d_u, p.d_v, p.dt)
                    .with_auto_dt(p.auto_dt),
            ),
            MorphogenesisKinetics::GrayScott => Arc::new(
                GrayScottCartographer::new(&theory_params, keys, f, k, p.d_u, p.d_v, p.dt)
                    .with_auto_dt(p.auto_dt),
            ),
        }
    }

    pub(crate) fn non_axis_context_sig(&self) -> u64 {
        let (xk, yk) = self.atlas.axis_keys();
        let (mode, f, k) = self.effective_kinetics_triplet();
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        let mut h = std::collections::hash_map::DefaultHasher::new();
        mode.hash(&mut h);
        p.auto_dt.hash(&mut h);
        for (name, val) in [
            ("a", p.a),
            ("b", p.b),
            ("f", f),
            ("k", k),
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
        let (_, f, k) = self.effective_kinetics_triplet();
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        let get = |key: &str| match key {
            "a" => p.a,
            "b" => p.b,
            "f" => f,
            "k" => k,
            "d_u" => p.d_u,
            "d_v" => p.d_v,
            "dt" => p.dt,
            _ => 0.0,
        };
        (get(xk), get(yk))
    }

    /// Returns `true` when the latest concentration grid (`u` or Gray-Scott `v`) is washed out.
    #[must_use]
    pub fn is_active_grid_washed_out(&self) -> bool {
        let (mode, _, _) = self.effective_kinetics_triplet();
        if let Some(sr) = &self.step_runner {
            return match mode {
                MorphogenesisKinetics::Schnakenberg => {
                    let v = grid_u_variance(sr.system.u());
                    v.is_finite() && v < WASHOUT_VARIANCE_THRESHOLD
                }
                MorphogenesisKinetics::GrayScott => is_gray_scott_v_washed_out(sr.system.v()),
            };
        }
        if let Some(snap) = self.worker.latest_snapshot() {
            let u_slice = snap.structured_data.as_ref().and_then(|d| {
                d.downcast_ref::<MorphoSnapshotData>()
                    .map(|m| m.u_data.as_slice())
                    .or_else(|| d.downcast_ref::<Vec<f64>>().map(Vec::as_slice))
            });
            if let Some(u_data) = u_slice {
                return match mode {
                    MorphogenesisKinetics::Schnakenberg => {
                        let v = grid_u_variance(u_data);
                        v.is_finite() && v < WASHOUT_VARIANCE_THRESHOLD
                    }
                    MorphogenesisKinetics::GrayScott => is_gray_scott_v_washed_out(u_data),
                };
            }
        }
        false
    }

    fn reseed_gray_scott_if_needed(&mut self, f: f64, k: f64, d_u: f64, d_v: f64, dt: f64) -> bool {
        if !is_gray_scott_pattern_viable(f, k, d_u, d_v, dt) || !self.is_active_grid_washed_out() {
            return false;
        }
        if let Some(sr) = &mut self.step_runner {
            sr.kinetics = MorphogenesisKinetics::GrayScott;
            sr.gray_scott.f = f;
            sr.gray_scott.k = k;
            sr.system.diffusion_coeffs = [d_u, d_v];
            initialize_gray_scott_system(&mut sr.system, self.width, self.height);
            sr.enforce_mask_concentrations();
        }
        let _ = self
            .worker
            .send_command(SimCommand::Custom("ensure_nucleated".to_string()));
        self.dirty = true;
        true
    }

    /// Re-seeds deterministic spatial perturbations or Gray-Scott catalyst patches if washed out.
    pub fn reseed_if_washed_out_entering_pattern(&mut self) -> bool {
        let (mode, f, k) = self.effective_kinetics_triplet();
        let p = *self.params.read().unwrap_or_else(|e| e.into_inner());
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
        if mode == MorphogenesisKinetics::GrayScott {
            return self.reseed_gray_scott_if_needed(f, k, p.d_u, p.d_v, eff_dt);
        }
        if !is_pattern_forming_regime(p.a, p.b, p.d_u, p.d_v, eff_dt)
            || !self.is_active_grid_washed_out()
        {
            return false;
        }
        if let Some(sr) = &mut self.step_runner {
            sr.kinetics = MorphogenesisKinetics::Schnakenberg;
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

    fn apply_gray_scott_teleport(&mut self, ev: AtlasTeleportEvent, was_faulted: bool) {
        let (_, mut next_f, mut next_k) = self.effective_kinetics_triplet();
        for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            match key {
                "f" => next_f = val.clamp(0.0, 0.2),
                "k" => next_k = val.clamp(0.0, 0.2),
                _ => {}
            }
        }
        self.theory.mode = MorphogenesisKinetics::GrayScott;
        {
            let mut ks = self
                .kinetics_state
                .write()
                .unwrap_or_else(|e| e.into_inner());
            ks.mode = MorphogenesisKinetics::GrayScott;
            ks.f = next_f;
            ks.k = next_k;
        }
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.kinetics = MorphogenesisKinetics::GrayScott;
            p.f = next_f;
            p.k = next_k;
            for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
                match key {
                    "d_u" => p.d_u = val.clamp(0.01, 5.0),
                    "d_v" => p.d_v = val.clamp(0.005, 200.0),
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
        self.pending_brush_frames = 2;
        self.dirty = true;
    }

    /// Applies an [`AtlasTeleportEvent`] while preserving the active kinetics mode (`Schnakenberg` or `GrayScott`).
    pub fn apply_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        let was_faulted = self.worker.state() == SimWorkerState::Faulted;
        let (current_mode, _, _) = self.effective_kinetics_triplet();
        let is_gray_scott = current_mode == MorphogenesisKinetics::GrayScott
            || matches!(ev.x_key, "f" | "k")
            || matches!(ev.y_key, "f" | "k");
        if is_gray_scott {
            self.apply_gray_scott_teleport(ev, was_faulted);
            return;
        }
        self.theory.mode = MorphogenesisKinetics::Schnakenberg;
        self.kinetics_state
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .mode = MorphogenesisKinetics::Schnakenberg;
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.kinetics = MorphogenesisKinetics::Schnakenberg;
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
        p.dt = (p.dt * 0.5).max(1e-4);
        drop(p);
        self.reset_system();
    }

    fn clamp_parameters(&mut self) {
        let (mode, f, k) = self.effective_kinetics_triplet();
        let cf = f.clamp(0.0, 0.2);
        let ck = k.clamp(0.0, 0.2);
        {
            let mut ks = self
                .kinetics_state
                .write()
                .unwrap_or_else(|e| e.into_inner());
            ks.mode = mode;
            ks.f = cf;
            ks.k = ck;
        }
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.kinetics = mode;
            p.a = p.a.clamp(0.0, 1.0);
            p.b = p.b.clamp(0.0, 2.0);
            p.f = cf;
            p.k = ck;
            p.d_u = p.d_u.clamp(0.01, 5.0);
            p.d_v = p.d_v.clamp(0.005, 200.0);
            p.dt = p.dt.clamp(1e-4, 2.0);
            p.auto_dt = true;
        }
        self.reset_system();
    }

    fn reset_parameters(&mut self) {
        self.theory.mode = MorphogenesisKinetics::Schnakenberg;
        *self
            .kinetics_state
            .write()
            .unwrap_or_else(|e| e.into_inner()) = MorphogenesisKineticsState::default();
        {
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
        }
        self.reset_system();
    }
}
