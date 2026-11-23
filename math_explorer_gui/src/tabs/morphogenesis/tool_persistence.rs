//! Persistence, recovery guard, and parameter-atlas teleport helpers for [`MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::is_pattern_forming_regime;
use super::state::MorphogenesisToolState;
use super::types::{MorphogenesisBrushMode, MorphogenesisFlowState, MorphogenesisKineticsMode};
use super::MorphogenesisTool;
use crate::async_sim::atlas::AtlasTeleportEvent;
use crate::async_sim::{SimCommand, SimWorkerState};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::InteractiveTool;
use math_explorer::biology::morphogenesis::GrayScottKinetics;
use std::hash::{Hash, Hasher};

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
        let mode = {
            let mut ks = self
                .kinetics_state
                .write()
                .unwrap_or_else(|e| e.into_inner());
            ks.f = ks.f.clamp(0.0, 0.12);
            ks.k = ks.k.clamp(0.0, 0.12);
            ks.mode
        };
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.a = p.a.clamp(0.0, 1.0);
            p.b = p.b.clamp(0.0, 2.0);
            match mode {
                MorphogenesisKineticsMode::Schnakenberg => {
                    p.d_u = p.d_u.clamp(0.1, 5.0);
                    p.d_v = p.d_v.clamp(10.0, 200.0);
                    p.dt = p.dt.clamp(0.001, 0.1);
                }
                MorphogenesisKineticsMode::GrayScott => {
                    p.d_u = p.d_u.clamp(0.01, 1.0);
                    p.d_v = p.d_v.clamp(0.005, 1.0);
                    p.dt = p.dt.clamp(0.001, 2.0);
                }
            }
        }
        {
            let mut flow = self.flow_state.write().unwrap_or_else(|e| e.into_inner());
            flow.inlet_velocity = flow.inlet_velocity.clamp(0.0, 2.0);
            flow.viscosity = flow.viscosity.clamp(0.003, 0.5);
            flow.circulation = flow.circulation.clamp(-25.0, 25.0);
            flow.brush_radius = flow.brush_radius.clamp(1, 25);
        }
        self.reset_system();
    }

    fn reset_parameters(&mut self) {
        let gs = GrayScottKinetics::mitosis();
        {
            let mut ks = self
                .kinetics_state
                .write()
                .unwrap_or_else(|e| e.into_inner());
            ks.mode = MorphogenesisKineticsMode::Schnakenberg;
            ks.f = gs.f;
            ks.k = gs.k;
        }
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.a = 0.1;
            p.b = 0.9;
            p.d_u = 1.0;
            p.d_v = 100.0;
            p.dt = 0.05;
        }
        {
            let mut flow = self.flow_state.write().unwrap_or_else(|e| e.into_inner());
            let tm = flow.theme_mode;
            *flow = MorphogenesisFlowState {
                theme_mode: tm,
                ..MorphogenesisFlowState::default()
            };
        }
        self.theory.set_mode(MorphogenesisKineticsMode::Schnakenberg);
        self.reset_system();
    }
}

impl MorphogenesisTool {
    pub(crate) fn compute_state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            p.a.to_bits().hash(&mut h);
            p.b.to_bits().hash(&mut h);
            p.d_u.to_bits().hash(&mut h);
            p.d_v.to_bits().hash(&mut h);
            p.dt.to_bits().hash(&mut h);
        }
        {
            let ks = self
                .kinetics_state
                .read()
                .unwrap_or_else(|e| e.into_inner());
            ks.mode.hash(&mut h);
            ks.f.to_bits().hash(&mut h);
            ks.k.to_bits().hash(&mut h);
        }
        {
            let flow = self.flow_state.read().unwrap_or_else(|e| e.into_inner());
            flow.flow_mode.hash(&mut h);
            flow.inlet_velocity.to_bits().hash(&mut h);
            flow.viscosity.to_bits().hash(&mut h);
            flow.circulation.to_bits().hash(&mut h);
            flow.overlay_mode.hash(&mut h);
            flow.brush_mode.hash(&mut h);
            flow.brush_radius.hash(&mut h);
        }
        self.width.hash(&mut h);
        self.height.hash(&mut h);
        self.mask_rle
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .hash(&mut h);
        std::mem::discriminant(&self.worker.state()).hash(&mut h);
        self.pending_initial_start.hash(&mut h);
        self.atlas.is_visible().hash(&mut h);
        self.atlas.axis_keys().hash(&mut h);
        h.finish().max(1)
    }

    pub(crate) fn serialize_tool_state(&self) -> Option<String> {
        let p = *self.params.read().unwrap_or_else(|e| e.into_inner());
        let ks = *self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let flow = *self.flow_state.read().unwrap_or_else(|e| e.into_inner());
        let mask_rle = self
            .mask_rle
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let (xk, yk) = self.atlas.axis_keys();
        serde_json::to_string(&MorphogenesisToolState {
            a: p.a,
            b: p.b,
            d_u: p.d_u,
            d_v: p.d_v,
            dt: p.dt,
            width: self.width,
            height: self.height,
            kinetics_mode: ks.mode,
            f: ks.f,
            k: ks.k,
            flow_mode: flow.flow_mode,
            inlet_velocity: flow.inlet_velocity,
            viscosity: flow.viscosity,
            circulation: flow.circulation,
            overlay_mode: flow.overlay_mode,
            brush_mode: flow.brush_mode,
            brush_radius: flow.brush_radius,
            state: Some(self.worker.state()),
            paused: None,
            pending_initial_start: self.pending_initial_start,
            mask_rle,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes: [xk.to_string(), yk.to_string()],
        })
        .ok()
    }

    pub(crate) fn load_tool_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: MorphogenesisToolState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Morphogenesis state: {e}"))?;
        parsed.validate()?;
        self.atlas
            .set_axis_keys(&parsed.atlas_axes[0], &parsed.atlas_axes[1])?;
        self.atlas.set_visible(parsed.atlas_visible);
        let target_state = parsed.resolve_target_state();
        self.theory.set_mode(parsed.kinetics_mode);
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.a = parsed.a;
            p.b = parsed.b;
            p.d_u = parsed.d_u;
            p.d_v = parsed.d_v;
            p.dt = parsed.dt;
        }
        {
            let mut ks = self
                .kinetics_state
                .write()
                .unwrap_or_else(|e| e.into_inner());
            ks.mode = parsed.kinetics_mode;
            ks.f = parsed.f;
            ks.k = parsed.k;
        }
        {
            let mut flow = self.flow_state.write().unwrap_or_else(|e| e.into_inner());
            flow.flow_mode = parsed.flow_mode;
            flow.inlet_velocity = parsed.inlet_velocity;
            flow.viscosity = parsed.viscosity;
            flow.circulation = parsed.circulation;
            flow.overlay_mode = parsed.overlay_mode;
            flow.brush_mode = parsed.brush_mode;
            flow.brush_radius = parsed.brush_radius;
        }
        if parsed.brush_mode == MorphogenesisBrushMode::PanZoom {
            self.brush_kit.interaction_mode = crate::framework::BrushInteractionMode::Pan;
        }
        *self.mask_rle.write().unwrap_or_else(|e| e.into_inner()) = parsed.mask_rle;
        self.apply_dimensions_and_target_state(
            parsed.width,
            parsed.height,
            target_state,
            parsed.pending_initial_start,
        );
        Ok(())
    }

    pub(crate) fn params_are_finite(&self) -> bool {
        let p = *self.params.read().unwrap_or_else(|e| e.into_inner());
        let ks = *self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let flow = *self.flow_state.read().unwrap_or_else(|e| e.into_inner());
        [
            p.a,
            p.b,
            ks.f,
            ks.k,
            p.d_u,
            p.d_v,
            p.dt,
            flow.inlet_velocity,
            flow.viscosity,
            flow.circulation,
        ]
        .into_iter()
        .all(f64::is_finite)
    }

    /// Re-seeds deterministic spatial noise if the active grid washed out and current parameters form patterns.
    pub fn reseed_if_washed_out_entering_pattern(&mut self) -> bool {
        let (a, b, d_u, d_v, dt) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v, p.dt)
        };
        let mode = self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .mode;
        if mode != MorphogenesisKineticsMode::Schnakenberg
            || !is_pattern_forming_regime(a, b, d_u, d_v, dt)
            || !self.is_active_grid_washed_out()
        {
            return false;
        }
        if let Some(sr) = &mut self.step_runner {
            sr.system.kinetics.mode = MorphogenesisKineticsMode::Schnakenberg;
            sr.system.kinetics.a = a;
            sr.system.kinetics.b = b;
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

    /// Applies an [`AtlasTeleportEvent`] to live morphogenesis parameters with CFL recovery and washout re-seeding.
    pub fn apply_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        let was_faulted = self.worker.state() == SimWorkerState::Faulted;
        let switched_from_gray_scott = {
            let mut ks = self
                .kinetics_state
                .write()
                .unwrap_or_else(|e| e.into_inner());
            let was_gs = ks.mode != MorphogenesisKineticsMode::Schnakenberg;
            ks.mode = MorphogenesisKineticsMode::Schnakenberg;
            was_gs
        };
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
            if switched_from_gray_scott {
                p.d_u = p.d_u.clamp(0.1, 5.0);
                p.d_v = p.d_v.clamp(10.0, 200.0);
                p.dt = p.dt.clamp(1e-4, 0.1);
            }
            if (was_faulted || switched_from_gray_scott)
                && ev.x_key != "dt"
                && ev.y_key != "dt"
                && 4.0 * p.d_u.max(p.d_v) * p.dt >= 1.0
            {
                p.dt = (0.2 / p.d_u.max(p.d_v).max(1e-6)).clamp(1e-4, 0.1);
            }
        }
        self.theory.set_mode(MorphogenesisKineticsMode::Schnakenberg);
        if was_faulted || switched_from_gray_scott {
            self.reset_system();
            return;
        }
        let _ = self.reseed_if_washed_out_entering_pattern();
        self.dirty = true;
    }
}
