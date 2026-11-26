#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Parameter teleportation and panel hosting for [`super::OdeRegimeCartographer`] on [`OdeSolverTool`].

use super::OdeRegimeCartographer;
use crate::async_sim::atlas::{AtlasTeleportEvent, ParameterAtlasController};
use crate::presets::OdePreset;
use crate::tabs::analysis::ode::OdeSolverTool;
use eframe::egui;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

impl OdeSolverTool {
    pub(crate) fn atlas_param_value(&self, key: &str) -> f64 {
        match key {
            "k" => self.param_k,
            "r" => self.param_r,
            "cap_k" => self.param_cap_k,
            "y0" => self.ic_y0,
            "v0" => self.ic_v0,
            "dt" => self.dt,
            _ => {
                if self.custom_coupled {
                    if let Some(pos) = self
                        .coupled_bank
                        .param_names()
                        .iter()
                        .position(|n| n == key)
                    {
                        return self
                            .coupled_bank
                            .param_slice()
                            .get(pos)
                            .copied()
                            .unwrap_or(1.0);
                    }
                }
                self.math_input.active_param_value(key).unwrap_or(0.2)
            }
        }
    }

    fn apply_single_atlas_key(&mut self, key: &str, val: f64) {
        match key {
            "k" => {
                self.param_k = val;
                self.math_input.set_param("k", val);
                self.coupled_bank.set_param("k", val);
            }
            "c" => {
                if self.preset == OdePreset::HarmonicOscillator && val.abs() > 1e-6 {
                    self.preset = OdePreset::Custom;
                    self.custom_coupled = true;
                    self.coupled_bank
                        .set_expressions(&["v".to_string(), "-k * y - c * v".to_string()]);
                    self.coupled_bank.set_param("k", self.param_k);
                }
                self.coupled_bank.set_param("c", val);
            }
            "r" => {
                self.param_r = val;
                self.math_input.set_param("r", val);
                self.coupled_bank.set_param("r", val);
            }
            "cap_k" => {
                self.param_cap_k = val;
                self.math_input.set_param("cap_k", val);
                self.math_input.set_param("K", val);
            }
            "y0" => self.ic_y0 = val,
            "v0" => self.ic_v0 = val,
            "dt" => {
                let min_dt =
                    (self.total_time.max(0.1) / crate::presets::MAX_ODE_STEPS as f64).max(1e-4);
                self.dt = val.clamp(min_dt, 1.0);
            }
            _ => {
                self.coupled_bank.set_param(key, val);
                self.math_input.set_param(key, val);
            }
        }
    }

    /// Teleports active ODE parameters to `ev`, pushing the prior run to `snapshot_manager`.
    pub fn apply_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        self.pending_sweep_run = None;
        let prev_summary = self.last_param_summary.clone();
        let prev_series = self.build_current_trace_series();
        self.snapshot_manager
            .push_history_run(prev_summary, prev_series);
        self.apply_single_atlas_key(ev.x_key, ev.x_val);
        self.apply_single_atlas_key(ev.y_key, ev.y_val);
        self.synced_legacy_params = [self.param_k, self.param_r, self.param_cap_k];
        self.recalculate();
    }

    fn atlas_context_signature(&self, xk: &str, yk: &str) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.preset as u8).hash(&mut h);
        (self.method as u8).hash(&mut h);
        self.custom_coupled.hash(&mut h);
        for (k, v) in [
            ("dt", self.dt),
            ("k", self.param_k),
            ("r", self.param_r),
            ("cap_k", self.param_cap_k),
            ("y0", self.ic_y0),
            ("v0", self.ic_v0),
        ] {
            if k != xk && k != yk {
                v.to_bits().hash(&mut h);
            }
        }
        for v in self.matrix_a.iter().flatten() {
            v.to_bits().hash(&mut h);
        }
        if self.preset == OdePreset::PlanarCustom {
            self.planar_x_input.raw_text.hash(&mut h);
            self.planar_y_input.raw_text.hash(&mut h);
        }
        if self.preset == OdePreset::Custom {
            self.math_input.raw_text.hash(&mut h);
            self.coupled_bank.raw_text(0).hash(&mut h);
            self.coupled_bank.raw_text(1).hash(&mut h);
        }
        h.finish()
    }

    pub(in crate::tabs::analysis::ode) fn render_ode_atlas_panel(
        &mut self,
        ui: &mut egui::Ui,
        changed: &mut bool,
    ) {
        if !self.atlas.is_visible() {
            return;
        }
        ui.add_space(6.0);
        let (xk, yk) = self.atlas.axis_keys();
        let active_x = self.atlas_param_value(xk);
        let active_y = self.atlas_param_value(yk);
        let sig = self.atlas_context_signature(xk, yk);
        let cartographer_seed = OdeRegimeCartographer::from_tool(self, (xk, yk));
        let mut atlas =
            std::mem::replace(&mut self.atlas, ParameterAtlasController::new(Vec::new()));
        let teleport = atlas.show_panel(ui, "ode", active_x, active_y, sig, |_keys| {
            Arc::new(cartographer_seed)
        });
        self.atlas = atlas;
        if let Some(ev) = teleport {
            self.apply_atlas_teleport(ev);
            *changed = true;
        }
    }
}
