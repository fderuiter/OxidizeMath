//! Timeline checkpointing, shadow-twin cloning, and theory metadata for [`super::lattice_boltzmann::LbmUnified`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::lattice_boltzmann::{LbmUnified, LBM_HEIGHT, LBM_WIDTH};
use crate::async_sim::unified::{UnifiedCheckpointSlot, UnifiedSimTool};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    BgkCollision, LatticeBoltzmannD2Q9,
};
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

impl FiniteStateGuard for LbmUnified {
    fn is_finite_state(&self) -> bool {
        if self.diverged {
            return false;
        }
        self.solver.state.velocity_x().iter().all(|&v| v.is_finite())
            && self.solver.state.velocity_y().iter().all(|&v| v.is_finite())
            && self.solver.state.density().iter().all(|&v| v.is_finite())
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.diagnostic_report.as_ref()
    }

    fn diverged(&self) -> bool {
        self.diverged || !FiniteStateGuard::is_finite_state(self)
    }

    fn reset_parameters(&mut self) {
        self.diverged = false;
        self.diagnostic_report = None;
        self.step_count = 0;
    }
}

impl Clone for LbmUnified {
    fn clone(&self) -> Self {
        Self {
            solver: self.solver.clone(),
            vorticity_buf: self.vorticity_buf.clone(),
            latest_aero: self.latest_aero,
            stamped_airfoil: self.stamped_airfoil,
            active_profile_param: self.active_profile_param,
            challenge_core_locked: self.challenge_core_locked,
            display_vorticity: self.display_vorticity,
            diverged: self.diverged,
            diagnostic_report: self.diagnostic_report.clone(),
            step_count: self.step_count,
        }
    }
}

impl LbmUnified {
    pub(super) fn build_checkpoint_template(&self) -> UnifiedCheckpointSlot {
        UnifiedCheckpointSlot {
            scalars: [0.0; 8],
            field: vec![0.0; self.solver.state.checkpoint_field_len()],
            mask: vec![0; LBM_WIDTH * LBM_HEIGHT],
        }
    }

    pub(super) fn write_checkpoint_slot(&self, slot: &mut UnifiedCheckpointSlot) {
        let mut s4 = [0.0; 4];
        self.solver
            .state
            .export_checkpoint(&mut s4, &mut slot.field, &mut slot.mask);
        slot.scalars[0..4].copy_from_slice(&s4);
        slot.scalars[4] = self.step_count as f64;
        slot.scalars[5] = self.active_profile_param;
        if let Some((code, aoa)) = self.stamped_airfoil {
            slot.scalars[6] = f64::from(code);
            slot.scalars[7] = aoa;
        } else {
            slot.scalars[6] = -1.0;
            slot.scalars[7] = 0.0;
        }
    }

    pub(super) fn restore_checkpoint_slot(
        &mut self,
        slot: &UnifiedCheckpointSlot,
        params: &HashMap<String, f64>,
    ) {
        let mut s4 = [0.0; 4];
        s4.copy_from_slice(&slot.scalars[0..4]);
        self.solver
            .state
            .restore_checkpoint(&s4, &slot.field, &slot.mask);
        self.step_count = slot.scalars[4].max(0.0) as usize;
        self.active_profile_param = slot.scalars[5];
        self.stamped_airfoil = if slot.scalars[6] >= 0.0 {
            Some((slot.scalars[6].round() as u16, slot.scalars[7]))
        } else {
            None
        };
        let viscosity = params.get("viscosity").copied().unwrap_or(0.02);
        if viscosity.is_finite() && viscosity > 0.0 {
            self.solver.collision_model.tau = 3.0 * viscosity + 0.5;
        }
        self.diverged = false;
        self.diagnostic_report = None;
        self.refresh_diagnostics_pub();
    }

    pub(super) fn copy_scalar_field(&self, out: &mut [f64]) -> usize {
        let w = self.solver.width();
        let h = self.solver.height();
        let total = (w * h).min(out.len());
        for (idx, cell) in out.iter_mut().enumerate().take(total) {
            let x = idx % w;
            let y = idx / w;
            let solver_y = h - 1 - y;
            if self.solver.is_obstacle(x, solver_y) {
                *cell = 0.0;
            } else if self.display_vorticity {
                let v = self.vorticity_buf[flatten_2d_index(x, solver_y, w)];
                *cell = if v.is_finite() { v } else { 0.0 };
            } else {
                let v = self.solver.get_velocity_magnitude(x, solver_y);
                *cell = if v.is_finite() { v } else { 0.0 };
            }
        }
        total
    }
}

impl TheoryDescribable for LbmUnified {
    fn theory_description(&self) -> String {
        LatticeBoltzmannD2Q9::<BgkCollision>::new(1, 1, 1.0).theory_description()
    }

    fn phonetic_description(&self) -> String {
        LatticeBoltzmannD2Q9::<BgkCollision>::new(1, 1, 1.0).phonetic_description()
    }

    fn theory_citation(&self) -> String {
        LatticeBoltzmannD2Q9::<BgkCollision>::new(1, 1, 1.0).theory_citation()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        LatticeBoltzmannD2Q9::<BgkCollision>::new(1, 1, 1.0).available_descriptions()
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "LatticeBoltzmannTool",
        display_name: "Lattice Boltzmann (Demo)",
        domain: "fluid_dynamics",
        tags: &[],
        build: || Box::new(UnifiedSimTool::<LbmUnified>::new()),
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "LBM Flow Simulation",
    "domain": "fluid_dynamics",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "viscosity": 0.02
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "lattice_boltzmann_starter",
        title: "LBM Flow Simulation",
        description: "f_i(x + e_i·dt, t + dt) = f_i(x, t) - (f_i - f_i^eq)/τ: Mesoscopic BGK collision streaming solver.",
        category: "Computational Fluids",
        icon: "🌊",
        domain: "fluid_dynamics",
        tool_name: Some("LatticeBoltzmannTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
