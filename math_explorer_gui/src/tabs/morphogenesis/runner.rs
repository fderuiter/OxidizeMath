#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::sculpt::{apply_stroke_to_morpho_system, overlay_cell_mask_colors};
use super::MorphogenesisTool;
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::decode_rle_mask;
use math_explorer::biology::diffusion::{
    FiniteDifference2D, MaskedFiniteDifference2D, ReactionCellRole, SpatialDiffusion,
};
use math_explorer::biology::morphogenesis::{SchnakenbergKinetics, TuringSystem};
use std::sync::Arc;

/// Snapshot payload bundling activator field `u` with active `CellMask2D` bytes.
#[derive(Clone)]
pub struct MorphoSnapshotData {
    /// Flattened activator `u` concentration grid.
    pub u_data: Vec<f64>,
    /// Optional row-major `ReactionCellRole` byte grid when any cell is non-`Free`.
    pub mask_roles: Vec<u8>,
}

#[allow(missing_docs)]
pub struct MorphogenesisSharedParams {
    pub a: f64,
    pub b: f64,
    pub d_u: f64,
    pub d_v: f64,
    pub dt: f64,
    pub high_contrast: bool,
}

#[allow(missing_docs)]
pub struct MorphogenesisRunner<D: SpatialDiffusion<2> = FiniteDifference2D> {
    pub(crate) system: TuringSystem<2, SchnakenbergKinetics, D>,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl<D: SpatialDiffusion<2>> FiniteStateGuard for MorphogenesisRunner<D> {
    fn is_finite_state(&self) -> bool {
        self.system.u().iter().all(|&x| x.is_finite())
            && self.system.v().iter().all(|&x| x.is_finite())
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        None
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }
}

impl<D: SpatialDiffusion<2>> MorphogenesisRunner<D> {
    fn sync_system_params(&mut self) -> Option<f64> {
        let (a, b, d_u, d_v, dt) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v, p.dt)
        };
        if !a.is_finite() || !b.is_finite() || !d_u.is_finite() || !d_v.is_finite() {
            return None;
        }
        self.system.kinetics.a = a;
        self.system.kinetics.b = b;
        self.system.diffusion_coeffs[0] = d_u;
        self.system.diffusion_coeffs[1] = d_v;
        (dt.is_finite() && dt > 0.0).then_some(dt)
    }

    fn diagnostic_if_diverged(&self) -> Option<DiagnosticReport> {
        (!FiniteStateGuard::is_finite_state(self)).then(|| {
            DiagnosticReport::new(
                0.0,
                0,
                "concentration",
                "Turing morphogenesis diverged: non-finite concentration encountered.",
            )
        })
    }
}

impl SimulationRunner for MorphogenesisRunner<FiniteDifference2D> {
    fn process_command(&mut self, cmd: SimCommand) {
        if matches!(cmd, SimCommand::Reset)
            || matches!(cmd, SimCommand::Custom(ref s) if s == "reinit")
        {
            let _ = self.sync_system_params();
            MorphogenesisTool::initialize_system(&mut self.system, self.width, self.height);
        }
    }

    fn is_finite_state(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        self.diagnostic_if_diverged()
    }

    fn step(&mut self) {
        if !FiniteStateGuard::is_finite_state(self) {
            return;
        }
        if let Some(dt) = self.sync_system_params() {
            self.system.step(dt);
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let high_contrast = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .high_contrast;
        let u_vec = self.system.u().to_vec();
        let image = MorphogenesisTool::plot_concentration_with_contrast(
            &u_vec,
            self.width,
            self.height,
            high_contrast,
        );
        StateSnapshot {
            width: self.width,
            height: self.height,
            pixels: Arc::new(std::sync::RwLock::new(image.pixels)),
            custom_data: Vec::new(),
            structured_data: Some(Box::new(MorphoSnapshotData {
                u_data: u_vec,
                mask_roles: Vec::new(),
            })),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }
}

impl MorphogenesisRunner<MaskedFiniteDifference2D> {
    pub(crate) fn enforce_mask_concentrations(&mut self) {
        if !self.system.diffusion.mask.has_active_cells() {
            return;
        }
        let emit_u = self.system.diffusion.mask.emitter_u_level;
        let emit_v = self.system.diffusion.mask.emitter_v_level;
        for idx in 0..(self.width * self.height) {
            match ReactionCellRole::from_u8(self.system.diffusion.mask.roles()[idx]) {
                ReactionCellRole::ZeroFluxWall | ReactionCellRole::Sink => {
                    self.system.u_mut()[idx] = 0.0;
                    self.system.v_mut()[idx] = 0.0;
                }
                ReactionCellRole::EmitterU => self.system.u_mut()[idx] = emit_u,
                ReactionCellRole::EmitterV => self.system.v_mut()[idx] = emit_v,
                ReactionCellRole::Free => {}
            }
        }
    }
}

impl SimulationRunner for MorphogenesisRunner<MaskedFiniteDifference2D> {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Reset => {
                let _ = self.sync_system_params();
                MorphogenesisTool::initialize_system(&mut self.system, self.width, self.height);
                self.enforce_mask_concentrations();
            }
            SimCommand::Custom(ref s) if s == "reinit" => {
                let _ = self.sync_system_params();
                MorphogenesisTool::initialize_system(&mut self.system, self.width, self.height);
                self.enforce_mask_concentrations();
            }
            SimCommand::ApplyBrush {
                cx,
                cy,
                r,
                is_obstacle,
            } => {
                let role = if is_obstacle {
                    ReactionCellRole::ZeroFluxWall
                } else {
                    ReactionCellRole::Free
                };
                self.system.diffusion.mask.paint_disk(cx, cy, r, role);
                self.enforce_mask_concentrations();
            }
            SimCommand::ApplyBrushStroke(ref stroke) => {
                apply_stroke_to_morpho_system(&mut self.system, self.width, self.height, stroke);
            }
            SimCommand::SyncSculptState(ref payload) => {
                if payload.rle_mask.is_empty() {
                    self.system.diffusion.mask.clear();
                } else if let Ok(bytes) =
                    decode_rle_mask(&payload.rle_mask, self.width * self.height)
                {
                    let _ = self.system.diffusion.mask.set_roles_slice(&bytes);
                }
                self.enforce_mask_concentrations();
            }
            SimCommand::ClearObstacles => self.system.diffusion.mask.clear(),
            SimCommand::Custom(ref s) if s == "clear_mask" => self.system.diffusion.mask.clear(),
            SimCommand::Custom(ref s) if s == "stamp_maze" => {
                self.system.diffusion.mask.stamp_microfluidic_maze();
                self.enforce_mask_concentrations();
            }
            _ => {}
        }
    }

    fn is_finite_state(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        self.diagnostic_if_diverged()
    }

    fn step(&mut self) {
        if !FiniteStateGuard::is_finite_state(self) {
            return;
        }
        if let Some(dt) = self.sync_system_params() {
            self.system.step(dt);
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let high_contrast = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .high_contrast;
        let u_vec = self.system.u().to_vec();
        let mut image = MorphogenesisTool::plot_concentration_with_contrast(
            &u_vec,
            self.width,
            self.height,
            high_contrast,
        );
        let mask_roles = if self.system.diffusion.mask.has_active_cells() {
            let roles = self.system.diffusion.mask.roles().to_vec();
            overlay_cell_mask_colors(&mut image, &roles, high_contrast);
            roles
        } else {
            Vec::new()
        };
        StateSnapshot {
            width: self.width,
            height: self.height,
            pixels: Arc::new(std::sync::RwLock::new(image.pixels)),
            custom_data: Vec::new(),
            structured_data: Some(Box::new(MorphoSnapshotData {
                u_data: u_vec,
                mask_roles,
            })),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }
}
