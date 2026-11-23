//! Background simulation runner for dual-kinetics Turing, Gray-Scott, and masked
//! microfluidic reaction-diffusion morphogenesis.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::cartographer::{grid_u_variance, is_pattern_forming_regime, WASHOUT_VARIANCE_THRESHOLD};
use super::sculpt::{apply_stroke_to_morpho_system, overlay_cell_mask_colors};
use super::state::{
    apply_chemical_brush, ensure_nucleated_for_kinetics, initialize_for_kinetics,
    initialize_schnakenberg_system, plot_concentration_with_contrast, MorphogenesisKinetics,
    DEFAULT_GRAY_SCOTT_F, DEFAULT_GRAY_SCOTT_K,
};
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{decode_rle_mask, BrushStroke};
use math_explorer::biology::diffusion::{
    FiniteDifference2D, MaskedFiniteDifference2D, ReactionCellRole, SpatialDiffusion,
};
use math_explorer::biology::morphogenesis::{
    FusedEulerSolver, GrayScottKinetics, SchnakenbergKinetics, TuringSystem,
};
use std::sync::Arc;

/// Maximum CFL substeps executed per runner step when `auto_dt` is active.
pub const MAX_CFL_SUBSTEPS: usize = 32;

/// Snapshot payload bundling active concentration field with `CellMask2D` role bytes.
#[derive(Clone, Debug)]
pub struct MorphoSnapshotData {
    /// Flattened species `u` concentration grid.
    pub u_data: Vec<f64>,
    /// Optional row-major `ReactionCellRole` byte grid when any cell is non-`Free`.
    pub mask_roles: Vec<u8>,
}

/// Diffusion stencil capabilities shared between unmasked and masked 2D operators.
pub trait MorphoDiffusion: SpatialDiffusion<2> + Send + 'static {
    /// Returns `true` when at least one cell role is non-`Free`.
    fn has_active_mask(&self) -> bool {
        false
    }

    /// Returns the row-major `ReactionCellRole` byte slice when active.
    fn mask_roles(&self) -> &[u8] {
        &[]
    }

    /// Returns `(emitter_u_level, emitter_v_level)` for active emitter cells.
    fn emitter_levels(&self) -> (f64, f64) {
        (1.8, 1.8)
    }

    /// Synchronizes emitter levels to suit the active reaction kinetics scale.
    fn sync_emitter_levels(&mut self, _kinetics: MorphogenesisKinetics) {}

    /// Clears all mask cells back to `Free`.
    fn clear_mask(&mut self) {}

    /// Loads validated `ReactionCellRole` bytes into the mask buffer.
    fn set_mask_slice(&mut self, _bytes: &[u8]) {}

    /// Stamps the serpentine microfluidic maze onto the mask buffer.
    fn stamp_maze(&mut self) {}

    /// Applies an interactive [`BrushStroke`] to `system`.
    fn apply_morpho_stroke(
        _system: &mut TuringSystem<2, SchnakenbergKinetics, Self>,
        _dims: (usize, usize),
        _stroke: &BrushStroke,
    ) where
        Self: Sized,
    {
    }
}

impl MorphoDiffusion for FiniteDifference2D {}

impl MorphoDiffusion for MaskedFiniteDifference2D {
    fn has_active_mask(&self) -> bool {
        self.mask.has_active_cells()
    }

    fn mask_roles(&self) -> &[u8] {
        self.mask.roles()
    }

    fn emitter_levels(&self) -> (f64, f64) {
        (self.mask.emitter_u_level, self.mask.emitter_v_level)
    }

    fn sync_emitter_levels(&mut self, kinetics: MorphogenesisKinetics) {
        match kinetics {
            MorphogenesisKinetics::Schnakenberg => {
                self.mask.emitter_u_level = 1.8;
                self.mask.emitter_v_level = 1.8;
            }
            MorphogenesisKinetics::GrayScott => {
                self.mask.emitter_u_level = 1.0;
                self.mask.emitter_v_level = 0.5;
            }
        }
    }

    fn clear_mask(&mut self) {
        self.mask.clear();
    }

    fn set_mask_slice(&mut self, bytes: &[u8]) {
        let _ = self.mask.set_roles_slice(bytes);
    }

    fn stamp_maze(&mut self) {
        self.mask.stamp_microfluidic_maze();
    }

    fn apply_morpho_stroke(
        system: &mut TuringSystem<2, SchnakenbergKinetics, Self>,
        dims: (usize, usize),
        stroke: &BrushStroke,
    ) {
        apply_stroke_to_morpho_system(system, dims.0, dims.1, stroke);
    }
}

/// Thread-safe parameter block shared between [`super::MorphogenesisTool`] and [`MorphogenesisRunner`].
#[derive(Clone, Debug)]
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

/// Zero-allocation dual-kinetics simulation runner executing on the background worker.
pub struct MorphogenesisRunner<D: SpatialDiffusion<2> = MaskedFiniteDifference2D> {
    pub(crate) system: TuringSystem<2, SchnakenbergKinetics, D>,
    pub(crate) kinetics: MorphogenesisKinetics,
    pub(crate) gray_scott: GrayScottKinetics,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl<D: MorphoDiffusion> FiniteStateGuard for MorphogenesisRunner<D> {
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

impl<D: MorphoDiffusion> MorphogenesisRunner<D> {
    pub(crate) fn enforce_mask_concentrations(&mut self) {
        self.system.diffusion.sync_emitter_levels(self.kinetics);
        if !self.system.diffusion.has_active_mask() {
            return;
        }
        let (emit_u, emit_v) = self.system.diffusion.emitter_levels();
        let roles_len = self
            .system
            .diffusion
            .mask_roles()
            .len()
            .min(self.width * self.height);
        for idx in 0..roles_len {
            let role = ReactionCellRole::from_u8(self.system.diffusion.mask_roles()[idx]);
            match role {
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

    fn sync_system_params(&mut self) -> Option<(f64, bool)> {
        let p = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if !p.a.is_finite()
            || !p.b.is_finite()
            || !p.f.is_finite()
            || !p.k.is_finite()
            || !p.d_u.is_finite()
            || !p.d_v.is_finite()
            || p.d_u <= 0.0
            || p.d_v <= 0.0
        {
            return None;
        }
        self.system.kinetics.a = p.a;
        self.system.kinetics.b = p.b;
        self.gray_scott.f = p.f;
        self.gray_scott.k = p.k;
        self.system.diffusion_coeffs[0] = p.d_u;
        self.system.diffusion_coeffs[1] = p.d_v;
        if self.kinetics != p.kinetics {
            self.kinetics = p.kinetics;
            initialize_for_kinetics(&mut self.system, self.kinetics, self.width, self.height);
            self.enforce_mask_concentrations();
        } else if self.kinetics == MorphogenesisKinetics::Schnakenberg
            && p.dt.is_finite()
            && p.dt > 0.0
        {
            let eff_dt = if p.auto_dt {
                FusedEulerSolver::cfl_substeps_2d(p.dt, 1.0, p.d_u, p.d_v, MAX_CFL_SUBSTEPS).1
            } else {
                p.dt
            };
            if is_pattern_forming_regime(p.a, p.b, p.d_u, p.d_v, eff_dt) {
                let var_u = grid_u_variance(self.system.u());
                if var_u.is_finite() && var_u < WASHOUT_VARIANCE_THRESHOLD {
                    initialize_schnakenberg_system(&mut self.system, self.width, self.height);
                    self.enforce_mask_concentrations();
                }
            }
        }
        (p.dt.is_finite() && p.dt > 0.0).then_some((p.dt, p.auto_dt))
    }

    #[inline]
    fn step_active_kinetics(&mut self, dt: f64) {
        match self.kinetics {
            MorphogenesisKinetics::Schnakenberg => self.system.step(dt),
            MorphogenesisKinetics::GrayScott => {
                self.system.step_with_kinetics(&self.gray_scott, dt);
            }
        }
    }

    fn process_custom_command(&mut self, name: &str) {
        match name {
            "reinit" => {
                let _ = self.sync_system_params();
                initialize_for_kinetics(&mut self.system, self.kinetics, self.width, self.height);
                self.enforce_mask_concentrations();
            }
            "ensure_nucleated" => {
                let _ = self.sync_system_params();
                ensure_nucleated_for_kinetics(
                    &mut self.system,
                    self.kinetics,
                    self.width,
                    self.height,
                );
                self.enforce_mask_concentrations();
            }
            "clear_mask" => self.system.diffusion.clear_mask(),
            "stamp_maze" => {
                let _ = self.sync_system_params();
                self.system.diffusion.sync_emitter_levels(self.kinetics);
                self.system.diffusion.stamp_maze();
                self.enforce_mask_concentrations();
            }
            _ => {}
        }
    }
}

impl<D: MorphoDiffusion> SimulationRunner for MorphogenesisRunner<D> {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Reset => self.process_custom_command("reinit"),
            SimCommand::Custom(ref s) => self.process_custom_command(s),
            SimCommand::ApplyBrush {
                cx,
                cy,
                r,
                is_obstacle,
            } => {
                let _ = self.sync_system_params();
                apply_chemical_brush(
                    &mut self.system,
                    self.kinetics,
                    (self.width, self.height),
                    (cx, cy, r),
                    is_obstacle,
                );
                self.enforce_mask_concentrations();
            }
            SimCommand::ApplyBrushStroke(ref stroke) => {
                let _ = self.sync_system_params();
                self.system.diffusion.sync_emitter_levels(self.kinetics);
                D::apply_morpho_stroke(&mut self.system, (self.width, self.height), stroke);
            }
            SimCommand::SyncSculptState(ref payload) => {
                if payload.rle_mask.is_empty() {
                    self.system.diffusion.clear_mask();
                } else if let Ok(bytes) =
                    decode_rle_mask(&payload.rle_mask, self.width * self.height)
                {
                    self.system.diffusion.set_mask_slice(&bytes);
                }
                self.enforce_mask_concentrations();
            }
            SimCommand::ClearObstacles => self.system.diffusion.clear_mask(),
            _ => {}
        }
    }

    fn is_finite_state(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        if !FiniteStateGuard::is_finite_state(self) {
            Some(DiagnosticReport::new(
                0.0,
                0,
                "concentration",
                "Reaction-diffusion solver diverged: non-finite concentration encountered.",
            ))
        } else {
            None
        }
    }

    fn step(&mut self) {
        if !FiniteStateGuard::is_finite_state(self) {
            return;
        }
        let Some((dt, auto_dt)) = self.sync_system_params() else {
            return;
        };
        self.enforce_mask_concentrations();
        if auto_dt {
            let [d_u, d_v] = self.system.diffusion_coeffs;
            let (substeps, sub_dt) =
                FusedEulerSolver::cfl_substeps_2d(dt, 1.0, d_u, d_v, MAX_CFL_SUBSTEPS);
            for _ in 0..substeps {
                self.step_active_kinetics(sub_dt);
                if !self.system.u().first().is_none_or(|x| x.is_finite()) {
                    break;
                }
            }
        } else {
            self.step_active_kinetics(dt);
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let high_contrast = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .high_contrast;
        let u_vec = self.system.u().to_vec();
        let mut image =
            plot_concentration_with_contrast(&u_vec, self.width, self.height, high_contrast);
        let mask_roles = if self.system.diffusion.has_active_mask() {
            let roles = self.system.diffusion.mask_roles().to_vec();
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
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        4
    }
}
