#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::{grid_u_variance, is_pattern_forming_regime, WASHOUT_VARIANCE_THRESHOLD};
use super::brush::{
    apply_mask_rle_to_kinetics, apply_morpho_cells_to_system, MorphoBrushMode,
};
use super::sculpt::{apply_stroke_to_morpho_system, overlay_cell_mask_colors};
use super::MorphogenesisTool;
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::decode_rle_mask;
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::presets::MorphogenesisKineticsMode;
use math_explorer::biology::diffusion::{MaskedFiniteDifference2D, ReactionCellRole};
use math_explorer::biology::morphogenesis::{
    GrayScottKinetics, ReactionKinetics, SchnakenbergKinetics, SpatialSchnakenbergKinetics,
    TuringSystem,
};
use std::sync::Arc;

/// Snapshot payload bundling active concentration field with `CellMask2D` role bytes.
#[derive(Clone)]
pub struct MorphoSnapshotData {
    /// Flattened active species concentration grid (`u` in Schnakenberg, `v` in Gray-Scott).
    pub u_data: Vec<f64>,
    /// Optional row-major `ReactionCellRole` byte grid when any cell is non-`Free`.
    pub mask_roles: Vec<u8>,
}

/// Zero-allocation kinetics adapter switching cleanly between [`SpatialSchnakenbergKinetics`]
/// and [`GrayScottKinetics`] inside [`TuringSystem`].
#[derive(Clone, Debug)]
pub struct MorphogenesisKineticsAdapter {
    /// Active reaction kinetics family.
    pub mode: MorphogenesisKineticsMode,
    /// Schnakenberg activator feed `a`.
    pub a: f64,
    /// Schnakenberg inhibitor feed `b`.
    pub b: f64,
    /// Gray-Scott substrate feed rate `f`.
    pub f: f64,
    /// Gray-Scott catalyst kill rate `k`.
    pub k: f64,
    /// Preallocated spatial `(a, b)` override buffers for Schnakenberg mode.
    pub spatial: SpatialSchnakenbergKinetics,
}

impl MorphogenesisKineticsAdapter {
    /// Creates an adapter with preallocated spatial Schnakenberg mask capacity.
    #[must_use]
    pub fn new(
        mode: MorphogenesisKineticsMode,
        a: f64,
        b: f64,
        f: f64,
        k: f64,
        grid_size: usize,
    ) -> Self {
        Self {
            mode,
            a,
            b,
            f,
            k,
            spatial: SpatialSchnakenbergKinetics::new(a, b, grid_size),
        }
    }

    /// Returns `true` when at least one cell holds an active spatial `(a, b)` override.
    #[cfg(test)]
    #[inline]
    #[must_use]
    pub fn has_spatial_mask(&self) -> bool {
        self.spatial.has_spatial_mask()
    }

    /// Returns the effective Schnakenberg `(a, b)` rates at linear cell `index`.
    #[inline]
    #[must_use]
    pub fn rates_at(&self, index: usize) -> (f64, f64) {
        if self.spatial.is_overridden(index) {
            self.spatial.rates_at(index)
        } else {
            (self.a, self.b)
        }
    }

    /// Assigns a local `(a, b)` reaction override at `index`.
    pub fn set_cell_rates(&mut self, index: usize, a: f64, b: f64) {
        self.spatial.set_cell_rates(index, a, b);
    }

    /// Clears any local override at `index`.
    pub fn clear_cell_override(&mut self, index: usize) {
        self.spatial.clear_cell_override(index);
    }

    /// Clears all spatial overrides in place.
    pub fn clear_mask(&mut self) {
        self.spatial.clear_mask();
    }
}

impl ReactionKinetics<2> for MorphogenesisKineticsAdapter {
    #[inline]
    fn reaction(&self, concentrations: [f64; 2]) -> [f64; 2] {
        match self.mode {
            MorphogenesisKineticsMode::Schnakenberg => {
                SchnakenbergKinetics::new(self.a, self.b).reaction(concentrations)
            }
            MorphogenesisKineticsMode::GrayScott => {
                GrayScottKinetics::new(self.f, self.k).reaction(concentrations)
            }
        }
    }

    #[inline]
    fn reaction_at(&self, index: usize, concentrations: [f64; 2]) -> [f64; 2] {
        match self.mode {
            MorphogenesisKineticsMode::Schnakenberg => {
                let (rate_a, rate_b) = self.rates_at(index);
                SchnakenbergKinetics::new(rate_a, rate_b).reaction(concentrations)
            }
            MorphogenesisKineticsMode::GrayScott => {
                GrayScottKinetics::new(self.f, self.k).reaction(concentrations)
            }
        }
    }
}

/// Shared Gray-Scott mode and `(f, k)` parameters paired with [`MorphogenesisSharedParams`].
#[derive(Clone, Copy, Debug)]
pub struct MorphogenesisKineticsState {
    /// Selected kinetics regime.
    pub mode: MorphogenesisKineticsMode,
    /// Gray-Scott substrate feed rate `f`.
    pub f: f64,
    /// Gray-Scott autocatalyst kill rate `k`.
    pub k: f64,
}

impl Default for MorphogenesisKineticsState {
    fn default() -> Self {
        let gs = GrayScottKinetics::mitosis();
        Self {
            mode: MorphogenesisKineticsMode::Schnakenberg,
            f: gs.f,
            k: gs.k,
        }
    }
}

#[allow(missing_docs)]
pub struct MorphogenesisSharedParams {
    pub a: f64,
    pub b: f64,
    pub d_u: f64,
    pub d_v: f64,
    pub dt: f64,
    pub high_contrast: bool,
    pub mask_rle: Option<String>,
}

/// Multi-kinetics worker runner supporting [`SchnakenbergKinetics`], [`GrayScottKinetics`],
/// and zero-allocation [`MaskedFiniteDifference2D`] microfluidic field sculpting.
pub struct MorphogenesisKineticsRunner {
    pub(crate) system: TuringSystem<2, MorphogenesisKineticsAdapter, MaskedFiniteDifference2D>,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    pub(crate) kinetics_state: Arc<std::sync::RwLock<MorphogenesisKineticsState>>,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl FiniteStateGuard for MorphogenesisKineticsRunner {
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

impl MorphogenesisKineticsRunner {
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

    fn sync_system_params(&mut self) -> Option<f64> {
        let (a, b, d_u, d_v, dt) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v, p.dt)
        };
        let (mode, f, k) = {
            let ks = self
                .kinetics_state
                .read()
                .unwrap_or_else(|e| e.into_inner());
            (ks.mode, ks.f, ks.k)
        };
        if !a.is_finite()
            || !b.is_finite()
            || !f.is_finite()
            || !k.is_finite()
            || !d_u.is_finite()
            || !d_v.is_finite()
        {
            return None;
        }
        self.system.kinetics.mode = mode;
        self.system.kinetics.a = a;
        self.system.kinetics.b = b;
        self.system.kinetics.spatial.a = a;
        self.system.kinetics.spatial.b = b;
        self.system.kinetics.f = f;
        self.system.kinetics.k = k;
        self.system.diffusion_coeffs[0] = d_u;
        self.system.diffusion_coeffs[1] = d_v;
        if mode == MorphogenesisKineticsMode::Schnakenberg
            && dt.is_finite()
            && dt > 0.0
            && is_pattern_forming_regime(a, b, d_u, d_v, dt)
        {
            let var_u = grid_u_variance(self.system.u());
            if var_u.is_finite() && var_u < WASHOUT_VARIANCE_THRESHOLD {
                MorphogenesisTool::initialize_adapter_system(
                    &mut self.system,
                    self.width,
                    self.height,
                );
                self.enforce_mask_concentrations();
            }
        }
        (dt.is_finite() && dt > 0.0).then_some(dt)
    }
    fn reset_with_optional_kinetics_mask(&mut self, reload_kinetics_mask: bool) {
        let _ = self.sync_system_params();
        if reload_kinetics_mask {
            let mask_rle = self
                .params
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .mask_rle
                .clone();
            apply_mask_rle_to_kinetics(
                &mut self.system.kinetics,
                self.width,
                self.height,
                mask_rle.as_deref(),
            );
        }
        MorphogenesisTool::initialize_adapter_system(&mut self.system, self.width, self.height);
        self.enforce_mask_concentrations();
    }
}

impl SimulationRunner for MorphogenesisKineticsRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Reset => self.reset_with_optional_kinetics_mask(true),
            SimCommand::Custom(ref s) if s == "reinit" => {
                self.reset_with_optional_kinetics_mask(false);
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
            SimCommand::ApplyBrushCells { ref cells, mode } => {
                apply_morpho_cells_to_system(
                    &mut self.system,
                    self.width,
                    self.height,
                    cells,
                    MorphoBrushMode::from_code(mode),
                );
                self.enforce_mask_concentrations();
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
            SimCommand::Custom(ref s) if s == "clear_kinetics_mask" => {
                self.system.kinetics.clear_mask();
            }
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
        (!FiniteStateGuard::is_finite_state(self)).then(|| {
            DiagnosticReport::new(
                0.0,
                0,
                "concentration",
                "Morphogenesis reaction-diffusion diverged: non-finite concentration encountered.",
            )
        })
    }

    fn step(&mut self) {
        if !FiniteStateGuard::is_finite_state(self) {
            return;
        }
        if let Some(dt) = self.sync_system_params() {
            self.enforce_mask_concentrations();
            self.system.step(dt);
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let high_contrast = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .high_contrast;
        let mode = self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .mode;
        let field_vec = match mode {
            MorphogenesisKineticsMode::Schnakenberg => self.system.u().to_vec(),
            MorphogenesisKineticsMode::GrayScott => self.system.v().to_vec(),
        };
        let mut image = MorphogenesisTool::plot_concentration_with_contrast(
            &field_vec,
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
                u_data: field_vec,
                mask_roles,
            })),
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        match self.system.kinetics.mode {
            MorphogenesisKineticsMode::Schnakenberg => 1,
            MorphogenesisKineticsMode::GrayScott => 4,
        }
    }
}
