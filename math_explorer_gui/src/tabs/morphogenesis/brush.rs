#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::runner::MorphogenesisKineticsAdapter;
use super::MorphogenesisTool;
use crate::async_sim::SimCommand;
use crate::framework::{
    decode_u8_grid, draw_brush_palette_controls, encode_u8_grid, RasterBrushStroke,
};
use crate::presets::MorphogenesisKineticsMode;
use eframe::egui;
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::{
    ReactionKinetics, SpatialSchnakenbergKinetics, TuringSystem,
};

/// Schnakenberg `(a, b)` kinetic rates stamped by [`MorphoBrushMode::StripeMask`].
pub const STRIPE_MASK_RATES: (f64, f64) = (0.14, 0.86);
/// Schnakenberg `(a, b)` kinetic rates stamped by [`MorphoBrushMode::SpotMask`].
pub const SPOT_MASK_RATES: (f64, f64) = (0.08, 1.12);

/// Interactive chemical perturbation and spatial reaction-rate brush modes for [`MorphogenesisTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MorphoBrushMode {
    /// Injects autocatalytic activator `u` concentration seeds.
    #[default]
    ActivatorSeed,
    /// Injects long-range inhibitor `v` concentration perturbations.
    InhibitorSeed,
    /// Stamps spatial stripe-regime `(a, b)` overrides onto `SpatialSchnakenbergKinetics`.
    StripeMask,
    /// Stamps spatial spot-regime `(a, b)` overrides onto `SpatialSchnakenbergKinetics`.
    SpotMask,
    /// Clears spatial kinetic overrides and equilibrates touched cells.
    EraseMask,
}

impl MorphoBrushMode {
    /// All selectable morphogenesis brush modes in UI display order.
    pub const ALL: [Self; 5] = [
        Self::ActivatorSeed,
        Self::InhibitorSeed,
        Self::StripeMask,
        Self::SpotMask,
        Self::EraseMask,
    ];

    /// User-facing label for side-panel controls.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::ActivatorSeed => "Seed Activator (u+)",
            Self::InhibitorSeed => "Seed Inhibitor (v+)",
            Self::StripeMask => "Stripe Kinetics Mask (a, b)",
            Self::SpotMask => "Spot Kinetics Mask (a, b)",
            Self::EraseMask => "Clear Mask / Equilibrate",
        }
    }

    /// Wire command byte passed via [`SimCommand::ApplyBrushCells`].
    #[must_use]
    pub const fn to_code(self) -> u8 {
        match self {
            Self::EraseMask => 0,
            Self::ActivatorSeed => 1,
            Self::InhibitorSeed => 2,
            Self::StripeMask => 3,
            Self::SpotMask => 4,
        }
    }

    /// Decodes a wire command byte back to [`MorphoBrushMode`].
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        match code {
            1 => Self::ActivatorSeed,
            2 => Self::InhibitorSeed,
            3 => Self::StripeMask,
            4 => Self::SpotMask,
            _ => Self::EraseMask,
        }
    }
}

/// Shared spatial kinetic mask interface implemented by [`SpatialSchnakenbergKinetics`]
/// and [`MorphogenesisKineticsAdapter`].
pub trait MorphoSpatialKinetics: ReactionKinetics<2> {
    /// Clears all cell-level kinetic overrides.
    fn clear_spatial_mask(&mut self);
    /// Clears any kinetic override at linear `index`.
    fn clear_cell_rate_override(&mut self, index: usize);
    /// Assigns a local `(a, b)` kinetic override at `index`.
    fn set_cell_rate_override(&mut self, index: usize, a: f64, b: f64);
    /// Returns the `(u_eq, v_eq)` steady state for [`MorphoBrushMode::EraseMask`].
    fn erase_equilibrium(&self) -> (f64, f64);
    /// Returns `(delta, ceiling)` for chemical perturbation brush modes.
    fn seed_perturbation(&self) -> (f64, f64);
}

impl MorphoSpatialKinetics for SpatialSchnakenbergKinetics {
    fn clear_spatial_mask(&mut self) {
        self.clear_mask();
    }

    fn clear_cell_rate_override(&mut self, index: usize) {
        self.clear_cell_override(index);
    }

    fn set_cell_rate_override(&mut self, index: usize, a: f64, b: f64) {
        self.set_cell_rates(index, a, b);
    }

    fn erase_equilibrium(&self) -> (f64, f64) {
        let u_eq = (self.a + self.b).max(1e-3);
        (u_eq, self.b / (u_eq * u_eq))
    }

    fn seed_perturbation(&self) -> (f64, f64) {
        (0.75, 8.0)
    }
}

impl MorphoSpatialKinetics for MorphogenesisKineticsAdapter {
    fn clear_spatial_mask(&mut self) {
        self.clear_mask();
    }

    fn clear_cell_rate_override(&mut self, index: usize) {
        self.clear_cell_override(index);
    }

    fn set_cell_rate_override(&mut self, index: usize, a: f64, b: f64) {
        self.set_cell_rates(index, a, b);
    }

    fn erase_equilibrium(&self) -> (f64, f64) {
        match self.mode {
            MorphogenesisKineticsMode::Schnakenberg => {
                let u_eq = (self.a + self.b).max(1e-3);
                (u_eq, self.b / (u_eq * u_eq))
            }
            MorphogenesisKineticsMode::GrayScott => (1.0, 0.0),
        }
    }

    fn seed_perturbation(&self) -> (f64, f64) {
        match self.mode {
            MorphogenesisKineticsMode::Schnakenberg => (0.75, 8.0),
            MorphogenesisKineticsMode::GrayScott => (0.25, 1.0),
        }
    }
}

/// Applies an RLE-encoded spatial mask (`0` = baseline, `1` = stripe, `2` = spot) onto `kinetics`.
pub fn apply_mask_rle_to_kinetics<K: MorphoSpatialKinetics>(
    kinetics: &mut K,
    width: usize,
    height: usize,
    mask_rle: Option<&str>,
) {
    kinetics.clear_spatial_mask();
    let Some(rle) = mask_rle else {
        return;
    };
    let Ok(decoded) = decode_u8_grid(rle, width, height, 2) else {
        return;
    };
    for (idx, sym) in decoded.into_iter().enumerate() {
        match sym {
            1 => kinetics.set_cell_rate_override(idx, STRIPE_MASK_RATES.0, STRIPE_MASK_RATES.1),
            2 => kinetics.set_cell_rate_override(idx, SPOT_MASK_RATES.0, SPOT_MASK_RATES.1),
            _ => {}
        }
    }
}

/// Applies a [`MorphoBrushMode`] cell batch directly to `TuringSystem` slices and kinetic masks.
pub fn apply_morpho_cells_to_system<K: MorphoSpatialKinetics>(
    system: &mut TuringSystem<2, K, FiniteDifference2D>,
    width: usize,
    height: usize,
    cells: &[(usize, usize)],
    mode: MorphoBrushMode,
) {
    let total = width * height;
    let (seed_delta, ceiling) = system.kinetics.seed_perturbation();
    for &(x, y) in cells {
        if x >= width || y >= height {
            continue;
        }
        let idx = y * width + x;
        if idx >= total {
            continue;
        }
        match mode {
            MorphoBrushMode::ActivatorSeed => {
                let next_u = (system.u()[idx] + seed_delta).clamp(0.0, ceiling);
                system.u_mut()[idx] = next_u;
            }
            MorphoBrushMode::InhibitorSeed => {
                let next_v = (system.v()[idx] + seed_delta).clamp(0.0, ceiling);
                system.v_mut()[idx] = next_v;
            }
            MorphoBrushMode::StripeMask => {
                system.kinetics.set_cell_rate_override(
                    idx,
                    STRIPE_MASK_RATES.0,
                    STRIPE_MASK_RATES.1,
                );
                system.u_mut()[idx] = (system.u()[idx] + 0.15).clamp(0.0, ceiling);
            }
            MorphoBrushMode::SpotMask => {
                system
                    .kinetics
                    .set_cell_rate_override(idx, SPOT_MASK_RATES.0, SPOT_MASK_RATES.1);
                system.u_mut()[idx] = (system.u()[idx] + 0.25).clamp(0.0, ceiling);
            }
            MorphoBrushMode::EraseMask => {
                system.kinetics.clear_cell_rate_override(idx);
                let (u_eq, v_eq) = system.kinetics.erase_equilibrium();
                system.u_mut()[idx] = u_eq;
                system.v_mut()[idx] = v_eq;
            }
        }
    }
}

/// Dispatches `clear_mask`, `ApplyBrushCells`, and `ApplyBrush` worker commands onto `system`.
pub(crate) fn apply_brush_sim_command<K: MorphoSpatialKinetics>(
    system: &mut TuringSystem<2, K, FiniteDifference2D>,
    width: usize,
    height: usize,
    cmd: &SimCommand,
) {
    match cmd {
        SimCommand::Custom(s) if s == "clear_mask" => {
            system.kinetics.clear_spatial_mask();
        }
        SimCommand::ApplyBrushCells { cells, mode } => {
            apply_morpho_cells_to_system(system, width, height, cells, MorphoBrushMode::from_code(*mode));
        }
        SimCommand::ApplyBrush {
            cx,
            cy,
            r,
            is_obstacle,
        } => {
            let cells = crate::framework::generate_brush_cells(
                crate::framework::BrushStencil::Circle,
                crate::framework::SymmetryMode::None,
                ((*cx).max(0) as usize, (*cy).max(0) as usize),
                ((*cx).max(0) as usize, (*cy).max(0) as usize),
                (*r).max(0) as usize,
                width,
                height,
            );
            let mode = if *is_obstacle {
                MorphoBrushMode::ActivatorSeed
            } else {
                MorphoBrushMode::EraseMask
            };
            apply_morpho_cells_to_system(system, width, height, &cells, mode);
        }
        _ => {}
    }
}

impl MorphogenesisTool {
    /// Applies a [`RasterBrushStroke`] to the local spatial mask mirror and dispatches to the worker.
    pub fn apply_brush_stroke(&mut self, stroke: RasterBrushStroke, ctx: Option<&egui::Context>) {
        let effective_mode = if stroke.secondary {
            MorphoBrushMode::EraseMask
        } else {
            self.brush_mode
        };
        if self.mask_cells.len() != self.width * self.height {
            self.mask_cells.resize(self.width * self.height, 0);
        }
        let maybe_sym = match effective_mode {
            MorphoBrushMode::StripeMask => Some(1u8),
            MorphoBrushMode::SpotMask => Some(2u8),
            MorphoBrushMode::EraseMask => Some(0u8),
            MorphoBrushMode::ActivatorSeed | MorphoBrushMode::InhibitorSeed => None,
        };
        if let Some(sym) = maybe_sym {
            for &(x, y) in &stroke.cells {
                if x < self.width && y < self.height {
                    self.mask_cells[y * self.width + x] = sym;
                }
            }
            self.sync_mask_rle_from_cells();
        }
        let _ = self.worker.send_command(SimCommand::ApplyBrushCells {
            cells: stroke.cells,
            mode: effective_mode.to_code(),
        });
        self.step_runner = None;
        self.dirty = true;
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    pub(crate) fn sync_mask_rle_from_cells(&mut self) {
        let next_rle = if self.mask_cells.iter().any(|&s| s != 0) {
            encode_u8_grid(self.width, self.height, &self.mask_cells, 2).ok()
        } else {
            None
        };
        self.params
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .mask_rle = next_rle;
    }

    /// Clears all spatial `(a, b)` reaction mask overrides on both UI and worker threads.
    pub fn clear_spatial_mask(&mut self, ctx: Option<&egui::Context>) {
        self.mask_cells.fill(0);
        self.params
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .mask_rle = None;
        let _ = self
            .worker
            .send_command(SimCommand::Custom("clear_mask".to_string()));
        self.step_runner = None;
        self.dirty = true;
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    pub(crate) fn draw_brush_section(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.separator();
        ui.label("Chemical & Mask Brush");
        egui::ComboBox::from_id_salt("morpho_brush_mode")
            .selected_text(self.brush_mode.label())
            .show_ui(ui, |ui| {
                for mode in MorphoBrushMode::ALL {
                    ui.selectable_value(&mut self.brush_mode, mode, mode.label());
                }
            });
        draw_brush_palette_controls(ui, "morphogenesis", &mut self.brush_state, 12);
        if ui.button("Clear Spatial Mask").clicked() {
            self.clear_spatial_mask(Some(ctx));
        }
    }
}

