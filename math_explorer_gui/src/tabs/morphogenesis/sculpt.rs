//! Microfluidic cell-role brush rasterization, chemical reagent brushing, RLE synchronization,
//! and high-contrast mask overlays for [`super::MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::runner::{MorphogenesisRunner, MorphogenesisSharedParams};
use super::state::initialize_for_kinetics;
use crate::async_sim::{BackgroundWorker, SculptStatePayload, SimCommand, SimWorkerState};
use crate::framework::{
    decode_rle_mask, encode_rle_mask, rasterize_disk_stroke, BrushInteractionMode, BrushMode,
    BrushStroke, SpatialBrushKit,
};
use eframe::egui::{self, Color32, ColorImage};
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::biology::diffusion::{
    CellMask2D, FiniteDifference2D, MaskedFiniteDifference2D, ReactionCellRole,
};
use math_explorer::biology::morphogenesis::{
    DiffusionCoeff, GrayScottKinetics, ReactionKinetics, SchnakenbergKinetics, TuringSystem,
};
use math_explorer::math_kernel::types::{Dimension, StepSize};
use std::sync::Arc;

pub(crate) fn default_morpho_brush_kit() -> SpatialBrushKit {
    SpatialBrushKit::new(BrushMode::ZeroFluxWall, 2.0)
        .with_interaction_mode(BrushInteractionMode::Sculpt)
        .with_radius_bounds(1.0, 10.0, 1.0)
        .with_cursor_step(2.0, 2.0)
}

pub(crate) fn mode_to_cell_role(mode: BrushMode) -> ReactionCellRole {
    match mode {
        BrushMode::ZeroFluxWall | BrushMode::Solid => ReactionCellRole::ZeroFluxWall,
        BrushMode::EmitterU => ReactionCellRole::EmitterU,
        BrushMode::EmitterV => ReactionCellRole::EmitterV,
        BrushMode::Sink => ReactionCellRole::Sink,
        _ => ReactionCellRole::Free,
    }
}

pub(crate) fn apply_stroke_to_morpho_system<K: ReactionKinetics<2>>(
    system: &mut TuringSystem<2, K, MaskedFiniteDifference2D>,
    width: usize,
    height: usize,
    stroke: &BrushStroke,
) {
    let (x0, y0) = SpatialBrushKit::plot_to_grid_y_down(stroke.start, width, height);
    let (x1, y1) = SpatialBrushKit::plot_to_grid_y_down(stroke.end, width, height);
    let r = (stroke.radius.round() as i32).clamp(0, 20);
    let role = mode_to_cell_role(stroke.mode);
    let emit_u = system.diffusion.mask.emitter_u_level;
    let emit_v = system.diffusion.mask.emitter_v_level;

    rasterize_disk_stroke((x0, y0), (x1, y1), r, (width, height), |gx, gy| {
        system.diffusion.mask.set(gx, gy, role);
        let idx = flatten_2d_index(gx, gy, width);
        match role {
            ReactionCellRole::ZeroFluxWall | ReactionCellRole::Sink => {
                system.u_mut()[idx] = 0.0;
                system.v_mut()[idx] = 0.0;
            }
            ReactionCellRole::EmitterU => {
                system.u_mut()[idx] = emit_u;
            }
            ReactionCellRole::EmitterV => {
                system.v_mut()[idx] = emit_v;
            }
            ReactionCellRole::Free => {}
        }
    });
}

pub(crate) fn record_morpho_stroke_rle(
    rle: &mut Vec<[u32; 2]>,
    width: usize,
    height: usize,
    stroke: &BrushStroke,
) {
    let total = width * height;
    let mut bytes = decode_rle_mask(rle, total).unwrap_or_else(|_| vec![0_u8; total]);
    let (x0, y0) = SpatialBrushKit::plot_to_grid_y_down(stroke.start, width, height);
    let (x1, y1) = SpatialBrushKit::plot_to_grid_y_down(stroke.end, width, height);
    let r = (stroke.radius.round() as i32).clamp(0, 20);
    let code = mode_to_cell_role(stroke.mode) as u8;
    rasterize_disk_stroke((x0, y0), (x1, y1), r, (width, height), |gx, gy| {
        bytes[flatten_2d_index(gx, gy, width)] = code;
    });
    *rle = encode_rle_mask(&bytes);
}

/// Generates RLE `[role, run_len]` pairs for the microfluidic serpentine maze on a `width x height` grid.
#[must_use]
pub fn build_maze_rle(width: usize, height: usize) -> Vec<[u32; 2]> {
    let mut mask = CellMask2D::new(width, height);
    mask.stamp_microfluidic_maze();
    encode_rle_mask(mask.roles())
}

pub(crate) fn overlay_cell_mask_colors(
    image: &mut ColorImage,
    mask_bytes: &[u8],
    high_contrast: bool,
) {
    if mask_bytes.len() != image.pixels.len() {
        return;
    }
    let wall_color = Color32::from_gray(if high_contrast { 210 } else { 55 });
    let sink_color = Color32::from_gray(if high_contrast { 145 } else { 25 });
    let emit_u_rgb = if high_contrast {
        heatmap_color_high_contrast(0.95)
    } else {
        heatmap_color(0.95)
    };
    let emit_v_rgb = if high_contrast {
        heatmap_color_high_contrast(0.35)
    } else {
        heatmap_color(0.35)
    };

    for (px, &role_byte) in image.pixels.iter_mut().zip(mask_bytes.iter()) {
        match ReactionCellRole::from_u8(role_byte) {
            ReactionCellRole::Free => {}
            ReactionCellRole::ZeroFluxWall => *px = wall_color,
            ReactionCellRole::Sink => *px = sink_color,
            ReactionCellRole::EmitterU => {
                *px = Color32::from_rgb(emit_u_rgb[0], emit_u_rgb[1], emit_u_rgb[2]);
            }
            ReactionCellRole::EmitterV => {
                *px = Color32::from_rgb(emit_v_rgb[0], emit_v_rgb[1], emit_v_rgb[2]);
            }
        }
    }
}

pub(crate) fn draw_morpho_sculpt_controls(
    ui: &mut egui::Ui,
    brush_kit: &mut SpatialBrushKit,
    reagent_brush: &mut bool,
) -> Option<&'static str> {
    let mut custom_cmd = None;
    ui.separator();
    ui.label("Microfluidic & Reagent Brush");
    ui.horizontal(|ui| {
        ui.selectable_value(
            &mut brush_kit.interaction_mode,
            BrushInteractionMode::Sculpt,
            "🖌 Sculpt",
        );
        ui.selectable_value(
            &mut brush_kit.interaction_mode,
            BrushInteractionMode::Pan,
            "✋ Pan",
        );
    });

    ui.horizontal_wrapped(|ui| {
        for (mode, label) in [
            (BrushMode::ZeroFluxWall, "Wall"),
            (BrushMode::EmitterU, "u Emitter"),
            (BrushMode::EmitterV, "v Emitter"),
            (BrushMode::Sink, "Sink"),
            (BrushMode::Erase, "Erase"),
        ] {
            if ui
                .selectable_label(!*reagent_brush && brush_kit.mode == mode, label)
                .clicked()
            {
                *reagent_brush = false;
                brush_kit.mode = mode;
            }
        }
        if ui.selectable_label(*reagent_brush, "Reagent").clicked() {
            *reagent_brush = true;
        }
    });

    ui.add(egui::Slider::new(&mut brush_kit.radius, 1.0..=10.0).text("Brush Radius"));

    ui.horizontal_wrapped(|ui| {
        if ui.button("🧩 Stamp Maze").clicked() {
            custom_cmd = Some("stamp_maze");
        }
        if ui.button("🧹 Clear Mask").clicked() {
            custom_cmd = Some("clear_mask");
        }
    });
    custom_cmd
}

impl super::MorphogenesisTool {
    pub(crate) fn create_masked_runner(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        mask_rle: &Arc<std::sync::RwLock<Vec<[u32; 2]>>>,
        width: usize,
        height: usize,
    ) -> MorphogenesisRunner {
        let p = params.read().unwrap_or_else(|e| e.into_inner()).clone();
        let kinetics = SchnakenbergKinetics { a: p.a, b: p.b };
        let gray_scott = GrayScottKinetics::new(p.f, p.k);
        let diffusion = FiniteDifference2D::new(
            Dimension(width),
            Dimension(height),
            StepSize(1.0),
            StepSize(1.0),
        )
        .with_mask();
        let mut system = TuringSystem::new_with_kinetics(
            Dimension(width * height),
            DiffusionCoeff(p.d_u),
            DiffusionCoeff(p.d_v),
            kinetics,
            diffusion,
        );
        initialize_for_kinetics(&mut system, p.kinetics, width, height);
        let rle = mask_rle.read().unwrap_or_else(|e| e.into_inner()).clone();
        if !rle.is_empty() {
            if let Ok(bytes) = decode_rle_mask(&rle, width * height) {
                let _ = system.diffusion.mask.set_roles_slice(&bytes);
            }
        }
        let mut runner = MorphogenesisRunner {
            system,
            kinetics: p.kinetics,
            gray_scott,
            params: Arc::clone(params),
            width,
            height,
        };
        runner.enforce_mask_concentrations();
        runner
    }

    pub(crate) fn apply_loaded_dimensions_and_state(
        &mut self,
        next_w: usize,
        next_h: usize,
        prev_had_mask: bool,
        target_state: SimWorkerState,
        next_pending: bool,
    ) {
        if next_w != self.width || next_h != self.height {
            self.width = next_w;
            self.height = next_h;
            let runner =
                Self::create_masked_runner(&self.params, &self.mask_rle, self.width, self.height);
            self.worker = BackgroundWorker::new(runner);
            self.stepped_pixels = None;
            self.step_runner = None;
            self.dirty = true;
        } else {
            self.reset();
            let rle_mask = self
                .mask_rle
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            if prev_had_mask || !rle_mask.is_empty() {
                let _ = self
                    .worker
                    .send_command(SimCommand::SyncSculptState(SculptStatePayload {
                        rle_mask,
                        sparse_potential: Vec::new(),
                    }));
            }
        }
        if target_state == SimWorkerState::Running {
            self.pending_initial_start = false;
            if !self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Start);
            }
        } else {
            self.pending_initial_start = next_pending;
            if self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Pause);
            }
        }
    }

    /// Dispatches a [`BrushStroke`] to UI RLE state and/or the background reaction-diffusion worker.
    pub fn dispatch_brush_stroke(&mut self, stroke: BrushStroke, ctx: Option<&egui::Context>) {
        if self.reagent_brush {
            let (cx, cy) =
                SpatialBrushKit::plot_to_grid_y_down(stroke.end, self.width, self.height);
            let r = (stroke.radius.round() as i32).clamp(1, 20);
            let _ = self.worker.send_command(SimCommand::ApplyBrush {
                cx,
                cy,
                r,
                is_obstacle: stroke.mode != BrushMode::Erase,
            });
        } else {
            {
                let mut rle = self.mask_rle.write().unwrap_or_else(|e| e.into_inner());
                record_morpho_stroke_rle(&mut rle, self.width, self.height, &stroke);
            }
            let _ = self
                .worker
                .send_command(SimCommand::ApplyBrushStroke(stroke));
        }
        self.stepped_pixels = None;
        self.step_runner = None;
        self.pending_brush_frames = 4;
        self.dirty = true;
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }
}
