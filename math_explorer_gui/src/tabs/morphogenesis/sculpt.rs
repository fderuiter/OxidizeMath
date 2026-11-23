//! Microfluidic cell-role brush rasterization, RLE synchronization, and high-contrast
//! mask overlays for [`super::MorphogenesisTool`] and [`super::runner::MorphogenesisKineticsRunner`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{
    decode_rle_mask, encode_rle_mask, rasterize_disk_stroke, BrushInteractionMode, BrushMode,
    BrushStroke, SpatialBrushKit,
};
use eframe::egui::{self, Color32, ColorImage};
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::biology::diffusion::{CellMask2D, MaskedFiniteDifference2D, ReactionCellRole};
use math_explorer::biology::morphogenesis::{ReactionKinetics, TuringSystem};

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

pub(crate) fn enforce_system_mask<K: ReactionKinetics<2>>(
    system: &mut TuringSystem<2, K, MaskedFiniteDifference2D>,
    width: usize,
    height: usize,
) {
    if !system.diffusion.mask.has_active_cells() {
        return;
    }
    let emit_u = system.diffusion.mask.emitter_u_level;
    let emit_v = system.diffusion.mask.emitter_v_level;
    for idx in 0..(width * height) {
        match ReactionCellRole::from_u8(system.diffusion.mask.roles()[idx]) {
            ReactionCellRole::ZeroFluxWall | ReactionCellRole::Sink => {
                system.u_mut()[idx] = 0.0;
                system.v_mut()[idx] = 0.0;
            }
            ReactionCellRole::EmitterU => system.u_mut()[idx] = emit_u,
            ReactionCellRole::EmitterV => system.v_mut()[idx] = emit_v,
            ReactionCellRole::Free => {}
        }
    }
}

pub(crate) fn apply_sculpt_sim_command<K: ReactionKinetics<2>>(
    system: &mut TuringSystem<2, K, MaskedFiniteDifference2D>,
    width: usize,
    height: usize,
    cmd: &crate::async_sim::SimCommand,
) -> bool {
    match cmd {
        crate::async_sim::SimCommand::ApplyBrush {
            cx,
            cy,
            r,
            is_obstacle,
        } => {
            let role = if *is_obstacle {
                ReactionCellRole::ZeroFluxWall
            } else {
                ReactionCellRole::Free
            };
            system.diffusion.mask.paint_disk(*cx, *cy, *r, role);
            enforce_system_mask(system, width, height);
            true
        }
        crate::async_sim::SimCommand::ApplyBrushStroke(ref stroke) => {
            apply_stroke_to_morpho_system(system, width, height, stroke);
            true
        }
        crate::async_sim::SimCommand::SyncSculptState(ref payload) => {
            if payload.rle_mask.is_empty() {
                system.diffusion.mask.clear();
            } else if let Ok(bytes) = decode_rle_mask(&payload.rle_mask, width * height) {
                let _ = system.diffusion.mask.set_roles_slice(&bytes);
            }
            enforce_system_mask(system, width, height);
            true
        }
        crate::async_sim::SimCommand::ClearObstacles => {
            system.diffusion.mask.clear();
            true
        }
        _ => false,
    }
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

pub(crate) fn build_maze_rle(width: usize, height: usize) -> Vec<[u32; 2]> {
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
) -> Option<&'static str> {
    let mut custom_cmd = None;
    ui.separator();
    ui.label("Microfluidic Field Sculpting");
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
        ui.selectable_value(&mut brush_kit.mode, BrushMode::ZeroFluxWall, "Wall");
        ui.selectable_value(&mut brush_kit.mode, BrushMode::EmitterU, "u Emitter");
        ui.selectable_value(&mut brush_kit.mode, BrushMode::EmitterV, "v Emitter");
        ui.selectable_value(&mut brush_kit.mode, BrushMode::Sink, "Sink");
        ui.selectable_value(&mut brush_kit.mode, BrushMode::Erase, "Erase");
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
    /// Dispatches a [`BrushStroke`] to UI-side RLE state, local twin runner, and background worker.
    pub fn dispatch_brush_stroke(&mut self, stroke: BrushStroke, ctx: Option<&egui::Context>) {
        {
            let mut rle = self.mask_rle.write().unwrap_or_else(|e| e.into_inner());
            record_morpho_stroke_rle(&mut rle, self.width, self.height, &stroke);
        }
        apply_stroke_to_morpho_system(
            &mut self.local_runner.system,
            self.width,
            self.height,
            &stroke,
        );
        if let Some(ref mut twin) = self.local_runner.twin {
            let dt = self.params.read().unwrap_or_else(|e| e.into_inner()).dt;
            let ub = super::counterfactual::active_morphogen_slice(&self.local_runner.system);
            twin.observe_lockstep(self.local_runner.step_count, dt, ub);
        }
        self.sync_branch_engine_from_local();
        let _ = self
            .worker
            .send_command(crate::async_sim::SimCommand::ApplyBrushStroke(stroke));
        self.stepped_pixels = None;
        self.step_runner = None;
        self.dirty = true;
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    pub(crate) fn apply_sculpt_quick_action(&mut self, cmd: &str, ctx: &egui::Context) {
        match cmd {
            "stamp_maze" => {
                *self.mask_rle.write().unwrap_or_else(|e| e.into_inner()) =
                    build_maze_rle(self.width, self.height);
                self.local_runner
                    .system
                    .diffusion
                    .mask
                    .stamp_microfluidic_maze();
                self.local_runner.enforce_mask_concentrations();
                self.sync_branch_engine_from_local();
                let _ = self
                    .worker
                    .send_command(crate::async_sim::SimCommand::Custom("stamp_maze".into()));
                self.stepped_pixels = None;
                self.step_runner = None;
                self.dirty = true;
                ctx.request_repaint();
            }
            "clear_mask" => {
                self.mask_rle
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .clear();
                self.local_runner.system.diffusion.mask.clear();
                let _ = self
                    .worker
                    .send_command(crate::async_sim::SimCommand::ClearObstacles);
                self.stepped_pixels = None;
                self.step_runner = None;
                self.dirty = true;
                ctx.request_repaint();
            }
            _ => {}
        }
    }
}
