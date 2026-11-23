//! Microfluidic cell-role brush rasterization, RLE synchronization, high-contrast
//! mask overlays, and coupled LBM / PotentialFlow obstacle sculpting.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::diffusion::CoupledMorphoDiffusion;
use super::runner::MorphogenesisKineticsRunner;
use super::types::{
    MorphogenesisBrushMode, MorphogenesisFlowMode, MorphogenesisFlowState,
    MorphogenesisKineticsMode,
};
use crate::framework::{
    decode_rle_mask, encode_rle_mask, rasterize_disk_stroke, BrushInteractionMode, BrushMode,
    BrushStroke, SpatialBrushKit,
};
use eframe::egui::{self, Color32, ColorImage};
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::biology::diffusion::{CellMask2D, ReactionCellRole};
use math_explorer::biology::morphogenesis::{ReactionKinetics, TuringSystem};
#[cfg(feature = "physics")]
use math_explorer::physics::fluid_dynamics::potential_flow::{
    Doublet, FlowElement, UniformFlow, Vortex,
};

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
    system: &mut TuringSystem<2, K, CoupledMorphoDiffusion>,
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

/// Generates an RLE-encoded microfluidic serpentine chicane maze for a `width x height` lattice.
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
    /// Dispatches a [`BrushStroke`] to both UI-side RLE state and the background reaction-diffusion worker.
    pub fn dispatch_brush_stroke(&mut self, stroke: BrushStroke, ctx: Option<&egui::Context>) {
        {
            let mut rle = self.mask_rle.write().unwrap_or_else(|e| e.into_inner());
            record_morpho_stroke_rle(&mut rle, self.width, self.height, &stroke);
        }
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
}

impl MorphogenesisKineticsRunner {
    pub(crate) fn sync_mask_to_hydrodynamics(&mut self) {
        self.system.diffusion.sync_obstacles_from_mask();
        #[cfg(feature = "physics")]
        {
            for y in 0..self.height {
                for x in 0..self.width {
                    let is_obs = self.system.diffusion.obstacles()[flatten_2d_index(x, y, self.width)];
                    self.lbm.set_obstacle(x, y, is_obs);
                }
            }
            self.potential_dirty = true;
        }
        let flow = self.read_flow();
        #[cfg(feature = "physics")]
        if flow.flow_mode == MorphogenesisFlowMode::PotentialFlow {
            self.rebuild_potential_elements(flow.inlet_velocity, flow.circulation);
        }
        if flow.flow_mode != MorphogenesisFlowMode::None {
            self.refresh_diagnostics();
        }
    }

    pub(crate) fn seed_default_bluff_obstacle(&mut self) {
        if self.width < 8 || self.height < 8 {
            return;
        }
        let cx = self.width / 4;
        let cy = self.height / 2;
        let radius = (self.height / 12).clamp(2, 6);
        self.system
            .diffusion
            .advection
            .paint_obstacle_disk(cx, cy, radius, true);
        #[cfg(feature = "physics")]
        {
            for y in 0..self.height {
                for x in 0..self.width {
                    let is_obs =
                        self.system.diffusion.obstacles()[flatten_2d_index(x, y, self.width)];
                    self.lbm.set_obstacle(x, y, is_obs);
                }
            }
            self.potential_dirty = true;
            let flow = self.read_flow();
            if flow.flow_mode == MorphogenesisFlowMode::PotentialFlow {
                self.rebuild_potential_elements(flow.inlet_velocity, flow.circulation);
            }
        }
        self.refresh_diagnostics();
    }

    pub(crate) fn paint_obstacle_both(
        &mut self,
        cx: usize,
        cy: usize,
        radius: usize,
        is_obstacle: bool,
    ) {
        self.system
            .diffusion
            .paint_obstacle_disk(cx, cy, radius, is_obstacle);
        self.enforce_mask_concentrations();
        #[cfg(feature = "physics")]
        {
            for y in 0..self.height {
                for x in 0..self.width {
                    let is_obs = self.system.diffusion.obstacles()[flatten_2d_index(x, y, self.width)];
                    self.lbm.set_obstacle(x, y, is_obs);
                }
            }
            self.potential_dirty = true;
        }
    }

    pub(crate) fn apply_reagent_disk(&mut self, cx: usize, cy: usize, radius: usize, inject: bool) {
        let r = radius as isize;
        let y_min = cy.saturating_sub(radius);
        let y_max = (cy + radius).min(self.height.saturating_sub(1));
        let x_min = cx.saturating_sub(radius);
        let x_max = (cx + radius).min(self.width.saturating_sub(1));
        let mode = self.system.kinetics.mode;
        let a = self.system.kinetics.a;
        let b = self.system.kinetics.b;

        for y in y_min..=y_max {
            for x in x_min..=x_max {
                let dx = x as isize - cx as isize;
                let dy = y as isize - cy as isize;
                let idx = flatten_2d_index(x, y, self.width);
                if dx * dx + dy * dy <= r * r && !self.system.diffusion.obstacles()[idx] {
                    let (u_val, v_val) = match (mode, inject) {
                        (MorphogenesisKineticsMode::Schnakenberg, true) => {
                            let u_eq = (a + b).max(0.05);
                            (u_eq + 1.2, b / (u_eq * u_eq))
                        }
                        (MorphogenesisKineticsMode::Schnakenberg, false) => {
                            let u_eq = (a + b).max(0.05);
                            (u_eq, b / (u_eq * u_eq))
                        }
                        (MorphogenesisKineticsMode::GrayScott, true) => (0.50, 0.25),
                        (MorphogenesisKineticsMode::GrayScott, false) => (1.0, 0.0),
                    };
                    self.system.u_mut()[idx] = u_val;
                    self.system.v_mut()[idx] = v_val;
                }
            }
        }
    }

    pub(crate) fn sync_flow_field(&mut self, flow: &MorphogenesisFlowState) {
        self.system.diffusion.flow_active = flow.flow_mode != MorphogenesisFlowMode::None;
        if self.last_flow_mode != flow.flow_mode {
            if flow.flow_mode == MorphogenesisFlowMode::None {
                self.system.diffusion.advection.clear_velocity();
                self.peclet = 0.0;
            } else if !self.system.diffusion.obstacles().iter().any(|&b| b) {
                self.seed_default_bluff_obstacle();
            }
            self.last_flow_mode = flow.flow_mode;
            #[cfg(feature = "physics")]
            {
                self.potential_dirty = true;
            }
        }
        match flow.flow_mode {
            MorphogenesisFlowMode::None => {}
            MorphogenesisFlowMode::LbmVortexWake => {
                #[cfg(feature = "physics")]
                self.step_lbm_coupling(flow.inlet_velocity, flow.viscosity);
                #[cfg(not(feature = "physics"))]
                self.system
                    .diffusion
                    .advection
                    .set_uniform_velocity(flow.inlet_velocity, 0.0);
            }
            MorphogenesisFlowMode::PotentialFlow => {
                #[cfg(feature = "physics")]
                self.rebuild_potential_elements(flow.inlet_velocity, flow.circulation);
                #[cfg(not(feature = "physics"))]
                self.system
                    .diffusion
                    .advection
                    .set_uniform_velocity(flow.inlet_velocity, 0.0);
            }
        }
    }

    #[cfg(feature = "physics")]
    pub(crate) fn rebuild_potential_elements(&mut self, inlet_velocity: f64, circulation: f64) {
        let key = (inlet_velocity.to_bits(), circulation.to_bits());
        if self.last_potential_key == key && !self.potential_dirty {
            return;
        }
        self.potential_flow.clear();
        let cx = (self.width as f64) * 0.35;
        let cy = (self.height as f64) * 0.50;
        let cyl_r = ((self.height as f64) / 10.0).max(2.0);
        let kappa = 2.0 * std::f64::consts::PI * inlet_velocity * cyl_r * cyl_r;
        self.potential_flow
            .add(Box::new(UniformFlow::new(inlet_velocity, 0.0)));
        self.potential_flow
            .add(Box::new(Doublet::new(kappa, cx, cy)));
        self.potential_flow
            .add(Box::new(Vortex::new(circulation, cx, cy)));
        self.last_potential_key = key;
        self.potential_dirty = false;
        self.evaluate_potential_grid();
    }

    #[cfg(feature = "physics")]
    fn evaluate_potential_grid(&mut self) {
        let w = self.width;
        let h = self.height;
        let (vx, vy, obs) = self.system.diffusion.advection.buffers_mut();
        for y in 0..h {
            for x in 0..w {
                let idx = flatten_2d_index(x, y, w);
                if obs[idx] {
                    vx[idx] = 0.0;
                    vy[idx] = 0.0;
                } else {
                    let vel = self.potential_flow.velocity(x as f64 + 0.5, y as f64 + 0.5);
                    vx[idx] = vel.x.clamp(-1.5, 1.5);
                    vy[idx] = vel.y.clamp(-1.5, 1.5);
                }
            }
        }
    }

    #[cfg(feature = "physics")]
    pub(crate) fn step_lbm_coupling(&mut self, inlet_velocity: f64, viscosity: f64) {
        self.lbm.collision_model.tau = (3.0 * viscosity + 0.5).max(0.51);
        let u_in = inlet_velocity.clamp(0.0, 0.22);
        let jet_h = self.height.saturating_sub(2).max(1);
        let jet_y = usize::from(self.height > 2);
        self.lbm.set_inlet(0, jet_y, 1, jet_h, u_in, 0.0);
        self.lbm.step();
        let (vx, vy, obs) = self.system.diffusion.advection.buffers_mut();
        self.lbm.state.copy_macroscopic_row_major(vx, vy, obs);
    }

    pub(crate) fn handle_brush_command(&mut self, cx: i32, cy: i32, r: i32, is_obstacle: bool) {
        if self.width == 0 || self.height == 0 {
            return;
        }
        let gx = cx.clamp(0, self.width as i32 - 1) as usize;
        let gy = cy.clamp(0, self.height as i32 - 1) as usize;
        let radius = r.max(0) as usize;
        let flow = self.read_flow();
        match flow.brush_mode {
            MorphogenesisBrushMode::Obstacle => {
                self.paint_obstacle_both(gx, gy, radius, is_obstacle);
            }
            MorphogenesisBrushMode::EraseObstacle => {
                self.paint_obstacle_both(gx, gy, radius, false);
            }
            MorphogenesisBrushMode::Reagent => {
                self.apply_reagent_disk(gx, gy, radius, is_obstacle);
            }
            MorphogenesisBrushMode::PanZoom => {}
        }
        #[cfg(feature = "physics")]
        if flow.flow_mode == MorphogenesisFlowMode::PotentialFlow && self.potential_dirty {
            self.rebuild_potential_elements(flow.inlet_velocity, flow.circulation);
        }
        self.refresh_diagnostics();
    }

    pub(crate) fn handle_clear_obstacles(&mut self) {
        self.system.diffusion.clear_all_obstacles();
        #[cfg(feature = "physics")]
        {
            self.lbm.clear_obstacles();
            self.potential_dirty = true;
        }
        let flow = self.read_flow();
        #[cfg(feature = "physics")]
        if flow.flow_mode == MorphogenesisFlowMode::PotentialFlow {
            self.rebuild_potential_elements(flow.inlet_velocity, flow.circulation);
        }
        self.refresh_diagnostics();
    }
}
