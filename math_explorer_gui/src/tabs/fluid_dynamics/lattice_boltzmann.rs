//! Dedicated D2Q9 Lattice Boltzmann Wind Tunnel tool with live bounce-back
//! aerodynamic telemetry (`C_D`, `C_L`, `Re`), URL-shareable obstacle sculpting,
//! and multi-frame vortex-shedding GIF export.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::{BackgroundWorker, SimCommand, StateSnapshot};
use crate::framework::InteractiveTool;
use eframe::egui;
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    BgkCollision, LatticeBoltzmannD2Q9, LatticeState,
};
use scientific_metadata::theory::TheoryDescribable;
use std::sync::{Arc, RwLock};

#[cfg(test)]
#[path = "lattice_boltzmann/legacy_unified.rs"]
mod legacy_unified;
#[cfg(test)]
pub use legacy_unified::LbmUnified;

#[path = "lattice_boltzmann/presets.rs"]
mod presets;

#[path = "lattice_boltzmann/runner.rs"]
pub mod runner;
pub use runner::*;

#[path = "lattice_boltzmann/state.rs"]
pub mod state;
pub use state::*;

#[path = "lattice_boltzmann/ui.rs"]
mod ui;

#[cfg(test)]
#[path = "lattice_boltzmann/tests.rs"]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "lattice_boltzmann/wasm_tests.rs"]
mod wasm_tests;

/// Interactive D2Q9 Lattice Boltzmann Wind Tunnel backed by [`BackgroundWorker<LbmRunner>`].
pub struct LatticeBoltzmannTool {
    pub(crate) viscosity: f64,
    pub(crate) inflow_speed: f64,
    pub(crate) steps_per_frame: usize,
    pub(crate) brush_radius: i32,
    pub(crate) area_budget: usize,
    pub(crate) view_mode: LbmViewMode,
    pub(crate) brush_mode: LbmBrushMode,
    pub(crate) high_contrast: bool,
    pub(crate) obstacles: Vec<bool>,
    pub(crate) mask_generation: u64,
    pub(crate) telemetry: LbmTelemetry,
    pub(crate) cursor_cell: (usize, usize),
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) dirty: bool,
    pub(crate) gif_solver: Option<LatticeBoltzmannD2Q9<BgkCollision>>,
    pub(crate) shared_config: Arc<RwLock<LbmSharedConfig>>,
    pub(crate) shared_lattice: Arc<RwLock<LatticeState<9>>>,
    pub(crate) worker: BackgroundWorker<LbmRunner>,
}

impl Default for LatticeBoltzmannTool {
    fn default() -> Self {
        let obstacles = cylinder_obstacle_grid();
        let initial_cfg = LbmSharedConfig {
            viscosity: DEFAULT_VISCOSITY,
            inflow_speed: DEFAULT_INFLOW_SPEED,
            steps_per_frame: DEFAULT_STEPS_PER_FRAME,
            view_mode: LbmViewMode::Speed,
            high_contrast: false,
            obstacles: obstacles.clone(),
            mask_generation: 0,
        };
        let shared_config = Arc::new(RwLock::new(initial_cfg));
        let shared_lattice = Arc::new(RwLock::new(LatticeState::new(LBM_WIDTH, LBM_HEIGHT)));
        let runner = LbmRunner::new(Arc::clone(&shared_config), Arc::clone(&shared_lattice));
        let worker = BackgroundWorker::new(runner);
        let telemetry = compute_aerodynamics(
            (0.0, 0.0),
            DEFAULT_INFLOW_SPEED,
            DEFAULT_VISCOSITY,
            &obstacles,
            0,
        );
        Self {
            viscosity: DEFAULT_VISCOSITY,
            inflow_speed: DEFAULT_INFLOW_SPEED,
            steps_per_frame: DEFAULT_STEPS_PER_FRAME,
            brush_radius: DEFAULT_BRUSH_RADIUS,
            area_budget: DEFAULT_AREA_BUDGET,
            view_mode: LbmViewMode::Speed,
            brush_mode: LbmBrushMode::Solid,
            high_contrast: false,
            obstacles,
            mask_generation: 0,
            telemetry,
            cursor_cell: (28, 25),
            texture: None,
            dirty: true,
            gif_solver: None,
            shared_config,
            shared_lattice,
            worker,
        }
    }
}

impl LatticeBoltzmannTool {
    pub(crate) fn push_shared_config(&mut self) {
        let mut cfg = self.shared_config.write().unwrap_or_else(|e| e.into_inner());
        cfg.viscosity = self.viscosity;
        cfg.inflow_speed = self.inflow_speed;
        cfg.steps_per_frame = self.steps_per_frame;
        cfg.view_mode = self.view_mode;
        cfg.high_contrast = self.high_contrast;
        self.gif_solver = None;
    }

    pub(crate) fn refresh_idle_telemetry(&mut self) {
        let force = (self.telemetry.drag_force, self.telemetry.lift_force);
        self.telemetry = compute_aerodynamics(
            force,
            self.inflow_speed,
            self.viscosity,
            &self.obstacles,
            self.telemetry.step_count,
        );
    }

    /// Sculpts or erases solid cells around `(cx, cy)` using `brush_radius` and `area_budget`.
    pub fn sculpt_at_cell(&mut self, cx: usize, cy: usize, paint_solid: bool) {
        apply_brush_with_budget(
            &mut self.obstacles,
            cx,
            cy,
            self.brush_radius,
            paint_solid,
            self.area_budget,
        );
        self.sync_obstacles_to_worker();
    }

    pub(crate) fn enforce_current_budget(&mut self) {
        enforce_area_budget(&mut self.obstacles, self.area_budget);
        self.sync_obstacles_to_worker();
    }

    pub(crate) fn apply_obstacle_preset(&mut self, mut grid: Vec<bool>) {
        enforce_area_budget(&mut grid, self.area_budget);
        self.obstacles = grid;
        self.mask_generation = self.mask_generation.wrapping_add(1);
        {
            let mut cfg = self.shared_config.write().unwrap_or_else(|e| e.into_inner());
            cfg.obstacles.clone_from(&self.obstacles);
            cfg.mask_generation = self.mask_generation;
        }
        self.reset_flow_field();
    }

    fn sync_obstacles_to_worker(&mut self) {
        self.mask_generation = self.mask_generation.wrapping_add(1);
        {
            let mut cfg = self.shared_config.write().unwrap_or_else(|e| e.into_inner());
            cfg.obstacles.clone_from(&self.obstacles);
            cfg.mask_generation = self.mask_generation;
        }
        self.gif_solver = None;
        self.refresh_idle_telemetry();
        let _ = self
            .worker
            .send_command(SimCommand::Custom("sync_mask".to_string()));
        self.dirty = true;
    }

    /// Resets fluid populations to uniform inflow equilibrium around the current obstacle mask.
    pub fn reset_flow_field(&mut self) {
        let cfg_arc = Arc::clone(&self.shared_config);
        let lat_arc = Arc::clone(&self.shared_lattice);
        self.worker
            .reset_or_respawn(|| LbmRunner::new(cfg_arc, lat_arc));
        self.gif_solver = None;
        self.telemetry = compute_aerodynamics(
            (0.0, 0.0),
            self.inflow_speed,
            self.viscosity,
            &self.obstacles,
            0,
        );
        self.dirty = true;
    }

    pub(crate) fn reset_parameters_and_flow(&mut self) {
        self.viscosity = DEFAULT_VISCOSITY;
        self.inflow_speed = DEFAULT_INFLOW_SPEED;
        self.steps_per_frame = DEFAULT_STEPS_PER_FRAME;
        self.brush_radius = DEFAULT_BRUSH_RADIUS;
        self.area_budget = DEFAULT_AREA_BUDGET;
        self.push_shared_config();
        self.reset_flow_field();
    }

    fn build_snapshot_solver(&self) -> LatticeBoltzmannD2Q9<BgkCollision> {
        let state_clone = self
            .shared_lattice
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let tau = 3.0 * self.viscosity.clamp(VISCOSITY_MIN, VISCOSITY_MAX) + 0.5;
        let mut solver = LatticeBoltzmannD2Q9::new(LBM_WIDTH, LBM_HEIGHT, tau);
        if state_clone.width() == LBM_WIDTH
            && state_clone.height() == LBM_HEIGHT
            && state_clone.density().iter().all(|v| v.is_finite())
        {
            solver.state = state_clone;
        } else {
            solver.set_inlet(0, 0, LBM_WIDTH, LBM_HEIGHT, self.inflow_speed.max(0.0), 0.0);
        }
        sync_solver_obstacles(&mut solver, &self.obstacles);
        solver
    }

    pub(crate) fn ensure_texture(&mut self, ctx: &egui::Context) {
        if self.texture.is_some() && !self.dirty {
            return;
        }
        let image = render_lbm_color_image(
            &self.build_snapshot_solver(),
            self.inflow_speed,
            self.view_mode,
            self.high_contrast,
        );
        if let Some(tex) = &mut self.texture {
            tex.set(image, egui::TextureOptions::NEAREST);
        } else {
            self.texture = Some(ctx.load_texture(
                "lbm_wind_tunnel_texture",
                image,
                egui::TextureOptions::NEAREST,
            ));
        }
        self.dirty = false;
    }

    fn sync_from_worker(&mut self, ctx: &egui::Context) {
        if let Some(snapshot) = self.worker.update() {
            Self::upload_worker_snapshot(
                &mut self.telemetry,
                &mut self.texture,
                &mut self.dirty,
                ctx,
                snapshot,
            );
        } else if self.dirty {
            self.ensure_texture(ctx);
        }
        if self.worker.is_running() {
            ctx.request_repaint();
        }
    }

    fn upload_worker_snapshot(
        telemetry: &mut LbmTelemetry,
        texture_slot: &mut Option<egui::TextureHandle>,
        dirty: &mut bool,
        ctx: &egui::Context,
        snapshot: &StateSnapshot,
    ) {
        if let Some(telem) = snapshot
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<LbmTelemetry>())
            .copied()
        {
            *telemetry = telem;
        }
        if let Ok(pixels) = snapshot.pixels.try_read() {
            if pixels.len() == snapshot.width * snapshot.height && !pixels.is_empty() {
                let image =
                    egui::ColorImage::new([snapshot.width, snapshot.height], pixels.clone());
                if let Some(tex) = texture_slot {
                    tex.set(image, egui::TextureOptions::NEAREST);
                } else {
                    *texture_slot = Some(ctx.load_texture(
                        "lbm_wind_tunnel_texture",
                        image,
                        egui::TextureOptions::NEAREST,
                    ));
                }
                *dirty = false;
            }
        }
    }

    fn current_tool_state(&self) -> LbmToolState {
        LbmToolState {
            viscosity: self.viscosity,
            inflow_speed: self.inflow_speed,
            steps_per_frame: self.steps_per_frame,
            brush_radius: self.brush_radius,
            area_budget: self.area_budget,
            view_mode: self.view_mode,
            obstacle_mask: encode_obstacle_mask(&self.obstacles),
        }
    }
}

impl InteractiveTool for LatticeBoltzmannTool {
    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Lattice Boltzmann Wind Tunnel"
    }

    fn domain(&self) -> &'static str {
        "fluid_dynamics"
    }

    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::raster_only()
    }

    fn export_raster_image(&self) -> Option<egui::ColorImage> {
        let u0 = if self.inflow_speed.is_finite() {
            self.inflow_speed.clamp(INFLOW_SPEED_MIN, INFLOW_SPEED_MAX)
        } else {
            DEFAULT_INFLOW_SPEED
        };
        if let Some(ref solver) = self.gif_solver {
            return Some(render_lbm_color_image(
                solver,
                u0,
                self.view_mode,
                self.high_contrast,
            ));
        }
        Some(render_lbm_color_image(
            &self.build_snapshot_solver(),
            u0,
            self.view_mode,
            self.high_contrast,
        ))
    }

    fn step_simulation(&mut self) {
        let u0 = self.inflow_speed.clamp(INFLOW_SPEED_MIN, INFLOW_SPEED_MAX);
        let visc = self.viscosity.clamp(VISCOSITY_MIN, VISCOSITY_MAX);
        if self.gif_solver.is_none() {
            self.gif_solver = Some(self.build_snapshot_solver());
        }
        let solver = self.gif_solver.as_mut().expect("initialized gif_solver");
        solver.collision_model.tau = 3.0 * visc + 0.5;
        sync_solver_obstacles(solver, &self.obstacles);
        let sub_steps = self
            .steps_per_frame
            .clamp(STEPS_PER_FRAME_MIN, STEPS_PER_FRAME_MAX)
            .max(8);
        for _ in 0..sub_steps {
            solver.set_inlet(0, 0, LBM_INLET_GUARD_COLS, LBM_HEIGHT, u0, 0.0);
            solver.step();
        }
        let net_force = solver.net_obstacle_force();
        self.telemetry = compute_aerodynamics(
            net_force,
            u0,
            visc,
            &self.obstacles,
            self.telemetry.step_count + sub_steps,
        );
        self.dirty = true;
    }

    fn has_finite_parameters(&self) -> bool {
        self.current_tool_state().validate().is_ok()
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.current_tool_state()).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: LbmToolState =
            serde_json::from_str(state).map_err(|e| format!("Invalid LBM wind-tunnel state: {e}"))?;
        let decoded_obstacles = parsed.validate()?;
        self.viscosity = parsed.viscosity;
        self.inflow_speed = parsed.inflow_speed;
        self.steps_per_frame = parsed.steps_per_frame;
        self.brush_radius = parsed.brush_radius;
        self.area_budget = parsed.area_budget;
        self.view_mode = parsed.view_mode;
        self.obstacles = decoded_obstacles;
        self.mask_generation = self.mask_generation.wrapping_add(1);
        {
            let mut cfg = self.shared_config.write().unwrap_or_else(|e| e.into_inner());
            cfg.viscosity = self.viscosity;
            cfg.inflow_speed = self.inflow_speed;
            cfg.steps_per_frame = self.steps_per_frame;
            cfg.view_mode = self.view_mode;
            cfg.obstacles.clone_from(&self.obstacles);
            cfg.mask_generation = self.mask_generation;
        }
        self.reset_flow_field();
        Ok(())
    }

    fn clamp_parameters(&mut self) -> bool {
        self.viscosity = if self.viscosity.is_finite() {
            self.viscosity.clamp(VISCOSITY_MIN, VISCOSITY_MAX)
        } else {
            DEFAULT_VISCOSITY
        };
        self.inflow_speed = if self.inflow_speed.is_finite() {
            self.inflow_speed.clamp(INFLOW_SPEED_MIN, INFLOW_SPEED_MAX)
        } else {
            DEFAULT_INFLOW_SPEED
        };
        self.steps_per_frame = self
            .steps_per_frame
            .clamp(STEPS_PER_FRAME_MIN, STEPS_PER_FRAME_MAX);
        self.brush_radius = self.brush_radius.clamp(BRUSH_RADIUS_MIN, BRUSH_RADIUS_MAX);
        self.area_budget = self.area_budget.clamp(AREA_BUDGET_MIN, AREA_BUDGET_MAX);
        self.enforce_current_budget();
        self.push_shared_config();
        self.reset_flow_field();
        true
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let ctx = parent_ui.ctx().clone();
        let is_hc = crate::app::ThemeMode::from_visuals(parent_ui.visuals())
            == crate::app::ThemeMode::HighContrast;
        if self.high_contrast != is_hc {
            self.high_contrast = is_hc;
            self.push_shared_config();
            self.dirty = true;
        }
        self.sync_from_worker(&ctx);
        self.draw_side_panel(parent_ui, &ctx);
        self.draw_central_panel(parent_ui, &ctx);
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "LatticeBoltzmannTool",
        display_name: "Lattice Boltzmann Wind Tunnel",
        domain: "fluid_dynamics",
        tags: &["cfd", "aerodynamics", "drag", "vorticity"],
        build: || Box::new(LatticeBoltzmannTool::default()),
    }
}
