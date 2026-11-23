//! Dedicated D2Q9 Lattice Boltzmann Wind Tunnel tool with live bounce-back
//! aerodynamic telemetry (`C_D`, `C_L`, `Re`), URL-shareable obstacle sculpting,
//! `SpatialBrushKit` stroke integration (`Solid`, `Erase`, `VelocityJet`),
//! inverse fairing challenge (`lbm_low_drag_fairing`), and vortex-shedding GIF export.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::lbm_fairing::{
    restore_inlet_and_core_mask, LbmFairingEvaluator, CHALLENGE_LBM_FAIRING,
    CHALLENGE_LBM_VISCOSITY, LBM_U_IN,
};
use crate::async_sim::{BackgroundWorker, SimCommand};
use crate::framework::{
    bresenham_line, BrushMode, BrushStroke, ChallengeEvaluation, InteractionContext,
    InteractiveTool, SpatialBrushKit,
};
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
pub use legacy_unified::{LbmSnapshotMetrics, LbmUnified, LBM_INLET_U};

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
    pub(crate) last_drag_cell: Option<(usize, usize)>,
    pub(crate) focused_canvas: Option<egui::Id>,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) dirty: bool,
    pub(crate) gif_solver: Option<LatticeBoltzmannD2Q9<BgkCollision>>,
    pub(crate) active_challenge: Option<&'static str>,
    pub(crate) challenge_evaluator: Option<LbmFairingEvaluator>,
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
            last_drag_cell: None,
            focused_canvas: None,
            texture: None,
            dirty: true,
            gif_solver: None,
            active_challenge: None,
            challenge_evaluator: None,
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

    pub(super) fn enforce_challenge_invariants_and_score(&mut self) {
        if self.active_challenge != Some(CHALLENGE_LBM_FAIRING) {
            return;
        }
        self.viscosity = CHALLENGE_LBM_VISCOSITY;
        self.inflow_speed = LBM_U_IN;
        self.area_budget = self.area_budget.max(DEFAULT_AREA_BUDGET);
        let mut mask = [false; LBM_CELL_COUNT];
        mask.copy_from_slice(&self.obstacles[..LBM_CELL_COUNT]);
        restore_inlet_and_core_mask(&mut mask);
        self.obstacles.copy_from_slice(&mask);
        let evaluator = self.challenge_evaluator.get_or_insert_with(Default::default);
        evaluator.evaluate_mask_in_place(&mask, CHALLENGE_LBM_VISCOSITY);
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
        self.enforce_challenge_invariants_and_score();
        self.sync_obstacles_to_worker();
    }

    /// Applies a continuous [`BrushStroke`] (`Solid`, `Erase`, or `VelocityJet`) in lattice coordinates.
    pub fn apply_brush_stroke(&mut self, stroke: &BrushStroke) {
        match stroke.mode {
            BrushMode::Solid | BrushMode::Erase => {
                let (x0, y0) =
                    SpatialBrushKit::plot_to_grid_y_up(stroke.start, LBM_WIDTH, LBM_HEIGHT);
                let (x1, y1) =
                    SpatialBrushKit::plot_to_grid_y_up(stroke.end, LBM_WIDTH, LBM_HEIGHT);
                let radius = (stroke.radius.round() as i32).clamp(BRUSH_RADIUS_MIN, BRUSH_RADIUS_MAX);
                let paint_solid = stroke.mode == BrushMode::Solid;
                for (cx, cy) in bresenham_line(x0, y0, x1, y1) {
                    if cx >= 0 && cy >= 0 {
                        apply_brush_with_budget(
                            &mut self.obstacles,
                            cx as usize,
                            cy as usize,
                            radius,
                            paint_solid,
                            self.area_budget,
                        );
                    }
                }
                self.enforce_challenge_invariants_and_score();
                self.sync_obstacles_to_worker();
            }
            BrushMode::VelocityJet => {
                self.gif_solver = None;
                let _ = self
                    .worker
                    .send_command(SimCommand::ApplyBrushStroke(*stroke));
                self.dirty = true;
            }
            _ => {}
        }
    }

    pub(super) fn sync_obstacles_to_worker(&mut self) {
        self.mask_generation = self.mask_generation.wrapping_add(1);
        {
            let mut cfg = self.shared_config.write().unwrap_or_else(|e| e.into_inner());
            cfg.viscosity = self.viscosity;
            cfg.inflow_speed = self.inflow_speed;
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
        self.enforce_challenge_invariants_and_score();
        self.push_shared_config();
        self.reset_flow_field();
    }

    pub(super) fn build_snapshot_solver(&self) -> LatticeBoltzmannD2Q9<BgkCollision> {
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

    fn current_tool_state(&self) -> LbmToolState {
        LbmToolState {
            viscosity: self.viscosity,
            inflow_speed: self.inflow_speed,
            steps_per_frame: self.steps_per_frame,
            brush_radius: self.brush_radius,
            area_budget: self.area_budget,
            view_mode: self.view_mode,
            obstacle_mask: encode_obstacle_mask(&self.obstacles),
            mask_hex: None,
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

    fn on_brush(&mut self, ctx: &InteractionContext) {
        if let Some(ref stroke) = ctx.brush_stroke {
            self.apply_brush_stroke(stroke);
        }
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }

    fn apply_challenge(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        match challenge_id {
            None => {
                self.active_challenge = None;
                Ok(())
            }
            Some(CHALLENGE_LBM_FAIRING) => {
                self.active_challenge = Some(CHALLENGE_LBM_FAIRING);
                self.enforce_challenge_invariants_and_score();
                self.sync_obstacles_to_worker();
                Ok(())
            }
            Some(other) => Err(format!(
                "Unsupported challenge '{other}' for LatticeBoltzmannTool"
            )),
        }
    }

    fn active_challenge_id(&self) -> Option<&'static str> {
        self.active_challenge
    }

    fn evaluate_challenge(&self) -> Option<ChallengeEvaluation> {
        self.active_challenge?;
        self.challenge_evaluator
            .as_ref()
            .map(LbmFairingEvaluator::challenge_evaluation)
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
        let parsed: LbmToolState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid LBM wind-tunnel state: {e}"))?;
        let decoded_obstacles = parsed.validate()?;
        self.viscosity = parsed.viscosity;
        self.inflow_speed = parsed.inflow_speed;
        self.steps_per_frame = parsed.steps_per_frame;
        self.brush_radius = parsed.brush_radius;
        self.area_budget = parsed.area_budget;
        self.view_mode = parsed.view_mode;
        self.obstacles = decoded_obstacles;
        self.enforce_challenge_invariants_and_score();
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
