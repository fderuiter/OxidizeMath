//! Legacy `UnifiedModel` adapter (`LbmUnified`) with `SpatialBrushKit` hooks retained for
//! async worker, fault-recovery, and multi-domain spatial-brush integration test suites.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::lattice_boltzmann_sculpt::{
    apply_lbm_brush_cells, apply_lbm_custom_action, apply_lbm_sculpt_payload, apply_stroke_to_lbm,
    draw_lbm_brush_panel, export_solver_rle, record_stroke_in_rle,
    reset_solver_preserving_obstacles, seed_initial_lbm_geometry, sync_lbm_parameters,
};
use super::super::lbm_atlas::{resolve_inlet_velocity, LbmCartographer, LBM_ATLAS_AXIS_PAIRS};
use super::super::lbm_fairing::{
    decode_mask_hex, FairingStamp, LBM_CELL_COUNT, LBM_HEIGHT, LBM_INLET_COLS, LBM_REF_DIAMETER,
    LBM_RHO_0, LBM_WIDTH,
};
use super::super::lbm_fairing_ui::LbmUiExtension;
use super::super::lbm_presets::{apply_obstacle_rle_to_solver, default_lbm_obstacle_mask};
use super::render::{inspect_solver_solid_and_core, render_lbm_snapshot_pixels};
use crate::async_sim::atlas::RegimeCartographer;
use crate::async_sim::unified::{UnifiedModel, UnifiedUiExtension};
use crate::async_sim::{SculptStatePayload, SimCommand, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{BrushMode, BrushStroke, SpatialBrushKit};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    advance_wind_tunnel_step, compute_aerodynamic_coefficients, compute_drag_coefficient,
    compute_vorticity_into, AerodynamicCoefficients, BgkCollision, LatticeBoltzmannD2Q9,
};
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Live hydrodynamic force and vorticity diagnostics attached to [`LbmUnified`] snapshots.
#[derive(Debug, Clone, Copy, Default)]
pub struct LbmSnapshotMetrics {
    /// Dimensionless lift coefficient $C_L$.
    pub lift_coeff: f64,
    /// Dimensionless drag coefficient $C_D$.
    pub drag_coeff: f64,
    /// Peak vorticity magnitude $|\omega_z|$ across interior fluid cells.
    pub max_vorticity: f64,
}

/// `UnifiedSimTool` model wrapper around [`LatticeBoltzmannD2Q9`] with spatial-brush and fairing hooks.
pub struct LbmUnified {
    pub(crate) solver: LatticeBoltzmannD2Q9<BgkCollision>,
    pub(crate) vorticity_buf: Vec<f64>,
    pub(crate) latest_aero: AerodynamicCoefficients,
    pub(crate) inlet_velocity: f64,
    pub(crate) stamped_airfoil: Option<(u16, f64)>,
    pub(crate) active_profile_param: f64,
    pub(crate) challenge_core_locked: bool,
    pub(crate) display_vorticity: bool,
    pub(crate) high_contrast: bool,
    pub(crate) diverged: bool,
    pub(crate) diagnostic_report: Option<DiagnosticReport>,
    pub(crate) step_count: usize,
}

impl LbmUnified {
    fn refresh_diagnostics(&mut self) {
        compute_vorticity_into(&self.solver, &mut self.vorticity_buf);
        let u_ref = self.inlet_velocity.max(1e-4);
        self.latest_aero = compute_aerodynamic_coefficients(&self.solver, u_ref, 24.0, 3);
    }

    /// Re-initializes equilibrium populations via `init_equilibrium` and restores
    /// open inlet columns (`x < 2`) plus the protected `R_core = 4` structural core
    /// (when in cylinder/fairing mode), while preserving user-sculpted obstacle cells elsewhere.
    pub fn reset_equilibrium_preserving_obstacles(&mut self, params: &HashMap<String, f64>) {
        let viscosity = params
            .get("viscosity")
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(0.02);
        let restore_core = self.stamped_airfoil.is_none();
        self.inlet_velocity = resolve_inlet_velocity(params);
        reset_solver_preserving_obstacles(
            &mut self.solver,
            viscosity,
            self.inlet_velocity,
            restore_core,
        );
        self.diverged = false;
        self.diagnostic_report = None;
        self.step_count = 0;
        self.refresh_diagnostics();
    }

    fn apply_mask(&mut self, mask: &[bool; LBM_CELL_COUNT], params: &HashMap<String, f64>) {
        for y in 0..LBM_HEIGHT.min(self.solver.height()) {
            for x in 0..LBM_WIDTH.min(self.solver.width()) {
                self.solver
                    .set_obstacle(x, y, mask[flatten_2d_index(x, y, LBM_WIDTH)]);
            }
        }
        self.stamped_airfoil = None;
        self.active_profile_param = 0.0;
        self.reset_equilibrium_preserving_obstacles(params);
    }

    fn apply_discrete_brush(&mut self, cx: i32, cy: i32, r: i32, is_obstacle: bool) {
        let width = self.solver.width() as i32;
        let height = self.solver.height() as i32;
        for y in (cy - r)..=(cy + r) {
            for x in (cx - r)..=(cx + r) {
                if x >= LBM_INLET_COLS as i32
                    && x < width
                    && y >= 0
                    && y < height
                    && (x - cx).pow(2) + (y - cy).pow(2) <= r * r
                {
                    self.solver
                        .set_obstacle(x as usize, y as usize, is_obstacle);
                }
            }
        }
        self.refresh_diagnostics();
    }
}

impl FiniteStateGuard for LbmUnified {
    fn is_finite_state(&self) -> bool {
        if self.diverged {
            return false;
        }
        self.solver
            .state
            .velocity_x()
            .iter()
            .all(|&v| v.is_finite())
            && self
                .solver
                .state
                .velocity_y()
                .iter()
                .all(|&v| v.is_finite())
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

impl UnifiedModel for LbmUnified {
    fn domain() -> &'static str {
        "fluid_dynamics"
    }

    fn new(params: &HashMap<String, f64>) -> Self {
        let viscosity = *params.get("viscosity").unwrap_or(&0.02);
        let tau = 3.0 * viscosity + 0.5;
        let mut solver = LatticeBoltzmannD2Q9::new(LBM_WIDTH, LBM_HEIGHT, tau);
        let active_profile_param = *params.get("naca_profile").unwrap_or(&0.0);
        let stamped_airfoil = seed_initial_lbm_geometry(&mut solver, params);
        let display_vorticity = *params.get("field_view").unwrap_or(&0.0) >= 0.5;
        let mut model = Self {
            solver,
            vorticity_buf: vec![0.0; LBM_WIDTH * LBM_HEIGHT],
            latest_aero: AerodynamicCoefficients::default(),
            inlet_velocity: resolve_inlet_velocity(params),
            stamped_airfoil,
            active_profile_param,
            challenge_core_locked: false,
            display_vorticity,
            high_contrast: false,
            diverged: false,
            diagnostic_report: None,
            step_count: 0,
        };
        model.refresh_diagnostics();
        model
    }

    fn reset_model(&mut self, params: &HashMap<String, f64>) {
        let next_profile = *params.get("naca_profile").unwrap_or(&0.0);
        if !self.challenge_core_locked && (next_profile - self.active_profile_param).abs() > 1e-6 {
            let viscosity = params
                .get("viscosity")
                .copied()
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(0.02);
            self.solver.collision_model.tau = 3.0 * viscosity + 0.5;
            self.solver.clear_obstacles();
            self.solver.init_equilibrium();
            self.stamped_airfoil = seed_initial_lbm_geometry(&mut self.solver, params);
            self.active_profile_param = next_profile;
            self.inlet_velocity = resolve_inlet_velocity(params);
            self.diverged = false;
            self.diagnostic_report = None;
            self.step_count = 0;
            self.refresh_diagnostics();
        } else {
            self.reset_equilibrium_preserving_obstacles(params);
        }
    }

    fn create_ui_extension() -> Option<Box<dyn UnifiedUiExtension>> {
        Some(Box::new(LbmUiExtension::default()))
    }

    fn step(&mut self, params: &HashMap<String, f64>) {
        if self.diverged {
            return;
        }
        let viscosity = *params.get("viscosity").unwrap_or(&0.02);
        if !viscosity.is_finite() || viscosity <= 0.0 {
            self.diverged = true;
            self.diagnostic_report = Some(DiagnosticReport::new(
                self.step_count as f64,
                self.step_count,
                "viscosity",
                format!("Invalid non-finite viscosity: {viscosity}"),
            ));
            return;
        }
        sync_lbm_parameters(
            &mut self.solver,
            &mut self.stamped_airfoil,
            &mut self.display_vorticity,
            params,
        );
        self.inlet_velocity = resolve_inlet_velocity(params);
        self.step_count += 1;
        advance_wind_tunnel_step(&mut self.solver, self.inlet_velocity);
        self.refresh_diagnostics();

        if !FiniteStateGuard::is_finite_state(self) {
            self.diverged = true;
            self.diagnostic_report = Some(DiagnosticReport::new(
                self.step_count as f64,
                self.step_count,
                "velocity",
                "Non-finite velocity detected in Lattice Boltzmann grid cells.",
            ));
        }
    }

    fn sync_parameters(&mut self, params: &HashMap<String, f64>) {
        sync_lbm_parameters(
            &mut self.solver,
            &mut self.stamped_airfoil,
            &mut self.display_vorticity,
            params,
        );
        self.inlet_velocity = resolve_inlet_velocity(params);
        self.refresh_diagnostics();
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let width = self.solver.width();
        let height = self.solver.height();
        let (pixels, max_vort) = render_lbm_snapshot_pixels(
            &self.solver,
            &self.vorticity_buf,
            self.display_vorticity,
            self.high_contrast,
        );

        let (f_drag, f_lift) = self.solver.hydrodynamic_forces();
        let c_d = compute_drag_coefficient(
            f_drag,
            LBM_RHO_0,
            self.inlet_velocity.max(1e-4),
            LBM_REF_DIAMETER,
        );
        let (n_solid, core_intact) = inspect_solver_solid_and_core(&self.solver);
        let metrics = LbmSnapshotMetrics {
            lift_coeff: self.latest_aero.lift_coeff,
            drag_coeff: self.latest_aero.drag_coeff,
            max_vorticity: max_vort,
        };

        StateSnapshot {
            width,
            height,
            pixels: Arc::new(RwLock::new(pixels)),
            custom_data: vec![
                f_drag,
                f_lift,
                c_d,
                n_solid as f64,
                f64::from(u8::from(core_intact)),
                self.step_count as f64,
            ],
            structured_data: Some(Box::new(metrics)),
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    fn process_command(&mut self, cmd: SimCommand, params: &HashMap<String, f64>) {
        self.display_vorticity = *params.get("field_view").unwrap_or(&0.0) >= 0.5;
        match cmd {
            SimCommand::Reset => self.reset_equilibrium_preserving_obstacles(params),
            SimCommand::ApplyBrush {
                cx,
                cy,
                r,
                is_obstacle,
            } => self.apply_discrete_brush(cx, cy, r, is_obstacle),
            SimCommand::ApplyBrushStroke(ref stroke) => {
                apply_stroke_to_lbm(&mut self.solver, stroke, self.challenge_core_locked);
                self.refresh_diagnostics();
            }
            SimCommand::ApplyBrushCells { ref cells, mode } => {
                apply_lbm_brush_cells(&mut self.solver, cells, mode);
                self.refresh_diagnostics();
            }
            SimCommand::ClearObstacles => {
                self.solver.clear_obstacles();
                self.stamped_airfoil = None;
                self.refresh_diagnostics();
            }
            SimCommand::Custom(ref action) if action.starts_with("ChallengeLockCore:") => {
                self.challenge_core_locked = action.ends_with('1');
            }
            SimCommand::Custom(ref action) if action.starts_with("HighContrast:") => {
                self.high_contrast = action.ends_with('1');
            }
            SimCommand::Custom(ref action)
                if apply_lbm_custom_action(
                    &mut self.solver,
                    &mut self.stamped_airfoil,
                    action,
                    params,
                ) =>
            {
                self.active_profile_param = *params.get("naca_profile").unwrap_or(&0.0);
                self.refresh_diagnostics();
            }
            SimCommand::Custom(ref action) => {
                if let Some(stamp) = FairingStamp::from_command(action) {
                    let mut mask = [false; LBM_CELL_COUNT];
                    stamp.write_mask(&mut mask);
                    self.apply_mask(&mask, params);
                } else if let Some(hex) = action.strip_prefix("MaskHex:") {
                    let mut mask = [false; LBM_CELL_COUNT];
                    if decode_mask_hex(hex, &mut mask).is_ok() {
                        self.apply_mask(&mask, params);
                    }
                }
            }
            _ => {}
        }
    }

    fn custom_actions() -> Vec<&'static str> {
        vec!["Clear Obstacles", "Default Obstacles"]
    }

    fn is_raster_model() -> bool {
        true
    }

    fn raster_brush_grid() -> Option<(usize, usize, bool)> {
        Some((LBM_WIDTH, LBM_HEIGHT, true))
    }

    fn default_obstacle_mask(_params: &HashMap<String, f64>) -> Option<Vec<bool>> {
        Some(default_lbm_obstacle_mask())
    }

    fn validate_obstacle_rle(encoded: &str) -> Result<Vec<bool>, String> {
        crate::framework::decode_bool_grid(encoded, LBM_WIDTH, LBM_HEIGHT)
    }

    fn apply_obstacle_rle(&mut self, encoded: &str) {
        if apply_obstacle_rle_to_solver(&mut self.solver, encoded, self.inlet_velocity) {
            self.stamped_airfoil = None;
            self.refresh_diagnostics();
        }
    }

    fn supports_spatial_brush() -> bool {
        true
    }

    fn default_brush_kit() -> SpatialBrushKit {
        SpatialBrushKit::new(BrushMode::Solid, 2.0)
            .with_radius_bounds(1.0, 8.0, 1.0)
            .with_cursor_step(2.0, 2.0)
    }

    fn draw_brush_controls(
        ui: &mut eframe::egui::Ui,
        brush_kit: &mut SpatialBrushKit,
        params: &Arc<RwLock<HashMap<String, f64>>>,
        sculpt_state: &Arc<RwLock<SculptStatePayload>>,
        snapshot: Option<&StateSnapshot>,
    ) -> Vec<SimCommand> {
        draw_lbm_brush_panel(ui, brush_kit, params, sculpt_state, snapshot)
    }

    fn record_ui_brush_stroke(
        stroke: &BrushStroke,
        params: &HashMap<String, f64>,
        _formula: Option<&crate::async_sim::unified::CompiledFormulaPayload>,
        sculpt_state: &mut SculptStatePayload,
    ) {
        record_stroke_in_rle(stroke, params, sculpt_state);
    }

    fn apply_sculpt_payload(
        &mut self,
        payload: &SculptStatePayload,
        params: &HashMap<String, f64>,
    ) {
        if apply_lbm_sculpt_payload(
            &mut self.solver,
            &mut self.active_profile_param,
            &mut self.stamped_airfoil,
            payload,
            params,
        ) {
            self.refresh_diagnostics();
        }
    }

    fn export_sculpt_payload(&self, _params: &HashMap<String, f64>) -> SculptStatePayload {
        SculptStatePayload {
            rle_mask: export_solver_rle(&self.solver),
            sparse_potential: Vec::new(),
        }
    }

    fn parameters() -> HashMap<String, ParameterConstraint> {
        super::super::lbm_presets::lbm_parameters()
    }

    fn parameter_label(name: &str) -> String {
        super::super::lbm_presets::lbm_parameter_label(name)
    }

    fn parameter_choices(name: &str) -> &'static [(&'static str, f64)] {
        super::super::lbm_presets::lbm_parameter_choices(name)
    }

    fn atlas_axis_pairs() -> &'static [(&'static str, &'static str)] {
        LBM_ATLAS_AXIS_PAIRS
    }

    fn build_atlas_cartographer(
        params: &HashMap<String, f64>,
        axis_keys: (&'static str, &'static str),
    ) -> Option<Arc<dyn RegimeCartographer>> {
        LbmCartographer::from_params(params, axis_keys)
            .map(|c| Arc::new(c) as Arc<dyn RegimeCartographer>)
    }

    fn name() -> &'static str {
        "Lattice Boltzmann (Demo)"
    }
}
