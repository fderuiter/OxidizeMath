#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::lattice_boltzmann_sculpt::{
    apply_lbm_custom_action, apply_stroke_to_lbm, draw_lbm_brush_panel, export_solver_rle,
    inspect_solver_solid_and_core, record_stroke_in_rle, reset_solver_preserving_obstacles,
    rotate_airfoil_in_place, seed_initial_lbm_geometry,
};
use super::lbm_fairing::{
    decode_mask_hex, FairingStamp, LBM_CELL_COUNT, LBM_INLET_COLS, LBM_REF_DIAMETER, LBM_RHO_0,
    LBM_U_IN,
};
use super::lbm_fairing_ui::LbmUiExtension;
use crate::async_sim::unified::{UnifiedModel, UnifiedSimTool, UnifiedUiExtension};
use crate::async_sim::{SculptStatePayload, SimCommand, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{decode_rle_mask, BrushMode, BrushStroke, SpatialBrushKit};
use eframe::egui::Color32;
use math_commons::math_kernel::colormap::heatmap_color;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    advance_wind_tunnel_step, compute_aerodynamic_coefficients, compute_drag_coefficient,
    compute_vorticity_into, AerodynamicCoefficients, BgkCollision, LatticeBoltzmannD2Q9,
};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub(crate) const LBM_WIDTH: usize = 100;
pub(crate) const LBM_HEIGHT: usize = 50;
pub(crate) const LBM_INLET_U: f64 = 0.1;

/// Live hydrodynamic force and vorticity diagnostics attached to LBM snapshots.
#[derive(Debug, Clone, Copy, Default)]
pub struct LbmSnapshotMetrics {
    /// Dimensionless lift coefficient $C_L$.
    pub lift_coeff: f64,
    /// Dimensionless drag coefficient $C_D$.
    pub drag_coeff: f64,
    /// Peak vorticity magnitude $|\omega_z|$ across interior fluid cells.
    pub max_vorticity: f64,
}

pub struct LbmUnified {
    pub(crate) solver: LatticeBoltzmannD2Q9<BgkCollision>,
    pub(crate) vorticity_buf: Vec<f64>,
    pub(crate) latest_aero: AerodynamicCoefficients,
    pub(crate) stamped_airfoil: Option<(u16, f64)>,
    pub(crate) active_profile_param: f64,
    pub(crate) challenge_core_locked: bool,
    pub(crate) display_vorticity: bool,
    pub(crate) diverged: bool,
    pub(crate) diagnostic_report: Option<DiagnosticReport>,
    pub(crate) step_count: usize,
}

impl LbmUnified {
    fn refresh_diagnostics(&mut self) {
        compute_vorticity_into(&self.solver, &mut self.vorticity_buf);
        self.latest_aero = compute_aerodynamic_coefficients(&self.solver, LBM_INLET_U, 24.0, 3);
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
        reset_solver_preserving_obstacles(&mut self.solver, viscosity, restore_core);
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
                    self.solver.set_obstacle(x as usize, y as usize, is_obstacle);
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
        self.solver.state.velocity_x().iter().all(|&v| v.is_finite())
            && self.solver.state.velocity_y().iter().all(|&v| v.is_finite())
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
            stamped_airfoil,
            active_profile_param,
            challenge_core_locked: false,
            display_vorticity,
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
        if let Some((code, prev_aoa)) = self.stamped_airfoil {
            let aoa = *params.get("angle_of_attack").unwrap_or(&0.0);
            if aoa.is_finite() && (aoa - prev_aoa).abs() > 1e-6 {
                rotate_airfoil_in_place(&mut self.solver, code, prev_aoa, aoa);
                self.stamped_airfoil = Some((code, aoa));
            }
        }
        self.display_vorticity = *params.get("field_view").unwrap_or(&0.0) >= 0.5;
        self.step_count += 1;
        self.solver.collision_model.tau = 3.0 * viscosity + 0.5;
        advance_wind_tunnel_step(&mut self.solver, LBM_INLET_U);
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

    fn get_snapshot(&self) -> StateSnapshot {
        let width = self.solver.width();
        let height = self.solver.height();
        let mut pixels = vec![Color32::BLACK; width * height];
        let mut max_vort = 0.0_f64;

        for y in 0..height {
            let solver_y = height - 1 - y;
            for x in 0..width {
                let pixel_idx = flatten_2d_index(x, y, width);
                let solver_idx = flatten_2d_index(x, solver_y, width);
                if self.solver.is_obstacle(x, solver_y) {
                    pixels[pixel_idx] = Color32::from_gray(50);
                } else {
                    let omega = self.vorticity_buf[solver_idx];
                    let safe_w = if omega.is_finite() { omega } else { 0.0 };
                    max_vort = max_vort.max(safe_w.abs());
                    let norm = if self.display_vorticity {
                        (0.5 + safe_w * 12.0).clamp(0.0, 1.0)
                    } else {
                        let v = self.solver.get_velocity_magnitude(x, solver_y);
                        let safe_v = if v.is_finite() { v } else { 0.0 };
                        (safe_v * 1000.0 / 255.0).clamp(0.0, 1.0)
                    };
                    let rgb = heatmap_color(norm);
                    pixels[pixel_idx] = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
                }
            }
        }

        let (f_drag, f_lift) = self.solver.hydrodynamic_forces();
        let c_d = compute_drag_coefficient(f_drag, LBM_RHO_0, LBM_U_IN, LBM_REF_DIAMETER);
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
            SimCommand::ClearObstacles => {
                self.solver.clear_obstacles();
                self.stamped_airfoil = None;
                self.refresh_diagnostics();
            }
            SimCommand::Custom(ref action) if action.starts_with("ChallengeLockCore:") => {
                self.challenge_core_locked = action.ends_with('1');
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
        vec!["Clear Obstacles"]
    }

    fn is_raster_model() -> bool {
        true
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

    fn apply_sculpt_payload(&mut self, payload: &SculptStatePayload, params: &HashMap<String, f64>) {
        if payload.rle_mask.is_empty() {
            return;
        }
        let w = self.solver.width();
        let h = self.solver.height();
        if let Ok(bytes) = decode_rle_mask(&payload.rle_mask, w * h) {
            for y in 0..h {
                for x in 0..w {
                    self.solver
                        .set_obstacle(x, y, bytes[flatten_2d_index(x, y, w)] != 0);
                }
            }
            let profile = *params.get("naca_profile").unwrap_or(&0.0);
            let aoa = *params.get("angle_of_attack").unwrap_or(&0.0);
            self.active_profile_param = profile;
            self.stamped_airfoil = (profile >= 1.0).then_some((profile.round() as u16, aoa));
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
        let c = |min, max, step| ParameterConstraint { min, max, step };
        HashMap::from([
            ("viscosity".to_string(), c(0.00333, 0.5, 0.005)),
            ("angle_of_attack".to_string(), c(-25.0, 25.0, 1.0)),
            ("naca_profile".to_string(), c(0.0, 4412.0, 1.0)),
            ("field_view".to_string(), c(0.0, 1.0, 1.0)),
        ])
    }

    fn parameter_label(name: &str) -> String {
        match name {
            "viscosity" => "Kinematic Viscosity (ν)".to_string(),
            "angle_of_attack" => "Angle of Attack α (°)".to_string(),
            "naca_profile" => "Obstacle / Airfoil Profile".to_string(),
            "field_view" => "Scalar Field View".to_string(),
            other => other.to_string(),
        }
    }

    fn parameter_choices(name: &str) -> &'static [(&'static str, f64)] {
        match name {
            "naca_profile" => &[
                ("Cylinder (Default)", 0.0),
                ("NACA 0012 (Symmetric)", 12.0),
                ("NACA 2412 (Cambered)", 2412.0),
                ("NACA 4412 (High Camber)", 4412.0),
            ],
            "field_view" => &[("Velocity |u|", 0.0), ("Vorticity ω_z", 1.0)],
            _ => &[],
        }
    }

    fn name() -> &'static str {
        "Lattice Boltzmann (Demo)"
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "LatticeBoltzmannTool",
        display_name: "Lattice Boltzmann (Demo)",
        domain: "fluid_dynamics",
        tags: &["lbm", "d2q9", "bgk", "vortex street", "navier-stokes", "fluid"],
        build: || Box::new(UnifiedSimTool::<LbmUnified>::new()),
    }
}

impl TheoryDescribable for LbmUnified {
    fn theory_description(&self) -> String {
        LatticeBoltzmannD2Q9::<BgkCollision>::new(1, 1, 1.0).theory_description()
    }

    fn phonetic_description(&self) -> String {
        LatticeBoltzmannD2Q9::<BgkCollision>::new(1, 1, 1.0).phonetic_description()
    }

    fn theory_citation(&self) -> String {
        LatticeBoltzmannD2Q9::<BgkCollision>::new(1, 1, 1.0).theory_citation()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        LatticeBoltzmannD2Q9::<BgkCollision>::new(1, 1, 1.0).available_descriptions()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "LBM Flow Simulation",
    "domain": "fluid_dynamics",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "viscosity": 0.02
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "lattice_boltzmann_starter",
        title: "LBM Flow Simulation",
        description: "f_i(x + e_i·dt, t + dt) = f_i(x, t) - (f_i - f_i^eq)/τ: Mesoscopic BGK collision streaming solver.",
        category: "Computational Fluids",
        icon: "🌊",
        domain: "fluid_dynamics",
        tool_name: Some("LatticeBoltzmannTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
