use super::lattice_boltzmann_sculpt::{
    apply_lbm_custom_action, apply_stroke_to_lbm, draw_lbm_brush_panel, export_solver_rle,
    record_stroke_in_rle, rotate_airfoil_in_place, stamp_airfoil_on_solver,
};
use crate::async_sim::unified::{UnifiedModel, UnifiedSimTool};
use crate::async_sim::{SculptStatePayload, SimCommand, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{decode_rle_mask, BrushMode, BrushStroke, SpatialBrushKit};
use eframe::egui::Color32;
use math_commons::math_kernel::colormap::heatmap_color;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    advance_wind_tunnel_step, compute_aerodynamic_coefficients, compute_vorticity_into,
    AerodynamicCoefficients, BgkCollision, LatticeBoltzmannD2Q9,
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

#[cfg_attr(any(), verified(opt_out = "gui_tool"))]
pub struct LbmUnified {
    pub(crate) solver: LatticeBoltzmannD2Q9<BgkCollision>,
    pub(crate) vorticity_buf: Vec<f64>,
    pub(crate) latest_aero: AerodynamicCoefficients,
    pub(crate) stamped_airfoil: Option<(u16, f64)>,
    pub(crate) display_vorticity: bool,
    pub(crate) diverged: bool,
    pub(crate) diagnostic_report: Option<DiagnosticReport>,
    pub(crate) step_count: usize,
}

impl LbmUnified {
    pub(crate) fn seed_initial_geometry(
        solver: &mut LatticeBoltzmannD2Q9<BgkCollision>,
        params: &HashMap<String, f64>,
    ) -> Option<(u16, f64)> {
        let profile = *params.get("naca_profile").unwrap_or(&0.0);
        let aoa = *params.get("angle_of_attack").unwrap_or(&0.0);
        let stamped = if profile >= 1.0 {
            let code = profile.round() as u16;
            stamp_airfoil_on_solver(solver, code, aoa);
            Some((code, aoa))
        } else {
            let center_x = 30;
            let center_y = 25;
            let radius = 5;
            for y in 0..solver.height() {
                for x in 0..solver.width() {
                    let dx = x as i32 - center_x;
                    let dy = y as i32 - center_y;
                    if dx * dx + dy * dy <= radius * radius {
                        solver.set_obstacle(x, y, true);
                    }
                }
            }
            None
        };
        solver.set_inlet(0, 0, solver.width(), solver.height(), LBM_INLET_U, 0.0);
        stamped
    }

    fn refresh_diagnostics(&mut self) {
        compute_vorticity_into(&self.solver, &mut self.vorticity_buf);
        self.latest_aero = compute_aerodynamic_coefficients(&self.solver, LBM_INLET_U, 24.0, 3);
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
        let stamped_airfoil = Self::seed_initial_geometry(&mut solver, params);

        let display_vorticity = *params.get("field_view").unwrap_or(&0.0) >= 0.5;
        let mut model = Self {
            solver,
            vorticity_buf: vec![0.0; LBM_WIDTH * LBM_HEIGHT],
            latest_aero: AerodynamicCoefficients::default(),
            stamped_airfoil,
            display_vorticity,
            diverged: false,
            diagnostic_report: None,
            step_count: 0,
        };
        model.refresh_diagnostics();
        model
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
                } else if self.display_vorticity {
                    let omega = self.vorticity_buf[solver_idx];
                    let safe_w = if omega.is_finite() { omega } else { 0.0 };
                    max_vort = max_vort.max(safe_w.abs());
                    let norm = (0.5 + safe_w * 12.0).clamp(0.0, 1.0);
                    let rgb = heatmap_color(norm);
                    pixels[pixel_idx] = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
                } else {
                    let omega = self.vorticity_buf[solver_idx];
                    if omega.is_finite() {
                        max_vort = max_vort.max(omega.abs());
                    }
                    let v = self.solver.get_velocity_magnitude(x, solver_y);
                    let safe_v = if v.is_finite() { v } else { 0.0 };
                    let normalized_v = (safe_v * 1000.0 / 255.0).clamp(0.0, 1.0);
                    let rgb = heatmap_color(normalized_v);
                    pixels[pixel_idx] = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
                }
            }
        }

        let metrics = LbmSnapshotMetrics {
            lift_coeff: self.latest_aero.lift_coeff,
            drag_coeff: self.latest_aero.drag_coeff,
            max_vorticity: max_vort,
        };

        StateSnapshot {
            width,
            height,
            pixels: Arc::new(RwLock::new(pixels)),
            custom_data: Vec::new(),
            structured_data: Some(Box::new(metrics)),
        }
    }

    fn process_command(&mut self, cmd: SimCommand, params: &HashMap<String, f64>) {
        self.display_vorticity = *params.get("field_view").unwrap_or(&0.0) >= 0.5;
        match cmd {
            SimCommand::Reset => {
                self.diverged = false;
                self.diagnostic_report = None;
                self.step_count = 0;
            }
            SimCommand::ApplyBrush {
                cx,
                cy,
                r,
                is_obstacle,
            } => {
                let width = self.solver.width() as i32;
                let height = self.solver.height() as i32;
                for y in (cy - r)..=(cy + r) {
                    for x in (cx - r)..=(cx + r) {
                        if x >= 0
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
            SimCommand::ApplyBrushStroke(ref stroke) => {
                apply_stroke_to_lbm(&mut self.solver, stroke);
                self.refresh_diagnostics();
            }
            SimCommand::ClearObstacles => {
                self.solver.clear_obstacles();
                self.stamped_airfoil = None;
                self.refresh_diagnostics();
            }
            SimCommand::Custom(ref action)
                if apply_lbm_custom_action(
                    &mut self.solver,
                    &mut self.stamped_airfoil,
                    action,
                    params,
                ) =>
            {
                self.refresh_diagnostics();
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
        let mut map = HashMap::new();
        map.insert(
            "viscosity".to_string(),
            ParameterConstraint {
                min: 0.00333,
                max: 0.5,
                step: 0.005,
            },
        );
        map.insert(
            "angle_of_attack".to_string(),
            ParameterConstraint {
                min: -25.0,
                max: 25.0,
                step: 1.0,
            },
        );
        map.insert(
            "naca_profile".to_string(),
            ParameterConstraint {
                min: 0.0,
                max: 4412.0,
                step: 1.0,
            },
        );
        map.insert(
            "field_view".to_string(),
            ParameterConstraint {
                min: 0.0,
                max: 1.0,
                step: 1.0,
            },
        );
        map
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
        tags: &[],
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
