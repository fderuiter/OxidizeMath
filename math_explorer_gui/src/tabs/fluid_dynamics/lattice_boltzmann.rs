use super::lbm_fairing::{
    decode_mask_hex, FairingStamp, LBM_CELL_COUNT, LBM_CORE_RADIUS, LBM_CORE_X, LBM_CORE_Y,
    LBM_HEIGHT, LBM_INLET_COLS, LBM_REF_DIAMETER, LBM_RHO_0, LBM_U_IN, LBM_WIDTH,
};
use super::lbm_fairing_ui::LbmUiExtension;
use crate::async_sim::unified::{UnifiedModel, UnifiedSimTool, UnifiedUiExtension};
use crate::async_sim::{SimCommand, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use eframe::egui::Color32;
use math_commons::math_kernel::colormap::heatmap_color;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    compute_drag_coefficient, BgkCollision, LatticeBoltzmannD2Q9,
};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg_attr(any(), verified(opt_out = "gui_tool"))]
pub struct LbmUnified {
    pub(crate) solver: LatticeBoltzmannD2Q9<BgkCollision>,
    pub(crate) diverged: bool,
    pub(crate) diagnostic_report: Option<DiagnosticReport>,
    pub(crate) step_count: usize,
}

impl LbmUnified {
    /// Re-initializes equilibrium populations via `init_equilibrium` and restores
    /// open inlet columns (`x < 2`) plus the protected `R_core = 4` structural core,
    /// while preserving user-sculpted obstacle cells elsewhere on the grid.
    pub fn reset_equilibrium_preserving_obstacles(&mut self, params: &HashMap<String, f64>) {
        let viscosity = params
            .get("viscosity")
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(0.02);
        self.solver.collision_model.tau = 3.0 * viscosity + 0.5;
        self.solver.init_equilibrium();
        let width = self.solver.width();
        let height = self.solver.height();
        for y in 0..height {
            for x in 0..width {
                if x < LBM_INLET_COLS {
                    self.solver.set_obstacle(x, y, false);
                } else {
                    let dx = x as i32 - LBM_CORE_X;
                    let dy = y as i32 - LBM_CORE_Y;
                    if dx * dx + dy * dy <= LBM_CORE_RADIUS * LBM_CORE_RADIUS {
                        self.solver.set_obstacle(x, y, true);
                    }
                }
            }
        }
        self.solver
            .set_inlet(0, 0, LBM_INLET_COLS, height, LBM_U_IN, 0.0);
        self.diverged = false;
        self.diagnostic_report = None;
        self.step_count = 0;
    }

    fn apply_mask(&mut self, mask: &[bool; LBM_CELL_COUNT], params: &HashMap<String, f64>) {
        for y in 0..LBM_HEIGHT.min(self.solver.height()) {
            for x in 0..LBM_WIDTH.min(self.solver.width()) {
                self.solver
                    .set_obstacle(x, y, mask[flatten_2d_index(x, y, LBM_WIDTH)]);
            }
        }
        self.reset_equilibrium_preserving_obstacles(params);
    }

    fn solid_and_core_status(&self) -> (usize, bool) {
        let mut n_solid = 0_usize;
        let mut core_intact = true;
        for y in 0..self.solver.height() {
            for x in 0..self.solver.width() {
                let obs = self.solver.is_obstacle(x, y);
                if obs {
                    n_solid += 1;
                }
                let dx = x as i32 - LBM_CORE_X;
                let dy = y as i32 - LBM_CORE_Y;
                if dx * dx + dy * dy <= LBM_CORE_RADIUS * LBM_CORE_RADIUS && !obs {
                    core_intact = false;
                }
            }
        }
        (n_solid, core_intact)
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
        let width = 100;
        let height = 50;
        let tau = 3.0 * viscosity + 0.5;

        let mut solver = LatticeBoltzmannD2Q9::new(width, height, tau);
        solver.set_inlet(0, 20, 5, 10, 0.1, 0.0);

        let center_x = 30;
        let center_y = 25;
        let radius = 5;
        for y in 0..height {
            for x in 0..width {
                let dx = x as i32 - center_x;
                let dy = y as i32 - center_y;
                if dx * dx + dy * dy <= radius * radius {
                    solver.set_obstacle(x, y, true);
                }
            }
        }

        solver.set_inlet(0, 0, 2, height, 0.1, 0.0);

        Self {
            solver,
            diverged: false,
            diagnostic_report: None,
            step_count: 0,
        }
    }

    fn reset_model(&mut self, params: &HashMap<String, f64>) {
        self.reset_equilibrium_preserving_obstacles(params);
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
                format!("Invalid non-finite viscosity: {}", viscosity),
            ));
            return;
        }
        self.step_count += 1;
        self.solver.collision_model.tau = 3.0 * viscosity + 0.5;
        self.solver
            .set_inlet(0, 0, 2, self.solver.height(), 0.1, 0.0);
        self.solver.step();

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
        let viscosity = *params.get("viscosity").unwrap_or(&0.02);
        if viscosity.is_finite() && viscosity > 0.0 {
            self.solver.collision_model.tau = 3.0 * viscosity + 0.5;
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let width = self.solver.width();
        let height = self.solver.height();

        let mut pixels = vec![Color32::BLACK; width * height];
        for y in 0..height {
            for x in 0..width {
                let pixel_idx = flatten_2d_index(x, y, width);
                if self.solver.is_obstacle(x, height - 1 - y) {
                    pixels[pixel_idx] = Color32::from_gray(50);
                } else {
                    let v = self.solver.get_velocity_magnitude(x, height - 1 - y);
                    let safe_v = if v.is_finite() { v } else { 0.0 };
                    let normalized_v = (safe_v * 1000.0 / 255.0).clamp(0.0, 1.0);
                    let safe_norm = if normalized_v.is_finite() {
                        normalized_v
                    } else {
                        0.0
                    };
                    let rgb = heatmap_color(safe_norm);
                    pixels[pixel_idx] = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
                }
            }
        }

        let (f_drag, f_lift) = self.solver.hydrodynamic_forces();
        let c_d = compute_drag_coefficient(f_drag, LBM_RHO_0, LBM_U_IN, LBM_REF_DIAMETER);
        let (n_solid, core_intact) = self.solid_and_core_status();

        StateSnapshot {
            width,
            height,
            pixels: Arc::new(std::sync::RwLock::new(pixels)),
            custom_data: vec![
                f_drag,
                f_lift,
                c_d,
                n_solid as f64,
                f64::from(u8::from(core_intact)),
                self.step_count as f64,
            ],
            structured_data: None,
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    fn process_command(&mut self, cmd: SimCommand, params: &HashMap<String, f64>) {
        match cmd {
            SimCommand::Reset => {
                self.reset_equilibrium_preserving_obstacles(params);
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
                        if x >= LBM_INLET_COLS as i32
                            && x < width
                            && y >= 0
                            && y < height
                            && (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r
                        {
                            self.solver
                                .set_obstacle(x as usize, y as usize, is_obstacle);
                        }
                    }
                }
            }
            SimCommand::ClearObstacles => {
                self.solver.clear_obstacles();
            }
            SimCommand::Custom(ref action) if action == "Clear Obstacles" => {
                self.solver.clear_obstacles();
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

    fn parameters() -> HashMap<String, ParameterConstraint> {
        let mut map = HashMap::new();

        // We will just hardcode constraint based on tau [0.51, 2.0] -> visc = (tau-0.5)/3.0
        // visc min = 0.00333, max = 0.5
        map.insert(
            "viscosity".to_string(),
            ParameterConstraint {
                min: 0.00333,
                max: 0.5,
                step: 0.01,
            },
        );
        map
    }

    fn parameter_label(name: &str) -> String {
        match name {
            "viscosity" => "Kinematic Viscosity (ν)".to_string(),
            other => other.to_string(),
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

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
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
