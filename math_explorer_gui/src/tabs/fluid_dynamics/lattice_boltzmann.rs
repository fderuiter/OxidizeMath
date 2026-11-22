use crate::async_sim::unified::{UnifiedModel, UnifiedSimTool};
use crate::async_sim::{SimCommand, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use eframe::egui::Color32;
use math_commons::math_kernel::colormap::heatmap_color;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    BgkCollision, LatticeBoltzmannD2Q9,
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

        StateSnapshot {
            width,
            height,
            pixels: Arc::new(std::sync::RwLock::new(pixels)),
            custom_data: Vec::new(),
            structured_data: None,
        }
    }

    fn process_command(&mut self, cmd: SimCommand, _params: &HashMap<String, f64>) {
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
                            && (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r
                        {
                            self.solver
                                .set_obstacle(x as usize, y as usize, is_obstacle);
                        }
                    }
                }
            }
            SimCommand::ApplyBrushCells { cells, mode } => {
                let is_obs = mode != 0;
                let w = self.solver.width();
                let h = self.solver.height();
                for (x, y) in cells {
                    if x < w && y < h {
                        self.solver.set_obstacle(x, y, is_obs);
                    }
                }
            }
            SimCommand::ClearObstacles => {
                self.solver.clear_obstacles();
            }
            SimCommand::Custom(ref action) if action == "Clear Obstacles" => {
                self.solver.clear_obstacles();
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
        Some((100, 50, true))
    }

    fn default_obstacle_mask(_params: &HashMap<String, f64>) -> Option<Vec<bool>> {
        let (width, height) = (100usize, 50usize);
        let mut mask = vec![false; width * height];
        for y in 0..height {
            for x in 0..width {
                let dx = x as i32 - 30;
                let dy = y as i32 - 25;
                if dx * dx + dy * dy <= 25 {
                    mask[flatten_2d_index(x, y, width)] = true;
                }
            }
        }
        Some(mask)
    }

    fn validate_obstacle_rle(encoded: &str) -> Result<Vec<bool>, String> {
        crate::framework::decode_bool_grid(encoded, 100, 50)
    }

    fn apply_obstacle_rle(&mut self, encoded: &str) {
        let Ok(mask) = Self::validate_obstacle_rle(encoded) else {
            return;
        };
        let w = self.solver.width();
        let h = self.solver.height();
        if mask.len() != w * h {
            return;
        }
        self.solver.clear_obstacles();
        for y in 0..h {
            for x in 0..w {
                if mask[flatten_2d_index(x, y, w)] {
                    self.solver.set_obstacle(x, y, true);
                }
            }
        }
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

fn generate_airfoil_starter_json() -> &'static str {
    static AIRFOIL_PRESET: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    AIRFOIL_PRESET
        .get_or_init(|| {
            let mut mask = vec![false; 100 * 50];
            for (x, y) in crate::framework::generate_brush_cells(
                crate::framework::BrushStencil::Airfoil,
                crate::framework::SymmetryMode::None,
                (32, 25),
                (32, 25),
                4,
                100,
                50,
            ) {
                mask[flatten_2d_index(x, y, 100)] = true;
            }
            let rle = crate::framework::encode_bool_grid(100, 50, &mask)
                .expect("valid 100x50 airfoil RLE");
            serde_json::json!({
                "metadata": {
                    "schema_version": "1.0",
                    "title": "LBM Airfoil Wake",
                    "domain": "fluid_dynamics",
                    "timestamp": "2026-11-21T00:00:00Z"
                },
                "config": {
                    "viscosity": 0.015,
                    "obstacles_rle": rle
                }
            })
            .to_string()
        })
        .as_str()
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "lbm_airfoil_wake",
        title: "LBM Airfoil Wake",
        description: "Sculpted symmetric NACA airfoil immersed in low-viscosity D2Q9 flow with RLE obstacle persistence.",
        category: "Computational Fluids",
        icon: "✈️",
        domain: "fluid_dynamics",
        tool_name: Some("LatticeBoltzmannTool"),
        preset_generator: generate_airfoil_starter_json,
    }
}

#[cfg(test)]
#[path = "lbm_tests.rs"]
mod tests;
