//! Theory metadata and Starter Gallery presets for [`super::lattice_boltzmann::LbmUnified`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::lattice_boltzmann::{LbmUnified, LBM_HEIGHT, LBM_WIDTH};
use super::lbm_fairing::{LBM_CELL_COUNT, LBM_CORE_RADIUS, LBM_CORE_X, LBM_CORE_Y, LBM_INLET_COLS};
use crate::async_sim::unified::UnifiedSimTool;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    BgkCollision, LatticeBoltzmannD2Q9,
};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;

pub(super) fn stamp_disk_into_fairing_mask(
    mask: &mut [bool; LBM_CELL_COUNT],
    cx: i32,
    cy: i32,
    r: i32,
    is_obstacle: bool,
    lock_core: bool,
) -> bool {
    let mut any_changed = false;
    for y in (cy - r)..=(cy + r) {
        for x in (cx - r)..=(cx + r) {
            if x >= LBM_INLET_COLS as i32
                && x < LBM_WIDTH as i32
                && y >= 0
                && y < LBM_HEIGHT as i32
                && (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r
            {
                let dx = x - LBM_CORE_X;
                let dy = y - LBM_CORE_Y;
                if !is_obstacle
                    && lock_core
                    && dx * dx + dy * dy <= LBM_CORE_RADIUS * LBM_CORE_RADIUS
                {
                    continue;
                }
                let idx = flatten_2d_index(x as usize, y as usize, LBM_WIDTH);
                if mask[idx] != is_obstacle {
                    mask[idx] = is_obstacle;
                    any_changed = true;
                }
            }
        }
    }
    any_changed
}

pub(super) fn lbm_parameters() -> HashMap<String, ParameterConstraint> {
    let c = |min, max, step| ParameterConstraint { min, max, step };
    HashMap::from([
        ("viscosity".to_string(), c(0.00333, 0.5, 0.005)),
        ("angle_of_attack".to_string(), c(-25.0, 25.0, 1.0)),
        ("naca_profile".to_string(), c(0.0, 4412.0, 1.0)),
        ("field_view".to_string(), c(0.0, 1.0, 1.0)),
    ])
}

pub(super) fn lbm_parameter_label(name: &str) -> String {
    match name {
        "viscosity" => "Kinematic Viscosity (ν)".to_string(),
        "angle_of_attack" => "Angle of Attack α (°)".to_string(),
        "naca_profile" => "Obstacle / Airfoil Profile".to_string(),
        "field_view" => "Scalar Field View".to_string(),
        other => other.to_string(),
    }
}

pub(super) fn lbm_parameter_choices(name: &str) -> &'static [(&'static str, f64)] {
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

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "LatticeBoltzmannTool",
        display_name: "Lattice Boltzmann (Demo)",
        domain: "fluid_dynamics",
        tags: &["lbm", "d2q9", "bgk", "vortex street", "navier-stokes", "fluid"],
        build: || Box::new(UnifiedSimTool::<LbmUnified>::new()),
    }
}

pub(crate) fn default_lbm_obstacle_mask() -> Vec<bool> {
    let mut mask = vec![false; LBM_WIDTH * LBM_HEIGHT];
    for y in 0..LBM_HEIGHT {
        for x in 0..LBM_WIDTH {
            let dx = x as i32 - 30;
            let dy = y as i32 - 25;
            if dx * dx + dy * dy <= 25 {
                mask[flatten_2d_index(x, y, LBM_WIDTH)] = true;
            }
        }
    }
    mask
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

fn generate_airfoil_starter_json() -> &'static str {
    static AIRFOIL_PRESET: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    AIRFOIL_PRESET
        .get_or_init(|| {
            let mut mask = vec![false; LBM_WIDTH * LBM_HEIGHT];
            for (x, y) in crate::framework::generate_brush_cells(
                crate::framework::BrushStencil::Airfoil,
                crate::framework::SymmetryMode::None,
                (32, 25),
                (32, 25),
                4,
                LBM_WIDTH,
                LBM_HEIGHT,
            ) {
                mask[flatten_2d_index(x, y, LBM_WIDTH)] = true;
            }
            let rle = crate::framework::encode_bool_grid(LBM_WIDTH, LBM_HEIGHT, &mask)
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
