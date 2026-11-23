#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::LbmUnified;
use crate::async_sim::unified::UnifiedSimTool;
use eframe::egui::Color32;
use math_commons::math_kernel::colormap::heatmap_color;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    BgkCollision, LatticeBoltzmannD2Q9,
};
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

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

pub(super) fn render_lbm_snapshot_buffers(
    solver: &LatticeBoltzmannD2Q9<BgkCollision>,
    vorticity_buf: &[f64],
    display_vorticity: bool,
) -> (Vec<Color32>, Vec<f64>, f64) {
    let width = solver.width();
    let height = solver.height();
    let mut pixels = vec![Color32::BLACK; width * height];
    let mut scalar_values = vec![0.0_f64; width * height];
    let mut max_vort = 0.0_f64;

    for y in 0..height {
        let solver_y = height - 1 - y;
        for x in 0..width {
            let pixel_idx = flatten_2d_index(x, y, width);
            let solver_idx = flatten_2d_index(x, solver_y, width);
            if solver.is_obstacle(x, solver_y) {
                pixels[pixel_idx] = Color32::from_gray(50);
                scalar_values[pixel_idx] = -0.01;
            } else {
                let omega = vorticity_buf[solver_idx];
                let safe_w = if omega.is_finite() { omega } else { 0.0 };
                max_vort = max_vort.max(safe_w.abs());
                let norm = if display_vorticity {
                    scalar_values[pixel_idx] = safe_w;
                    (0.5 + safe_w * 12.0).clamp(0.0, 1.0)
                } else {
                    let v = solver.get_velocity_magnitude(x, solver_y);
                    let safe_v = if v.is_finite() { v } else { 0.0 };
                    scalar_values[pixel_idx] = safe_v;
                    (safe_v * 1000.0 / 255.0).clamp(0.0, 1.0)
                };
                let rgb = heatmap_color(norm);
                pixels[pixel_idx] = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
            }
        }
    }
    (pixels, scalar_values, max_vort)
}
