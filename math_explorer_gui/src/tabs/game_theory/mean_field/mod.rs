//! Interactive 1D Mean-Field Game HJB–Fokker-Planck Workbench (`MeanFieldGameTool`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_explorer::applied::game_theory::mean_field::{
    Density, FixedPointSolver, MFGConfigBuilder, MFGSolver, Position,
};
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;

pub mod presets;
mod ui;

#[cfg(test)]
mod tests;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MeanFieldConfig {
    /// Crowd aversion coupling weight (first scalar float for perturbation tests).
    pub congestion_weight: f64,
    pub viscosity: f64,
    pub time_horizon: f64,
    pub terminal_attraction: f64,
    pub target_center: f64,
    pub bimodal_initial: bool,
    pub grid_points: usize,
    pub iterations: usize,
    pub slice_fraction: f64,
}

impl MeanFieldConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.congestion_weight.is_finite() || !(0.0..=30.0).contains(&self.congestion_weight) {
            return Err("congestion_weight must be finite and in [0, 30]".into());
        }
        if !self.viscosity.is_finite() || !(0.015..=0.30).contains(&self.viscosity) {
            return Err("viscosity must be finite and in [0.015, 0.30]".into());
        }
        if !self.time_horizon.is_finite() || !(0.10..=0.80).contains(&self.time_horizon) {
            return Err("time_horizon must be finite and in [0.10, 0.80]".into());
        }
        if !self.terminal_attraction.is_finite()
            || !(0.05..=10.0).contains(&self.terminal_attraction)
        {
            return Err("terminal_attraction must be finite and in [0.05, 10.0]".into());
        }
        if !self.target_center.is_finite() || !(-0.8..=0.8).contains(&self.target_center) {
            return Err("target_center must be finite and in [-0.8, 0.8]".into());
        }
        if !(17..=41).contains(&self.grid_points) {
            return Err("grid_points must be in 17..=41".into());
        }
        if !(1..=15).contains(&self.iterations) {
            return Err("iterations must be in 1..=15".into());
        }
        if !self.slice_fraction.is_finite() || !(0.0..=1.0).contains(&self.slice_fraction) {
            return Err("slice_fraction must be finite and in [0, 1]".into());
        }
        Ok(())
    }
}

pub struct MeanFieldGameTool {
    pub(crate) cfg: MeanFieldConfig,
    pub(crate) cfl_time_steps: usize,
    pub(crate) density_initial: Vec<[f64; 2]>,
    pub(crate) density_slice: Vec<[f64; 2]>,
    pub(crate) density_terminal: Vec<[f64; 2]>,
    pub(crate) value_slice: Vec<[f64; 2]>,
    pub(crate) peak_terminal_density: f64,
    pub(crate) origin_initial_value: f64,
    pub(crate) terminal_variance: f64,
    state_rev: u64,
}

impl Default for MeanFieldGameTool {
    fn default() -> Self {
        let mut tool = Self {
            cfg: presets::default_config(),
            cfl_time_steps: 100,
            density_initial: Vec::new(),
            density_slice: Vec::new(),
            density_terminal: Vec::new(),
            value_slice: Vec::new(),
            peak_terminal_density: 0.0,
            origin_initial_value: 0.0,
            terminal_variance: 0.0,
            state_rev: 1,
        };
        tool.recompute();
        tool
    }
}

impl MeanFieldGameTool {
    pub(crate) fn recompute(&mut self) {
        let nx = self.cfg.grid_points.clamp(17, 41);
        let dx = 2.0 / ((nx - 1) as f64);
        let cfl_nt =
            ((8.0 * self.cfg.viscosity * self.cfg.time_horizon) / (dx * dx)).ceil() as usize;
        let nt = cfl_nt.clamp(80, 480);
        self.cfl_time_steps = nt;

        let (Some(nz_nx), Some(nz_nt)) = (NonZeroUsize::new(nx), NonZeroUsize::new(nt)) else {
            return;
        };
        let Ok(builder) = MFGConfigBuilder::new()
            .viscosity(self.cfg.viscosity)
            .time_horizon(self.cfg.time_horizon)
            .grid_points(nz_nx)
            .time_steps(nz_nt)
            .space_bounds(-1.0, 1.0)
        else {
            return;
        };
        let Ok(mfg_cfg) = builder.build() else {
            return;
        };

        let cong = self.cfg.congestion_weight;
        let target = self.cfg.target_center;
        let term_w = self.cfg.terminal_attraction;
        let bimodal = self.cfg.bimodal_initial;

        let cost_fn = |p: Position, d: Density| -> f64 {
            let diff = p.0 - target;
            cong * d.0 + 0.5 * diff * diff
        };
        let term_fn = |p: Position, _d: Density| -> f64 {
            let diff = p.0 - target;
            term_w * diff * diff
        };
        let init_dist = |p: Position| -> f64 {
            if bimodal {
                (-18.0 * (p.0 + 0.45).powi(2)).exp() + (-18.0 * (p.0 - 0.45).powi(2)).exp()
            } else {
                (-12.0 * p.0 * p.0).exp()
            }
        };

        let solver = FixedPointSolver::new(self.cfg.iterations);
        let (u, m) = solver.solve(&mfg_cfg, &cost_fn, &term_fn, &init_dist);
        self.extract_profiles(&mfg_cfg, &u, &m);
        self.state_rev = self.state_rev.wrapping_add(1).max(1);
    }

    fn extract_profiles(
        &mut self,
        mfg_cfg: &math_explorer::applied::game_theory::mean_field::MFGConfig,
        u: &nalgebra::DMatrix<f64>,
        m: &nalgebra::DMatrix<f64>,
    ) {
        let nx = mfg_cfg.grid_points.get();
        let nt = mfg_cfg.time_steps.get();
        let slice_col = ((self.cfg.slice_fraction * (nt as f64)).round() as usize).min(nt);
        let inv_dx = 1.0 / mfg_cfg.dx;

        self.density_initial.clear();
        self.density_slice.clear();
        self.density_terminal.clear();
        self.value_slice.clear();

        let mut peak_m = 0.0_f64;
        let mut mean_x = 0.0;
        for i in 0..nx {
            let x = mfg_cfg.space_min + (i as f64) * mfg_cfg.dx;
            let m0 = m[(i, 0)] * inv_dx;
            let ms = m[(i, slice_col)] * inv_dx;
            let mt = m[(i, nt)] * inv_dx;
            let us = u[(i, slice_col)];
            peak_m = peak_m.max(mt);
            mean_x += x * m[(i, nt)];
            self.density_initial.push([x, m0]);
            self.density_slice.push([x, ms]);
            self.density_terminal.push([x, mt]);
            self.value_slice.push([x, us]);
        }
        let mut var_x = 0.0;
        for i in 0..nx {
            let x = mfg_cfg.space_min + (i as f64) * mfg_cfg.dx;
            var_x += (x - mean_x).powi(2) * m[(i, nt)];
        }
        self.peak_terminal_density = peak_m;
        self.origin_initial_value = u[(nx / 2, 0)];
        self.terminal_variance = var_x;
    }
}

impl InteractiveTool for MeanFieldGameTool {
    fn name(&self) -> &'static str {
        "Mean-Field Crowd Games"
    }

    fn domain(&self) -> &'static str {
        "game_theory"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn state_revision(&self) -> u64 {
        self.state_rev
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::new(1024, 4, 64)
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("peak_terminal_density", self.peak_terminal_density),
            ("origin_initial_value", self.origin_initial_value),
            ("terminal_variance", self.terminal_variance),
            ("cfl_time_steps", self.cfl_time_steps as f64),
        ]
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.cfg).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let cfg: MeanFieldConfig = serde_json::from_str(state)
            .map_err(|e| format!("Invalid MeanFieldGameTool state: {e}"))?;
        cfg.validate()?;
        self.cfg = cfg;
        self.recompute();
        Ok(())
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        vec![
            (
                "Initial Density m(x, 0)".to_string(),
                self.density_initial.clone(),
            ),
            (
                "Slice Density m(x, t)".to_string(),
                self.density_slice.clone(),
            ),
            (
                "Terminal Density m(x, T)".to_string(),
                self.density_terminal.clone(),
            ),
            (
                "HJB Value Function u(x, t)".to_string(),
                self.value_slice.clone(),
            ),
        ]
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let side_id = crate::framework::scoped_panel_id(parent_ui, "mean_field_game_side");
        crate::framework::PaneSidePanel::left(side_id).show_inside(parent_ui, |ui| {
            self.show_controls(ui);
        });
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.show_main_view(ui);
        });
    }
}

impl scientific_metadata::theory::TheoryDescribable for MeanFieldGameTool {
    fn theory_description(&self) -> String {
        "Solves coupled 1D Mean-Field Game PDEs in the continuum $N \\to \\infty$ player limit: backward Hamilton-Jacobi-Bellman $-\
         \\partial_t u - \\nu \\partial_{xx} u + \\frac{1}{2}(\\partial_x u)^2 = F(x, m)$ for optimal agent control and forward Fokker-Planck $\\partial_t m - \\nu \\partial_{xx} m - \\partial_x(m \\partial_x u) = 0$ for crowd density evolution.".into()
    }

    fn phonetic_description(&self) -> String {
        "Mean field game coupled Hamilton Jacobi Bellman and Fokker Planck crowd dynamics workbench"
            .into()
    }

    fn theory_citation(&self) -> String {
        "Mean Field Games (Lasry & Lions, Japanese Journal of Mathematics, 2007); Large Population Stochastic Dynamic Games (Huang, Malhamé & Caines, 2006)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "MeanFieldGameTool",
        display_name: "Mean-Field Crowd Games",
        domain: "game_theory",
        tags: &["mean-field", "mfg", "hjb", "fokker-planck", "congestion", "crowd"],
        build: || Box::new(MeanFieldGameTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "mfg_high_congestion_repulsion",
        title: "Crowd-Averse Congestion Dispersion",
        description: "F(x, m) = 6.0 m + ½x²: Strong crowd repulsion flattens terminal density m(x, T).",
        category: "Game Theory",
        icon: "🌊",
        domain: "game_theory",
        tool_name: Some("MeanFieldGameTool"),
        preset_generator: || presets::PRESET_HIGH_CONGESTION,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "mfg_bimodal_corridor_coalescence",
        title: "Bimodal Crowd Coalescence to Target",
        description: "Two separated initial sub-populations steer toward x* = 0.25 under HJB drift −∂_x u.",
        category: "Game Theory",
        icon: "👥",
        domain: "game_theory",
        tool_name: Some("MeanFieldGameTool"),
        preset_generator: || presets::PRESET_BIMODAL_COALESCENCE,
    }
}
