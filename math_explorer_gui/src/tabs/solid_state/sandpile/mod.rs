#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Abelian Sandpile Self-Organized Criticality & Group Studio (`AbelianSandpileStudio`).

use eframe::egui;
use egui_plot::ExportPrimitive;
use math_explorer::fit::{CurveFit, FitModel};
use math_explorer::physics::stat_mech::sandpile::{
    AvalancheObservable, BurningTestResult, SandpileGrid, MAX_POWER_LAW_FIT_POINTS,
    MAX_STABLE_HEIGHT,
};

use crate::audio::frame::SonificationFrame;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};

pub mod audio;
mod canvas;
mod presets;
mod ui;

#[cfg(test)]
mod tests;

/// Minimum interactive grid dimension (`3` allows inspecting exact Bareiss Kirchhoff determinants).
pub(crate) const MIN_SANDPILE_DIM: usize = 3;

/// Guided presets covering `1/f` avalanches, `63x63` Creutz crystals, and `(r + e)^o = r` proofs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum SandpilePreset {
    CriticalAvalanches1F,
    CreutzIdentityCrystal63,
    AbelianGroupIdentityProof,
}

impl SandpilePreset {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::CriticalAvalanches1F => "1/f Critical Avalanche Cascades",
            Self::CreutzIdentityCrystal63 => "63×63 Recurrent Identity Crystal",
            Self::AbelianGroupIdentityProof => "(r + e)° = r Abelian Group Proof",
        }
    }
}

/// Interactive brush geometry for manual grain perturbations on the 2D lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum SandpileBrush {
    SingleGrain,
    CriticalTower,
    SymmetricCross,
}

impl SandpileBrush {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::SingleGrain => "Single Cell Drop",
            Self::CriticalTower => "Critical Tower Burst",
            Self::SymmetricCross => "4-Fold Symmetric Cross",
        }
    }
}

/// Primary field rendered on the 2D sandpile crystal canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum SandpileViewMode {
    GrainHeights,
    DharBurningWaves,
}

impl SandpileViewMode {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::GrainHeights => "Grain Heights z(x, y)",
            Self::DharBurningWaves => "Dhar Burning Wave Depth",
        }
    }
}

/// Shared accessible colormaps backed by `math_commons::math_kernel::colormap`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum SandpileColormap {
    CriticalThermal,
    ApollonianHighContrast,
    BurningWavePhase,
}

impl SandpileColormap {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::CriticalThermal => "Perceptual Thermal",
            Self::ApollonianHighContrast => "WCAG High-Contrast Crystal",
            Self::BurningWavePhase => "Cyclic Burning Wave",
        }
    }
}

/// Serializable URL/preset state for [`AbelianSandpileStudio`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SandpileStudioState {
    pub preset: SandpilePreset,
    pub grid_size: usize,
    pub drops_per_frame: usize,
    pub animate_wavefronts: bool,
    pub brush: SandpileBrush,
    pub brush_grains: u32,
    pub view_mode: SandpileViewMode,
    pub colormap: SandpileColormap,
    pub observable: AvalancheObservable,
    pub show_burning_overlay: bool,
    pub cursor_x: usize,
    pub cursor_y: usize,
    pub rng_seed: u64,
}

/// Interactive Solid-State studio uniting 2D Bak-Tang-Wiesenfeld `1/f` avalanches and
/// finite Abelian sandpile group algebra.
pub struct AbelianSandpileStudio {
    pub(crate) state: SandpileStudioState,
    pub(crate) grid: SandpileGrid,
    pub(crate) identity_cache: Vec<u32>,
    pub(crate) register_r: Vec<u32>,
    pub(crate) burning_result: BurningTestResult,
    pub(crate) latest_fit: Option<CurveFit>,
    pub(crate) survival_points: [[f64; 2]; MAX_POWER_LAW_FIT_POINTS],
    pub(crate) survival_len: usize,
    pub(crate) latest_sonification: SonificationFrame,
    pub(crate) group_status: String,
    pub(crate) subgroup_summary: String,
    pub(crate) rgba_pixels: Vec<egui::Color32>,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) paused: bool,
    pub(crate) last_wavefront_sites: usize,
    pub(crate) state_rev: u64,
    pub(crate) focused_canvas: Option<egui::Id>,
}

impl Default for AbelianSandpileStudio {
    fn default() -> Self {
        let state = presets::default_state_for_preset(SandpilePreset::CriticalAvalanches1F);
        let n = state.grid_size;
        let grid = SandpileGrid::new(n, n, Some(state.rng_seed)).expect("valid default sandpile");
        let mut studio = Self {
            state,
            grid,
            identity_cache: vec![0; n * n],
            register_r: vec![0; n * n],
            burning_result: BurningTestResult::default(),
            latest_fit: None,
            survival_points: [[0.0; 2]; MAX_POWER_LAW_FIT_POINTS],
            survival_len: 0,
            latest_sonification: SonificationFrame::silent(),
            group_status: String::new(),
            subgroup_summary: String::new(),
            rgba_pixels: vec![egui::Color32::BLACK; n * n],
            texture: None,
            paused: false,
            last_wavefront_sites: 0,
            state_rev: 1,
            focused_canvas: None,
        };
        studio.rebuild_for_current_state();
        studio
    }
}

impl AbelianSandpileStudio {
    pub(crate) fn apply_preset(&mut self, preset: SandpilePreset) {
        self.state = presets::default_state_for_preset(preset);
        self.paused = matches!(
            preset,
            SandpilePreset::CreutzIdentityCrystal63 | SandpilePreset::AbelianGroupIdentityProof
        );
        self.state_rev = self.state_rev.wrapping_add(1);
        self.rebuild_for_current_state();
    }

    pub(crate) fn rebuild_for_current_state(&mut self) {
        let n = self.state.grid_size;
        self.grid =
            SandpileGrid::new(n, n, Some(self.state.rng_seed)).expect("validated grid_size");
        self.rgba_pixels.resize(n * n, egui::Color32::BLACK);
        self.identity_cache.resize(n * n, 0);
        self.register_r.resize(n * n, 0);
        self.texture = None;

        // Warm up critical avalanche statistics first so every preset has immediate PowerLaw plots.
        self.seed_critical_attractor_warmup();

        match self.state.preset {
            SandpilePreset::CriticalAvalanches1F => {
                self.capture_identity_into_cache();
                self.register_r.copy_from_slice(self.grid.heights());
                self.group_status = "Critical attractor active • ready for group probes".into();
            }
            SandpilePreset::CreutzIdentityCrystal63 => {
                let topples = self.grid.load_creutz_identity();
                self.identity_cache.copy_from_slice(self.grid.heights());
                self.register_r.copy_from_slice(self.grid.heights());
                self.group_status = format!(
                    "Creutz 63×63 identity e converged in {topples} topples • (e + e)° = e"
                );
            }
            SandpilePreset::AbelianGroupIdentityProof => {
                self.capture_identity_into_cache();
                self.register_r.copy_from_slice(self.grid.heights());
                self.verify_identity_law_action();
            }
        }
        self.refresh_zn_subgroup_summary();
        self.refresh_analysis_and_audio();
    }

    fn seed_critical_attractor_warmup(&mut self) {
        self.grid.reseed(self.state.rng_seed);
        self.grid.fill(MAX_STABLE_HEIGHT);
        let burn_in = if self.state.grid_size > 48 { 140 } else { 220 };
        let sample_drops = if self.state.grid_size > 48 {
            650
        } else {
            1_400
        };
        for _ in 0..burn_in {
            let _ = self.grid.drop_random_grain();
            let _ = self.grid.stabilize_wavefront();
        }
        self.grid.telemetry_mut().clear();
        for _ in 0..sample_drops {
            let _ = self.grid.drop_random_grain();
            let _ = self.grid.stabilize_wavefront();
        }
    }

    pub(super) fn capture_identity_into_cache(&mut self) {
        let saved: Vec<u32> = self.grid.heights().to_vec();
        let _ = self.grid.load_creutz_identity();
        self.identity_cache.copy_from_slice(self.grid.heights());
        let _ = self.grid.set_heights(&saved);
    }

    pub(crate) fn advance_simulation_step(&mut self) {
        if self.state.animate_wavefronts && self.grid.has_active_wavefront() {
            let (sites, _diss) = self.grid.step_wavefront();
            self.last_wavefront_sites = sites;
        } else {
            let mut peak_wave = 0usize;
            for _ in 0..self.state.drops_per_frame {
                let (rx, ry) = self.grid.drop_random_grain();
                self.state.cursor_x = rx;
                self.state.cursor_y = ry;
                if self.state.animate_wavefronts && self.grid.has_active_wavefront() {
                    let (sites, _) = self.grid.step_wavefront();
                    peak_wave = peak_wave.max(sites);
                    break;
                }
                let stats = self.grid.stabilize_wavefront();
                peak_wave = peak_wave.max(stats.peak_wavefront);
            }
            self.last_wavefront_sites = peak_wave;
        }
        self.refresh_analysis_and_audio();
    }

    pub(crate) fn apply_brush_at(&mut self, cx: usize, cy: usize) {
        let n = self.state.grid_size;
        if cx >= n || cy >= n {
            return;
        }
        let g = self.state.brush_grains;
        match self.state.brush {
            SandpileBrush::SingleGrain => {
                let _ = self.grid.add_grains(cx, cy, g);
            }
            SandpileBrush::CriticalTower => {
                let _ = self.grid.add_grains(cx, cy, g.saturating_mul(4));
            }
            SandpileBrush::SymmetricCross => {
                let mx = (n - 1) - cx;
                let my = (n - 1) - cy;
                for (px, py) in [(cx, cy), (mx, cy), (cx, my), (mx, my)] {
                    let _ = self.grid.add_grains(px, py, g);
                }
            }
        }
        if !self.state.animate_wavefronts {
            let stats = self.grid.stabilize_wavefront();
            self.last_wavefront_sites = stats.peak_wavefront;
        }
        self.refresh_analysis_and_audio();
    }

    pub(crate) fn refresh_analysis_and_audio(&mut self) {
        if self.state.view_mode == SandpileViewMode::DharBurningWaves
            || self.state.show_burning_overlay
            || self.paused
        {
            self.burning_result = self.grid.dhar_burning_test();
        }
        let cutoff = self.grid.scaling_cutoff(self.state.observable);
        self.survival_len = self.grid.telemetry().survival_curve_into(
            self.state.observable,
            cutoff,
            &mut self.survival_points,
        );
        self.latest_fit = self
            .grid
            .fit_avalanche_power_law(self.state.observable)
            .ok();
        let slope = self
            .latest_fit
            .as_ref()
            .filter(|f| f.fit_model == FitModel::PowerLaw)
            .map_or(-0.25, |f| f.coefficients()[1]);
        let last = self.grid.telemetry().last_stats();
        self.latest_sonification = audio::sandpile_sonification_frame(
            self.last_wavefront_sites,
            last.topples,
            last.area,
            self.state.grid_size * self.state.grid_size,
            slope,
        );
    }
}

impl InteractiveTool for AbelianSandpileStudio {
    fn name(&self) -> &'static str {
        "Abelian Sandpile Studio"
    }

    fn domain(&self) -> &'static str {
        "solid_state"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn state_revision(&self) -> u64 {
        self.state_rev
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds {
            max_bytes: 4_096,
            max_depth: 5,
            max_array_len: 16,
        }
    }

    fn has_finite_parameters(&self) -> bool {
        presets::is_valid_state(&self.state)
    }

    fn clamp_parameters(&mut self) -> bool {
        presets::clamp_state_fields(&mut self.state);
        self.state_rev = self.state_rev.wrapping_add(1);
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: SandpileStudioState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid Abelian Sandpile Studio state JSON: {e}"))?;
        if !presets::is_valid_state(&parsed) {
            return Err("Abelian Sandpile Studio parameters out of valid bounds".into());
        }
        self.state = parsed;
        self.paused = matches!(
            self.state.preset,
            SandpilePreset::CreutzIdentityCrystal63 | SandpilePreset::AbelianGroupIdentityProof
        );
        self.state_rev = self.state_rev.wrapping_add(1);
        self.rebuild_for_current_state();
        Ok(())
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_raster_image(&self) -> Option<egui::ColorImage> {
        Some(self.build_export_color_image())
    }

    fn export_plot_primitives(&self) -> Vec<ExportPrimitive> {
        self.build_export_primitives()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_export_datasets()
    }

    fn step_simulation(&mut self) {
        self.advance_simulation_step();
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
        self.draw_controls_panel(parent_ui);
        if !self.paused {
            self.advance_simulation_step();
            parent_ui.ctx().request_repaint();
        }
        self.draw_central_workspace(parent_ui);
    }
}
