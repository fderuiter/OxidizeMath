#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:graph_parameters_rust]

use eframe::egui;
use egui_plot::ExportPrimitive;
use math_explorer::pure_math::analysis::pde::wave_2d::{
    apply_biconvex_lens, apply_circular_drumhead_mask, apply_multi_slit_barrier, apply_point_clamp,
    apply_sponge_layer, BoundaryMode2D, ChladniSandSystem, DriverConfig2D, MarchingSquares2D,
    SpectralAnalyzer2D, SpectralPeak2D, Wave2DMedium, Wave2DParams, Wave2DSolver,
};

use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};

mod canvas;
mod presets;
mod ui;

#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

pub(crate) const GRID_SIZE: usize = 128;
pub(crate) const SAND_COUNT: usize = 4_000;
pub(crate) const FREQ_MIN: f64 = 0.02;
pub(crate) const FREQ_MAX: f64 = 40.0;
pub(crate) const AMP_MIN: f64 = 0.0;
pub(crate) const AMP_MAX: f64 = 200.0;
pub(crate) const SPEED_MIN: f64 = 0.0;
pub(crate) const SPEED_MAX: f64 = 5.0;
pub(crate) const STIFFNESS_MIN: f64 = 0.0;
pub(crate) const STIFFNESS_MAX: f64 = 0.05;
pub(crate) const DAMPING_MIN: f64 = 0.0;
pub(crate) const DAMPING_MAX: f64 = 5.0;
pub(crate) const DT_MIN: f64 = 1.0e-4;
pub(crate) const DT_MAX: f64 = 0.65;
pub(crate) const SLIT_SEP_MIN: f64 = 6.0;
pub(crate) const SLIT_SEP_MAX: f64 = 64.0;
pub(crate) const REFRACTIVE_MIN: f64 = 0.6;
pub(crate) const REFRACTIVE_MAX: f64 = 3.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum TankPreset {
    SquareChladniPlate,
    CircularDrumhead,
    DoubleSlitInterference,
    AcousticLens,
}

impl TankPreset {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::SquareChladniPlate => "Square Chladni Plate",
            Self::CircularDrumhead => "Circular Drumhead",
            Self::DoubleSlitInterference => "Double-Slit Interference",
            Self::AcousticLens => "Acoustic Lens",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum ClampLayout {
    None,
    CenterPin,
    QuadSymmetry,
    EdgeMidpoints,
}

impl ClampLayout {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::None => "Free / Perimeter Only",
            Self::CenterPin => "Center Nodal Pin",
            Self::QuadSymmetry => "Fourfold Symmetry Pins",
            Self::EdgeMidpoints => "Edge Midpoint Clamps",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum WaveColormap {
    PerceptualThermal,
    BipolarPhase,
    WcagHighContrast,
}

impl WaveColormap {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::PerceptualThermal => "Perceptual Thermal",
            Self::BipolarPhase => "Bipolar Wave Phase",
            Self::WcagHighContrast => "WCAG High Contrast",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum SecondaryPlotMode {
    DetectorCrossSection,
    RadialFftSpectrum,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChladniWaveTankState {
    pub preset: TankPreset,
    pub medium: Wave2DMedium,
    pub boundary: BoundaryMode2D,
    pub clamp_layout: ClampLayout,
    pub colormap: WaveColormap,
    pub secondary_plot: SecondaryPlotMode,
    pub wave_speed: f64,
    pub plate_stiffness: f64,
    pub damping: f64,
    pub time_step: f64,
    pub driver_enabled: bool,
    pub line_source: bool,
    pub driver_x: f64,
    pub driver_y: f64,
    pub driver_frequency: f64,
    pub driver_amplitude: f64,
    pub barrier_enabled: bool,
    pub slit_count: usize,
    pub slit_width: usize,
    pub slit_separation: f64,
    pub barrier_x: usize,
    pub detector_x: usize,
    pub lens_enabled: bool,
    pub refractive_index: f64,
    pub lens_aperture: f64,
    pub show_nodal_contours: bool,
    pub show_sand_particles: bool,
    pub rng_seed: u64,
}

pub struct ChladniWaveTankTool {
    pub(crate) state: ChladniWaveTankState,
    pub(crate) custom_clamps: Vec<[f64; 2]>,
    pub(crate) solver: Wave2DSolver,
    pub(crate) sand: ChladniSandSystem,
    pub(crate) fft: SpectralAnalyzer2D,
    pub(crate) contours: MarchingSquares2D,
    pub(crate) detector_buf: Vec<f64>,
    pub(crate) radial_spectrum: Vec<f64>,
    pub(crate) radial_counts: Vec<usize>,
    pub(crate) dominant_peak: SpectralPeak2D,
    pub(crate) rgba_pixels: Vec<egui::Color32>,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) paused: bool,
    pub(crate) steps_per_frame: usize,
    pub(crate) last_step_ms: f64,
    pub(crate) last_substeps: usize,
    pub(crate) state_rev: u64,
    pub(crate) focused_canvas: Option<egui::Id>,
    pub(crate) analysis_dirty: bool,
}

impl Default for ChladniWaveTankTool {
    fn default() -> Self {
        let state = presets::default_state_for_preset(TankPreset::SquareChladniPlate);
        let solver = Wave2DSolver::new_unit_square(GRID_SIZE);
        let sand = ChladniSandSystem::new(
            SAND_COUNT,
            state.rng_seed,
            GRID_SIZE,
            GRID_SIZE,
            &solver.wall_mask,
        );
        let mut tool = Self {
            state,
            custom_clamps: Vec::with_capacity(16),
            solver,
            sand,
            fft: SpectralAnalyzer2D::new(GRID_SIZE, GRID_SIZE),
            contours: MarchingSquares2D::new(8_192),
            detector_buf: vec![0.0; GRID_SIZE],
            radial_spectrum: vec![0.0; GRID_SIZE / 2],
            radial_counts: vec![0; GRID_SIZE / 2],
            dominant_peak: SpectralPeak2D::default(),
            rgba_pixels: vec![egui::Color32::BLACK; GRID_SIZE * GRID_SIZE],
            texture: None,
            paused: false,
            steps_per_frame: 2,
            last_step_ms: 0.0,
            last_substeps: 1,
            state_rev: 1,
            focused_canvas: None,
            analysis_dirty: true,
        };
        tool.rebuild_domain_and_warmup(0);
        tool
    }
}

impl ChladniWaveTankTool {
    pub(crate) fn pde_params(&self) -> Wave2DParams {
        Wave2DParams {
            medium: self.state.medium,
            boundary: self.state.boundary,
            wave_speed: self.state.wave_speed,
            plate_stiffness: self.state.plate_stiffness,
            damping: self.state.damping,
        }
    }

    pub(crate) fn driver_config(&self) -> DriverConfig2D {
        DriverConfig2D {
            enabled: self.state.driver_enabled,
            line_source: self.state.line_source,
            x_norm: self.state.driver_x,
            y_norm: self.state.driver_y,
            frequency: self.state.driver_frequency,
            amplitude: self.state.driver_amplitude,
            radius_cells: 2.0,
        }
    }

    pub(crate) fn apply_preset(&mut self, preset: TankPreset) {
        self.custom_clamps.clear();
        self.state = presets::default_state_for_preset(preset);
        self.state_rev = self.state_rev.wrapping_add(1);
        let warmup = match preset {
            TankPreset::DoubleSlitInterference | TankPreset::AcousticLens => 24,
            TankPreset::SquareChladniPlate | TankPreset::CircularDrumhead => 6,
        };
        self.rebuild_domain_and_warmup(warmup);
    }

    pub(crate) fn rebuild_domain_and_warmup(&mut self, warmup_steps: usize) {
        let cell_spacing =
            if self.state.barrier_enabled || self.state.lens_enabled || self.state.boundary == BoundaryMode2D::Sponge {
                1.0
            } else {
                1.0 / ((GRID_SIZE - 1) as f64)
            };
        if (self.solver.dx() - cell_spacing).abs() > 1.0e-12 {
            self.solver = Wave2DSolver::new(GRID_SIZE, GRID_SIZE, cell_spacing, cell_spacing);
        } else {
            self.solver.reset_all();
        }
        self.rebuild_masks();
        presets::seed_initial_resonant_field(&mut self.solver, &self.state);
        self.sand.reseed(
            self.state.rng_seed,
            GRID_SIZE,
            GRID_SIZE,
            &self.solver.wall_mask,
        );
        if warmup_steps > 0 {
            for _ in 0..warmup_steps {
                self.advance_physics_step();
            }
            self.refresh_analysis_outputs();
        } else {
            self.analysis_dirty = true;
        }
    }

    pub(crate) fn rebuild_masks(&mut self) {
        self.solver.wall_mask.fill(1.0);
        self.solver.sponge_mask.fill(0.0);
        self.solver.wave_speed_mask.fill(1.0);

        if self.state.preset == TankPreset::CircularDrumhead {
            apply_circular_drumhead_mask(&mut self.solver.wall_mask, GRID_SIZE, GRID_SIZE, 0.45);
        }
        if self.state.boundary == BoundaryMode2D::Sponge {
            apply_sponge_layer(&mut self.solver.sponge_mask, GRID_SIZE, GRID_SIZE, 12, 1.8);
        }
        match self.state.clamp_layout {
            ClampLayout::None => {}
            ClampLayout::CenterPin => {
                apply_point_clamp(&mut self.solver.wall_mask, GRID_SIZE, GRID_SIZE, 0.5, 0.5, 2.0);
            }
            ClampLayout::QuadSymmetry => {
                for (cx, cy) in [(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)] {
                    apply_point_clamp(&mut self.solver.wall_mask, GRID_SIZE, GRID_SIZE, cx, cy, 1.8);
                }
            }
            ClampLayout::EdgeMidpoints => {
                for (cx, cy) in [(0.5, 0.04), (0.5, 0.96), (0.04, 0.5), (0.96, 0.5)] {
                    apply_point_clamp(&mut self.solver.wall_mask, GRID_SIZE, GRID_SIZE, cx, cy, 2.2);
                }
            }
        }
        for &[cx, cy] in &self.custom_clamps {
            apply_point_clamp(&mut self.solver.wall_mask, GRID_SIZE, GRID_SIZE, cx, cy, 1.8);
        }
        if self.state.barrier_enabled {
            apply_multi_slit_barrier(
                &mut self.solver.wall_mask,
                GRID_SIZE,
                GRID_SIZE,
                self.state.barrier_x,
                2,
                self.state.slit_count,
                self.state.slit_width,
                self.state.slit_separation,
            );
        }
        if self.state.lens_enabled {
            apply_biconvex_lens(
                &mut self.solver.wave_speed_mask,
                GRID_SIZE,
                GRID_SIZE,
                44,
                self.state.lens_aperture,
                8.0,
                self.state.refractive_index,
            );
        }
    }

    pub(crate) fn advance_physics_step(&mut self) {
        let start = crate::web_io::PlatformInstant::now();
        let params = self.pde_params();
        let driver = self.driver_config();
        let det_col = Some(self.state.detector_x.min(GRID_SIZE - 2));
        self.last_substeps =
            self.solver
                .step_driven(&params, &driver, self.state.time_step, det_col);
        if self.state.show_sand_particles {
            self.sand.step(
                self.solver.energy_field(),
                &self.solver.wall_mask,
                GRID_SIZE,
                GRID_SIZE,
                14.0,
                0.30,
                0.15,
            );
        }
        self.last_step_ms = start.elapsed().as_secs_f64() * 1_000.0;
    }

    pub(crate) fn refresh_analysis_outputs(&mut self) {
        self.analysis_dirty = false;
        self.solver
            .read_detector_profile(self.state.detector_x, &mut self.detector_buf);
        self.fft.compute_spectrum(self.solver.displacement());
        self.dominant_peak = self.fft.dominant_peak();
        self.fft
            .compute_radial_profile(&mut self.radial_spectrum, &mut self.radial_counts);
        if self.state.show_nodal_contours {
            let field = if self.state.driver_enabled && !self.state.line_source {
                self.solver.modal_field()
            } else {
                self.solver.displacement()
            };
            self.contours.extract_zero_contours(
                field,
                &self.solver.wall_mask,
                GRID_SIZE,
                GRID_SIZE,
                1.0e-5,
            );
        }
    }
}

impl InteractiveTool for ChladniWaveTankTool {
    fn name(&self) -> &'static str {
        "Wave Tank & 2D Chladni Plate"
    }

    fn domain(&self) -> &'static str {
        "analysis"
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
            max_bytes: 8_192,
            max_depth: 6,
            max_array_len: 32,
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
        let parsed: ChladniWaveTankState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid Chladni Wave Tank state JSON: {e}"))?;
        if !presets::is_valid_state(&parsed) {
            return Err("Chladni Wave Tank parameters out of finite valid bounds".into());
        }
        self.state = parsed;
        self.custom_clamps.clear();
        self.state_rev = self.state_rev.wrapping_add(1);
        self.rebuild_domain_and_warmup(0);
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
        self.advance_physics_step();
        self.refresh_analysis_outputs();
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
            for _ in 0..self.steps_per_frame {
                self.advance_physics_step();
            }
            self.refresh_analysis_outputs();
            parent_ui.ctx().request_repaint();
        } else if self.analysis_dirty {
            self.refresh_analysis_outputs();
        }
        self.draw_central_workspace(parent_ui);
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "ChladniWaveTankTool",
        display_name: "Wave Tank & 2D Chladni Plate",
        domain: "analysis",
        tags: &["pde", "chladni", "waves", "diffraction", "fft"],
        build: || Box::new(ChladniWaveTankTool::default()),
    }
}
