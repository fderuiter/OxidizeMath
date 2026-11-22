#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use eframe::egui::ColorImage;
use math_commons::math_kernel::colormap::{
    ensure_dark_canvas_contrast, heatmap_color, heatmap_color_high_contrast,
};
use math_explorer::physics::optics::caustics::HamiltonianRayTracer;
use math_explorer::physics::optics::refraction::{OpticalMediumConfig, OpticsPreset};
use math_explorer::physics::optics::sensing::{
    sample_detector_line_in_place, DetectorConfig, DetectorReadout,
};
use math_explorer::pure_math::analysis::pde::wave::{WaveField2D, WaveFieldConfig2D};
use std::sync::Arc;

/// Shared thread-safe simulation parameters between [`super::tool::OpticsWaveTool`] and [`OpticsWaveRunner`].
#[derive(Clone, Debug)]
pub struct OpticsSharedParams {
    /// Active optical element or aperture preset.
    pub preset: OpticsPreset,
    /// Coherent source frequency $f$ (`4.0..=24.0`).
    pub frequency: f64,
    /// Double-slit center-to-center separation $d$ (`0.08..=0.40`).
    pub slit_separation: f64,
    /// Double-slit opening width $a$ (`0.015..=0.14`).
    pub aperture_width: f64,
    /// Lens radius $R$ (`0.10..=0.34`).
    pub lens_radius: f64,
    /// Virtual vertical detector x-coordinate (`0.45..=0.90`).
    pub detector_x: f64,
    /// Fraction of the 2D CFL stability step (`0.10..=1.00`).
    pub cfl_scale: f64,
    /// Whether to draw geometric ray and caustic cusp overlays.
    pub show_rays: bool,
    /// Whether to colorize time-smoothed irradiance $\langle u^2\rangle$ instead of wave amplitude $u$.
    pub show_irradiance: bool,
    /// Number of rays in the Hamiltonian ray bundle (`3..=31`).
    pub ray_count: usize,
    /// Quantizer bit depth for `lws_range_resolution` (`4..=16`).
    pub adc_bits: u32,
    /// Active theme mode for WCAG contrast-aware raster rendering.
    pub high_contrast: bool,
    /// Optional test-only fault injection flag.
    pub force_non_finite: bool,
}

impl Default for OpticsSharedParams {
    fn default() -> Self {
        Self {
            preset: OpticsPreset::LuneburgLens,
            frequency: 14.0,
            slit_separation: 0.18,
            aperture_width: 0.04,
            lens_radius: 0.22,
            detector_x: 0.68,
            cfl_scale: 0.95,
            show_rays: true,
            show_irradiance: false,
            ray_count: 15,
            adc_bits: 12,
            high_contrast: false,
            force_non_finite: false,
        }
    }
}

impl OpticsSharedParams {
    /// Converts shared parameters into an [`OpticalMediumConfig`].
    #[must_use]
    pub fn to_medium(&self) -> OpticalMediumConfig {
        OpticalMediumConfig {
            preset: self.preset,
            domain_width: 1.0,
            domain_height: 1.0,
            frequency: self.frequency,
            slit_separation: self.slit_separation,
            aperture_width: self.aperture_width,
            lens_radius: self.lens_radius,
            droplet_index: 1.50,
            veselago_index: -1.0,
        }
    }

    /// Converts shared parameters into a [`DetectorConfig`].
    #[must_use]
    pub fn to_detector_config(&self) -> DetectorConfig {
        let ref_x = match self.preset {
            OpticsPreset::DoubleSlit => 0.28,
            OpticsPreset::VeselagoLens => 0.26,
            OpticsPreset::LuneburgLens
            | OpticsPreset::MaxwellFishEye
            | OpticsPreset::BiconvexDroplet => 0.24,
        };
        DetectorConfig {
            detector_x: self.detector_x,
            reference_x: ref_x,
            active_span_fraction: 0.78,
            transmitted_power: 1.0,
            lambertian_mode: 1.0,
            delta_r_max: 0.08,
            adc_bits: self.adc_bits,
        }
    }
}

/// Structured snapshot payload delivered from [`OpticsWaveRunner`] to the UI thread.
#[derive(Clone, Debug)]
pub struct OpticsFrameData {
    /// Scalar wave displacement slice $u(x, y, t)$.
    pub field: Vec<f64>,
    /// Time-smoothed irradiance slice $\langle u^2 \rangle(x, y)$.
    pub irradiance: Vec<f64>,
    /// Signed refractive index slice $n(x, y)$.
    pub refractive_index: Vec<f64>,
    /// Obstacle mask slice.
    pub obstacle: Vec<u8>,
    /// Active geometric ray polylines in normalized `[0, 1]^2` coordinates.
    pub rays: Vec<Vec<[f64; 2]>>,
    /// Detected geometric caustic cusp and envelope points in `[0, 1]^2`.
    pub caustics: Vec<[f64; 2]>,
    /// Smoothed irradiance profile along the virtual detector line.
    pub detector_profile: Vec<f64>,
    /// Calibrated optical sensor metrics at `detector_x`.
    pub readout: DetectorReadout,
    /// Active optical medium geometry.
    pub medium: OpticalMediumConfig,
}

/// Background worker simulation runner stepping [`WaveField2D`] and [`HamiltonianRayTracer`].
pub struct OpticsWaveRunner {
    pub(crate) field: WaveField2D,
    pub(crate) ray_tracer: HamiltonianRayTracer,
    pub(crate) detector_buf: Vec<f64>,
    pub(crate) last_readout: DetectorReadout,
    pub(crate) applied_medium: OpticalMediumConfig,
    pub(crate) applied_rays: usize,
    pub(crate) params: Arc<std::sync::RwLock<OpticsSharedParams>>,
    pub(crate) diverged_flag: bool,
}

impl OpticsWaveRunner {
    /// Constructs a pre-allocated runner of size `grid_size x grid_size`.
    pub fn new(params: Arc<std::sync::RwLock<OpticsSharedParams>>, grid_size: usize) -> Self {
        let n = grid_size.clamp(64, 256);
        let dx = 1.0 / (n as f64);
        let config = WaveFieldConfig2D {
            width: n,
            height: n,
            dx,
            dy: dx,
            c0: 1.0,
            sponge_width: (n / 12).max(6),
            sponge_max_damping: 24.0,
            irradiance_decay: 0.03,
        };
        let mut field = WaveField2D::new(config);
        let snap_params = params.read().unwrap_or_else(|e| e.into_inner()).clone();
        let medium = snap_params.to_medium();
        medium.apply_to_wave_field(&mut field);
        let mut ray_tracer = HamiltonianRayTracer::new(32, 360, 256);
        let ray_count = snap_params.ray_count.clamp(3, 31);
        ray_tracer.trace_in_place(&medium, ray_count, 0.0035);
        let mut detector_buf = vec![0.0; n];
        let last_readout = sample_detector_line_in_place(
            &field,
            &snap_params.to_detector_config(),
            &mut detector_buf,
        );
        Self {
            field,
            ray_tracer,
            detector_buf,
            last_readout,
            applied_medium: medium,
            applied_rays: ray_count,
            params,
            diverged_flag: false,
        }
    }

    fn sync_params_in_place(&mut self) -> Option<(OpticalMediumConfig, DetectorConfig, f64)> {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        if p.force_non_finite
            || !p.frequency.is_finite()
            || !p.slit_separation.is_finite()
            || !p.aperture_width.is_finite()
            || !p.lens_radius.is_finite()
            || !p.detector_x.is_finite()
            || !p.cfl_scale.is_finite()
        {
            self.diverged_flag = true;
            return None;
        }
        let medium = p.to_medium();
        let det_cfg = p.to_detector_config();
        let ray_count = p.ray_count.clamp(3, 31);
        let cfl_scale = p.cfl_scale.clamp(0.05, 1.0);
        drop(p);

        if medium != self.applied_medium {
            if medium.preset != self.applied_medium.preset {
                self.field.clear_state();
            }
            medium.apply_to_wave_field(&mut self.field);
            self.ray_tracer.trace_in_place(&medium, ray_count, 0.0035);
            self.applied_medium = medium;
            self.applied_rays = ray_count;
        } else if ray_count != self.applied_rays {
            self.ray_tracer.trace_in_place(&medium, ray_count, 0.0035);
            self.applied_rays = ray_count;
        }
        Some((medium, det_cfg, cfl_scale))
    }
}

impl FiniteStateGuard for OpticsWaveRunner {
    fn is_finite_state(&self) -> bool {
        !self.diverged_flag && self.field.is_finite()
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        None
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }
}

impl SimulationRunner for OpticsWaveRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        if matches!(cmd, SimCommand::Reset) {
            self.diverged_flag = false;
            self.field.clear_state();
            let _ = self.sync_params_in_place();
        }
    }

    fn is_finite_state(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        if !FiniteStateGuard::is_finite_state(self) {
            Some(DiagnosticReport::new(
                self.field.time(),
                self.field.step_count(),
                "wave_field",
                "2D FDTD wave solver encountered non-finite state or parameter instability.",
            ))
        } else {
            None
        }
    }

    fn step(&mut self) {
        if !FiniteStateGuard::is_finite_state(self) {
            return;
        }
        let Some((medium, det_cfg, cfl_scale)) = self.sync_params_in_place() else {
            return;
        };
        let src = medium.default_wave_source(self.field.config());
        let dt = cfl_scale * self.field.cfl_dt();
        self.field.step(dt, Some(&src));
        self.last_readout =
            sample_detector_line_in_place(&self.field, &det_cfg, &mut self.detector_buf);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let (show_irradiance, high_contrast) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.show_irradiance, p.high_contrast)
        };
        let cfg = self.field.config();
        let img = render_wavefield_image(WaveRenderInput {
            field: self.field.field(),
            irradiance: self.field.irradiance(),
            refractive_index: self.field.refractive_index(),
            obstacle: self.field.obstacle_mask(),
            width: cfg.width,
            height: cfg.height,
            show_irradiance,
            high_contrast,
        });
        let mut rays = Vec::with_capacity(self.ray_tracer.ray_count());
        for r in 0..self.ray_tracer.ray_count() {
            rays.push(self.ray_tracer.ray_points(r).to_vec());
        }
        let frame_data = OpticsFrameData {
            field: self.field.field().to_vec(),
            irradiance: self.field.irradiance().to_vec(),
            refractive_index: self.field.refractive_index().to_vec(),
            obstacle: self.field.obstacle_mask().to_vec(),
            rays,
            caustics: self.ray_tracer.caustic_points().to_vec(),
            detector_profile: self.detector_buf.clone(),
            readout: self.last_readout,
            medium: self.applied_medium,
        };
        StateSnapshot {
            width: cfg.width,
            height: cfg.height,
            pixels: Arc::new(std::sync::RwLock::new(img.pixels)),
            custom_data: Vec::new(),
            structured_data: Some(Box::new(frame_data)),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        3
    }
}

/// Input slices and display flags for [`render_wavefield_image`].
pub struct WaveRenderInput<'a> {
    /// Instantaneous 2D scalar displacement slice $u(x, y)$.
    pub field: &'a [f64],
    /// Smoothed irradiance slice $\langle u^2 \rangle(x, y)$.
    pub irradiance: &'a [f64],
    /// Signed refractive index slice $n(x, y)$.
    pub refractive_index: &'a [f64],
    /// Dirichlet obstacle mask (`1` = opaque wall).
    pub obstacle: &'a [u8],
    /// Grid width in cells.
    pub width: usize,
    /// Grid height in cells.
    pub height: usize,
    /// Whether to visualize smoothed irradiance instead of instantaneous wave phase.
    pub show_irradiance: bool,
    /// Whether `ThemeMode::HighContrast` (`>= 3.0:1` on `#000000`) is active.
    pub high_contrast: bool,
}

/// Maps 2D wave amplitude or irradiance plus optical medium geometry onto a contrast-verified [`ColorImage`].
pub fn render_wavefield_image(inp: WaveRenderInput<'_>) -> ColorImage {
    let total = inp.width * inp.height;
    let mut pixels = Vec::with_capacity(total);
    let data = if inp.show_irradiance {
        inp.irradiance
    } else {
        inp.field
    };
    let peak = data
        .iter()
        .fold(
            0.0_f64,
            |acc, &v| if v.is_finite() { acc.max(v.abs()) } else { acc },
        )
        .max(1e-6);

    for idx in 0..total {
        if inp.obstacle.get(idx).copied().unwrap_or(0) != 0 {
            let wall_rgb = if inp.high_contrast {
                [255, 255, 255]
            } else {
                ensure_dark_canvas_contrast([148, 163, 184])
            };
            pixels.push(eframe::egui::Color32::from_rgb(
                wall_rgb[0],
                wall_rgb[1],
                wall_rgb[2],
            ));
            continue;
        }
        let raw = data.get(idx).copied().unwrap_or(0.0);
        let safe = if raw.is_finite() { raw } else { 0.0 };
        let base_t = if inp.show_irradiance {
            (safe.max(0.0) / peak).sqrt().clamp(0.0, 1.0)
        } else {
            (0.5 + 0.5 * (safe / peak)).clamp(0.0, 1.0)
        };
        let n_val = inp.refractive_index.get(idx).copied().unwrap_or(1.0);
        let medium_shift = if n_val < 0.0 {
            -0.12
        } else {
            0.08 * (n_val - 1.0).clamp(-0.5, 1.2)
        };
        let norm_t = (base_t + medium_shift).clamp(0.0, 1.0);
        let rgb = if inp.high_contrast {
            heatmap_color_high_contrast(norm_t)
        } else {
            heatmap_color(norm_t)
        };
        pixels.push(eframe::egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
    }

    ColorImage {
        size: [inp.width, inp.height],
        pixels,
        ..Default::default()
    }
}
