#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Interactive 2D Helmholtz angular-spectrum diffraction and Gerchberg-Saxton phase-holography tool.

// [cite:graph_parameters_rust]

use std::cell::RefCell;

use eframe::egui;
use math_explorer::physics::optics::holography::HologramTargetPreset;
use serde::{Deserialize, Serialize};

use crate::framework::{ExportCapabilities, InteractiveTool, ToolMetadata, ToolStateBounds};
use crate::web_io::PlatformInstant;

#[path = "canvas.rs"]
mod canvas;
#[path = "sim.rs"]
mod sim;
#[path = "ui.rs"]
mod ui;

pub(crate) use sim::WorkbenchCache;

pub(crate) const DIFF_GRID: usize = 128;
pub(crate) const CARPET_STEPS: usize = 96;
pub(crate) const GS_GRID: usize = 64;
pub(crate) const REAL_SCALAR_TWIN_FLOOR: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Primary physics engine mode inside [`FourierOpticsWorkbenchTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OpticsMode {
    /// Forward/backward 2D Helmholtz angular-spectrum diffraction and Talbot carpets.
    #[default]
    HelmholtzDiffraction,
    /// Iterative Gerchberg-Saxton unit-modulus spatial light modulator phase retrieval.
    GerchbergSaxtonHolography,
}

impl OpticsMode {
    /// Human-readable mode label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::HelmholtzDiffraction => "Helmholtz Diffraction",
            Self::GerchbergSaxtonHolography => "Gerchberg-Saxton Holography",
        }
    }
}

/// Complex transmission aperture generator selectable in diffraction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AperturePreset {
    /// Opaque circular disk illuminated by a unit plane wave (Arago-Poisson bright spot).
    #[default]
    OpaqueCircularDisk,
    /// Single-slit or multi-slit vertical aperture mask.
    SingleMultiSlit,
    /// Periodic binary Ronchi diffraction grating.
    RonchiGrating,
    /// Circular pupil with quadratic thin-lens phase and optional Zernike aberration.
    ThinQuadraticLens,
}

impl AperturePreset {
    /// Human-readable aperture label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::OpaqueCircularDisk => "Opaque Circular Disk (Arago)",
            Self::SingleMultiSlit => "Single / Multi-Slit",
            Self::RonchiGrating => "Ronchi Periodic Grating",
            Self::ThinQuadraticLens => "Thin Quadratic Lens + Zernike",
        }
    }
}

/// 2D raster visualization mode for viewport display and PNG/SVG/GIF export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OpticsDisplayMode {
    /// Transverse intensity I(x, y, z) or replay intensity |U_replay(x, y)|^2.
    #[default]
    IntensityXY,
    /// Wrapped wavefront phase arg U(x, y, z) or continuous SLM phase phi(x, y).
    WrappedPhaseXY,
    /// Axial propagation carpet I(x, z) from z = 0 to z_max (or target amplitude map in GS mode).
    AxialCarpetXZ,
    /// Calibrated 8-bit [0..=255] grayscale spatial light modulator phase mask.
    SlmPhaseMask8Bit,
}

impl OpticsDisplayMode {
    /// Human-readable display view label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::IntensityXY => "Transverse Intensity (x, y)",
            Self::WrappedPhaseXY => "Wrapped Wavefront Phase",
            Self::AxialCarpetXZ => "Axial Carpet (x, z) / Target",
            Self::SlmPhaseMask8Bit => "8-Bit SLM Phase Mask [0..255]",
        }
    }
}

/// Bounded JSON state persisted for URL sharing and Starter Gallery presets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FourierOpticsState {
    /// Active simulation mode.
    pub mode: OpticsMode,
    /// Active aperture preset for Helmholtz diffraction.
    pub aperture: AperturePreset,
    /// Active 2D raster display mode.
    pub display: OpticsDisplayMode,
    /// Monochromatic vacuum wavelength lambda in nanometers [380, 1550].
    pub wavelength_nm: f64,
    /// Transverse grid pixel pitch Delta x = Delta y in micrometers [2, 25].
    pub pixel_pitch_um: f64,
    /// Axial propagation distance z in millimeters [-150, 150].
    pub propagation_mm: f64,
    /// Maximum axial range z_max in millimeters [1, 200] for the (x, z) carpet.
    pub carpet_max_mm: f64,
    /// Characteristic aperture scale (disk radius, slit width, period, or pupil radius) in um.
    pub feature_scale_um: f64,
    /// Center-to-center slit pitch in micrometers [16, 400].
    pub slit_pitch_um: f64,
    /// Number of parallel slits [1..=7].
    pub slit_count: usize,
    /// Ronchi grating open duty cycle [0.1, 0.9].
    pub duty_cycle: f64,
    /// Thin-lens paraxial focal length f in millimeters [2, 120].
    pub focal_length_mm: f64,
    /// Zernike radial degree n [0..=6].
    pub zernike_n: u32,
    /// Zernike azimuthal frequency m [-6..=6].
    pub zernike_m: i32,
    /// Zernike RMS wavefront aberration in waves [-2, 2].
    pub zernike_rms_waves: f64,
    /// Asymmetric amplitude target for Gerchberg-Saxton phase retrieval.
    pub gs_target: HologramTargetPreset,
    /// When true, couples SLM to replay via near-field angular spectrum at propagation_mm.
    #[serde(default)]
    pub gs_lensless: bool,
    /// Gerchberg-Saxton iteration budget [1..=100].
    pub gs_iterations: usize,
    /// Deterministic OxidizeRng seed for initial SLM phase synthesis.
    pub gs_seed: u64,
    /// Toggle high-contrast perceptual colormap.
    #[serde(default)]
    pub high_contrast_colormap: bool,
    /// Continuous axial-sweep or iteration playback state.
    #[serde(default)]
    pub is_playing: bool,
}

impl Default for FourierOpticsState {
    fn default() -> Self {
        Self {
            mode: OpticsMode::HelmholtzDiffraction,
            aperture: AperturePreset::OpaqueCircularDisk,
            display: OpticsDisplayMode::IntensityXY,
            wavelength_nm: 500.0,
            pixel_pitch_um: 8.0,
            propagation_mm: 16.384,
            carpet_max_mm: 40.0,
            feature_scale_um: 128.0,
            slit_pitch_um: 96.0,
            slit_count: 2,
            duty_cycle: 0.5,
            focal_length_mm: 14.0,
            zernike_n: 0,
            zernike_m: 0,
            zernike_rms_waves: 0.0,
            gs_target: HologramTargetPreset::OffAxisRingAndSpot,
            gs_lensless: false,
            gs_iterations: 50,
            gs_seed: 0x0000_002E_9A8B_3D1E,
            high_contrast_colormap: false,
            is_playing: false,
        }
    }
}

/// Interactive Fourier diffraction & Gerchberg-Saxton phase-holography workbench.
pub struct FourierOpticsWorkbenchTool {
    pub(crate) state: FourierOpticsState,
    pub(crate) cache: RefCell<Option<WorkbenchCache>>,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) texture_key: u64,
    pub(crate) focused_canvas: Option<egui::Id>,
    pub(crate) last_tick: PlatformInstant,
    pub(crate) revision: u64,
}

impl Default for FourierOpticsWorkbenchTool {
    fn default() -> Self {
        Self {
            state: FourierOpticsState::default(),
            cache: RefCell::new(None),
            texture: None,
            texture_key: 0,
            focused_canvas: None,
            last_tick: PlatformInstant::now(),
            revision: 1,
        }
    }
}

impl InteractiveTool for FourierOpticsWorkbenchTool {
    fn name(&self) -> &'static str {
        "Fourier Optics Workbench"
    }

    fn domain(&self) -> &'static str {
        "optics"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn state_revision(&self) -> u64 {
        self.revision
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }

    fn has_finite_parameters(&self) -> bool {
        let s = &self.state;
        s.wavelength_nm.is_finite()
            && s.pixel_pitch_um.is_finite()
            && s.propagation_mm.is_finite()
            && s.carpet_max_mm.is_finite()
            && s.feature_scale_um.is_finite()
            && s.slit_pitch_um.is_finite()
            && s.duty_cycle.is_finite()
            && s.focal_length_mm.is_finite()
            && s.zernike_rms_waves.is_finite()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.clamp_state_in_place();
        self.revision = self.revision.wrapping_add(1);
        true
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        self.ensure_evaluated();
        let guard = self.cache.borrow();
        let Some(c) = guard.as_ref() else {
            return Vec::new();
        };
        let rel_power_err = if c.power_in > 1.0e-24 {
            (c.power_out - c.power_in).abs() / c.power_in
        } else {
            0.0
        };
        vec![
            ("fresnel_number", c.fresnel_nf),
            ("talbot_length_mm", c.talbot_mm),
            ("center_intensity", c.center_intensity),
            ("parseval_rel_error", rel_power_err),
            ("target_nrmse", c.gs_metrics.target_nrmse),
            ("diffraction_efficiency", c.gs_metrics.diffraction_efficiency),
            ("twin_leakage", c.gs_metrics.twin_leakage),
        ]
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::DEFAULT
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, raw: &str) -> Result<(), String> {
        self.state_bounds().validate_json(raw)?;
        let loaded: FourierOpticsState =
            serde_json::from_str(raw).map_err(|e| format!("Invalid optics state JSON: {e}"))?;
        if !loaded.wavelength_nm.is_finite()
            || !loaded.pixel_pitch_um.is_finite()
            || !loaded.propagation_mm.is_finite()
            || !loaded.carpet_max_mm.is_finite()
            || !loaded.feature_scale_um.is_finite()
            || !loaded.slit_pitch_um.is_finite()
            || !loaded.duty_cycle.is_finite()
            || !loaded.focal_length_mm.is_finite()
            || !loaded.zernike_rms_waves.is_finite()
        {
            return Err("Non-finite parameter in optics state JSON".to_string());
        }
        self.state = loaded;
        self.clamp_state_in_place();
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        Vec::new()
    }

    fn export_raster_image(&self) -> Option<egui::ColorImage> {
        Some(self.build_active_raster_image())
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.collect_export_datasets()
    }

    fn step_simulation(&mut self) {
        self.advance_playback_step();
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_workbench(ui);
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

impl scientific_metadata::theory::TheoryDescribable for FourierOpticsWorkbenchTool {
    fn theory_description(&self) -> String {
        format!(
            "Propagates scalar monochromatic Helmholtz wavefields U(x, y, z) via exact 2D angular-spectrum transfer H(fx, fy; z) = exp(i z sqrt(k^2 - kx^2 - ky^2)) with evanescent mode filtering at wavelength {:.0} nm, and synthesizes unit-modulus Gerchberg-Saxton SLM phase holograms exp(i phi(x, y)) eliminating conjugate twin-image symmetry ({}, {}).",
            self.state.wavelength_nm,
            self.state.mode.label(),
            self.state.display.label()
        )
    }

    fn phonetic_description(&self) -> String {
        "Helmholtz angular spectrum wave diffraction, Talbot self-imaging, Zernike aberrations, and Gerchberg-Saxton phase holography".into()
    }

    fn theory_citation(&self) -> String {
        "Goodman, Introduction to Fourier Optics, 4th ed. (2017); Gerchberg & Saxton, Optik 35, 237-246 (1972); Rayleigh, Phil. Mag. 11, 196-205 (1881)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    ToolMetadata {
        id: "FourierOpticsWorkbenchTool",
        display_name: "Fourier Optics Workbench",
        domain: "optics",
        tags: &["optics", "diffraction", "holography", "fourier", "talbot", "zernike"],
        build: || Box::new(FourierOpticsWorkbenchTool::default()),
    }
}
