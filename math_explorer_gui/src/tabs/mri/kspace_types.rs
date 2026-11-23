#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::SimWorkerState;
use crate::framework::Camera3D;
use eframe::egui::Color32;
use nalgebra::Vector3;
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Default spatial grid width and height (`64x64`) for WebAssembly and desktop `< 1 ms` budgets.
pub const DEFAULT_STUDIO_GRID_SIZE: usize = 64;
/// Maximum number of compact brush strokes persisted in tool/URL state (`<= 4 KiB` budget).
pub const MAX_STUDIO_STROKES: usize = 16;
/// Sub-voxel Bloch isochromat count modeling reversible $T_2^*$ dephasing and $180^\circ$ refocusing.
pub const ISOCHROMAT_COUNT: usize = 3;
/// Physical field-of-view width and height in meters (`24 cm`).
pub const STUDIO_FOV_METERS: f64 = 0.24;
/// Maximum selected-voxel 3D spin trajectory history length.
pub const MAX_TRAJECTORY_POINTS: usize = 96;

/// Spatial tissue phantom family scanned by the 2D MRI studio.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhantomPreset {
    /// Analytical 10-ellipse Modified Shepp-Logan head phantom.
    SheppLogan,
    /// Reaction-diffusion Schnakenberg activator/inhibitor spot morphology.
    #[default]
    TuringSpots,
    /// Reaction-diffusion Schnakenberg labyrinthine stripe morphology.
    TuringStripes,
}

impl PhantomPreset {
    /// Human-readable label for UI combo boxes and accessibility summaries.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SheppLogan => "Shepp-Logan Ellipses",
            Self::TuringSpots => "Turing Spots (u, v)",
            Self::TuringStripes => "Turing Stripes (u, v)",
        }
    }
}

/// Frequency-domain sampling mask topology centered at k-space DC `(R/2, C/2)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KSpaceMaskPreset {
    /// Fully sampled 2D k-space (`1.0` everywhere).
    #[default]
    Full,
    /// Circular low-pass filter preserving central contrast while smoothing edges.
    LowPass,
    /// Circular high-pass filter erasing central k-space while preserving crisp edges.
    HighPass,
    /// Radial spoke compressed-sensing trajectory through DC.
    Radial,
    /// Uniform Cartesian phase-encode undersampling with central ACS lines.
    Cartesian,
}

impl KSpaceMaskPreset {
    /// Human-readable label for UI buttons and accessibility metadata.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Full => "Full k-Space",
            Self::LowPass => "Low-Pass",
            Self::HighPass => "High-Pass (Center Erased)",
            Self::Radial => "Radial Spokes",
            Self::Cartesian => "Cartesian Undersampled",
        }
    }
}

/// Interactive brush tool mode applied via `on_brush` and `SimCommand::ApplyBrush`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StudioBrushMode {
    /// Erases k-space mask samples (`0.0`) around the cursor.
    #[default]
    KSpaceMaskErase,
    /// Opens/paints k-space mask passband samples (`1.0`) around the cursor.
    KSpaceMaskPaint,
    /// Injects an off-resonance RF spike impulse in k-space (creates herringbone striping).
    KSpaceRfSpike,
    /// Paints a localized RF excitation flip-angle boost onto spatial voxels.
    SpatialFlipAnglePulse,
    /// Paints a localized $\Delta B_0$ off-resonance field pocket inducing $T_2^*$ dropout.
    SpatialDeltaB0Pocket,
}

impl StudioBrushMode {
    /// Compact integer discriminant used in [`CompactBrushStroke`] and `SimCommand` encoding.
    #[must_use]
    pub const fn code(self) -> i32 {
        match self {
            Self::KSpaceMaskErase => 0,
            Self::KSpaceMaskPaint => 1,
            Self::KSpaceRfSpike => 2,
            Self::SpatialFlipAnglePulse => 3,
            Self::SpatialDeltaB0Pocket => 4,
        }
    }

    /// Reconstructs [`StudioBrushMode`] from its compact integer discriminant.
    #[must_use]
    pub const fn from_code(code: i32) -> Option<Self> {
        match code {
            0 => Some(Self::KSpaceMaskErase),
            1 => Some(Self::KSpaceMaskPaint),
            2 => Some(Self::KSpaceRfSpike),
            3 => Some(Self::SpatialFlipAnglePulse),
            4 => Some(Self::SpatialDeltaB0Pocket),
            _ => None,
        }
    }

    /// Returns `true` when this brush targets the centered k-space canvas rather than spatial tissue.
    #[must_use]
    pub const fn targets_kspace(self) -> bool {
        matches!(
            self,
            Self::KSpaceMaskErase | Self::KSpaceMaskPaint | Self::KSpaceRfSpike
        )
    }

    /// Human-readable label for UI brush selectors.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::KSpaceMaskErase => "Erase k-Space Mask",
            Self::KSpaceMaskPaint => "Paint k-Space Passband",
            Self::KSpaceRfSpike => "k-Space RF Spike",
            Self::SpatialFlipAnglePulse => "RF Flip-Angle Pulse",
            Self::SpatialDeltaB0Pocket => "ΔB0 Inhomogeneity Pocket",
        }
    }
}

/// Compact `[mode_code, cx, cy, radius]` brush stroke tuple keeping URL JSON state `< 4 KiB`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompactBrushStroke(pub [i32; 4]);

impl CompactBrushStroke {
    /// Creates a new compact brush stroke record.
    #[must_use]
    pub const fn new(mode: StudioBrushMode, cx: i32, cy: i32, radius: i32) -> Self {
        Self([mode.code(), cx, cy, radius])
    }

    /// Validates stroke mode, coordinates, and radius against `grid_size`.
    pub fn validate(self, grid_size: usize) -> Result<(StudioBrushMode, i32, i32, i32), String> {
        let [code, cx, cy, r] = self.0;
        let mode = StudioBrushMode::from_code(code)
            .ok_or_else(|| format!("Invalid MRI brush stroke mode code {code}"))?;
        let max_coord = grid_size as i32;
        if cx < 0 || cx >= max_coord || cy < 0 || cy >= max_coord {
            return Err(format!(
                "Brush stroke ({cx}, {cy}) outside 0..{grid_size} grid"
            ));
        }
        if !(1..=24).contains(&r) {
            return Err(format!("Brush stroke radius {r} must be in 1..=24"));
        }
        Ok((mode, cx, cy, r))
    }
}

/// Validated, compact state schema for [`super::kspace_tool::MriKSpaceTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MriKSpaceToolState {
    /// Global RF excitation flip angle in degrees (`5.0..=180.0`).
    pub flip_angle_deg: f64,
    /// Repetition time $T_R$ in seconds (`0.1..=5.0`) governing $T_1$ recovery contrast.
    pub tr: f64,
    /// Initial echo/readout delay $T_E$ in seconds (`0.005..=0.4`) governing $T_2$ contrast.
    pub te: f64,
    /// Normalized radial cutoff (`0.05..=1.0`) for low-pass and high-pass k-space masks.
    pub mask_cutoff: f64,
    /// Spatial readout gradient integral $\int G_x dt$ in T·s/m (`-5e-5..=5e-5`).
    pub grad_gx_integral: f64,
    /// Phase-encoding gradient integral $\int G_y dt$ in T·s/m (`-5e-5..=5e-5`).
    pub grad_gy_integral: f64,
    /// Spatial and k-space matrix dimension (`64` or `128`).
    pub grid_size: usize,
    /// Number of radial spokes (`4..=64`) for radial compressed-sensing masks.
    pub radial_spokes: usize,
    /// Phase-encode line stride (`1..=8`) for Cartesian undersampling masks.
    pub cartesian_stride: usize,
    /// Interactive brush radius in grid cells (`1..=16`).
    pub brush_radius: i32,
    /// Selected `[row, col]` voxel inspected in the 3D Bloch readout.
    pub selected_voxel: [usize; 2],
    /// Active tissue phantom family.
    pub phantom: PhantomPreset,
    /// Active baseline k-space mask topology.
    pub mask_preset: KSpaceMaskPreset,
    /// Active interactive brush mode.
    pub brush_mode: StudioBrushMode,
    /// Whether live reaction-diffusion steps advance alongside Bloch evolution.
    pub evolve_turing: bool,
    /// 3D Bloch readout camera pose.
    pub camera: Camera3D,
    /// Bounded compact brush strokes (`<= MAX_STUDIO_STROKES`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub strokes: Vec<CompactBrushStroke>,
    /// Optional worker execution state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<SimWorkerState>,
}

impl Default for MriKSpaceToolState {
    fn default() -> Self {
        Self {
            flip_angle_deg: 90.0,
            tr: 1.2,
            te: 0.03,
            mask_cutoff: 0.30,
            grad_gx_integral: 0.0,
            grad_gy_integral: 0.0,
            grid_size: DEFAULT_STUDIO_GRID_SIZE,
            radial_spokes: 16,
            cartesian_stride: 2,
            brush_radius: 4,
            selected_voxel: [DEFAULT_STUDIO_GRID_SIZE / 2, DEFAULT_STUDIO_GRID_SIZE / 2],
            phantom: PhantomPreset::TuringSpots,
            mask_preset: KSpaceMaskPreset::Full,
            brush_mode: StudioBrushMode::KSpaceMaskErase,
            evolve_turing: false,
            camera: Camera3D::new(0.45, 0.55, 1.0),
            strokes: Vec::new(),
            state: Some(SimWorkerState::Ready),
        }
    }
}

impl MriKSpaceToolState {
    /// Validates finiteness, discrete resolution caps (`64` or `128`), camera pose, and bounded strokes.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("flip_angle_deg", self.flip_angle_deg, 5.0, 180.0),
            ("tr", self.tr, 0.1, 5.0),
            ("te", self.te, 0.005, 0.4),
            ("mask_cutoff", self.mask_cutoff, 0.05, 1.0),
            ("grad_gx_integral", self.grad_gx_integral, -5.0e-5, 5.0e-5),
            ("grad_gy_integral", self.grad_gy_integral, -5.0e-5, 5.0e-5),
        ];
        for (name, val, min, max) in checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "MRI studio '{name}' = {val} must be finite within [{min}, {max}]"
                ));
            }
        }
        if !matches!(self.grid_size, 64 | 128) {
            return Err(format!(
                "MRI studio grid_size {} must be 64 or 128",
                self.grid_size
            ));
        }
        if !(4..=64).contains(&self.radial_spokes) || !(1..=8).contains(&self.cartesian_stride) {
            return Err("MRI studio mask spoke/stride bounds out of range".to_string());
        }
        if !(1..=16).contains(&self.brush_radius) {
            return Err(format!(
                "MRI studio brush_radius {} must be within 1..=16",
                self.brush_radius
            ));
        }
        if self.selected_voxel[0] >= self.grid_size || self.selected_voxel[1] >= self.grid_size {
            return Err("MRI studio selected_voxel exceeds grid_size".to_string());
        }
        if self.strokes.len() > MAX_STUDIO_STROKES {
            return Err(format!(
                "MRI studio stroke count {} exceeds cap {}",
                self.strokes.len(),
                MAX_STUDIO_STROKES
            ));
        }
        let _ = Camera3D::from_preset_fields(Some(self.camera), None, None, None, None)?;
        for s in &self.strokes {
            let _ = s.validate(self.grid_size)?;
        }
        Ok(())
    }
}

/// Lock-shared parameters read by [`super::kspace_runner::MriKSpaceRunner`] without heap allocation.
#[derive(Clone, Debug)]
pub struct MriKSpaceSharedParams {
    /// Active validated parameter configuration.
    pub config: MriKSpaceToolState,
    /// Whether high-contrast accessibility colormap rendering is active.
    pub high_contrast: bool,
}

/// Structured worker frame payload consumed by [`super::kspace_tool::MriKSpaceTool`].
#[derive(Clone)]
pub struct MriKSpaceSnapshot {
    /// Grid width and height (`64` or `128`).
    pub grid_size: usize,
    /// Elapsed time since last RF excitation pulse in seconds.
    pub current_time: f64,
    /// Recorded $\tau$ delay when a $180^\circ$ spin-echo pulse has been applied ($T_E = 2\tau$).
    pub echo_tau: Option<f64>,
    /// Active k-space coordinate $(k_x, k_y)$ in cycles/meter from `scanner::k_space_coordinate`.
    pub k_coord_cycles_per_m: [f64; 2],
    /// Accumulated gradient phase at the selected voxel in radians from `scanner::accumulated_phase`.
    pub selected_grad_phase_rad: f64,
    /// Colored spatial phantom raster (`grid_size * grid_size`).
    pub phantom_rgba: Vec<Color32>,
    /// Colored centered log-k-space raster (`grid_size * grid_size`).
    pub kspace_rgba: Vec<Color32>,
    /// Colored $1/(R \cdot C)$-normalized reconstructed image raster (`grid_size * grid_size`).
    pub recon_rgba: Vec<Color32>,
    /// Raw reconstructed voxel magnitude $|I(r, c)|$ for quantitative inspection and tests.
    pub recon_mag: Vec<f64>,
    /// Sub-voxel isochromat magnetization vectors at `selected_voxel`.
    pub selected_spins: [Vector3<f64>; ISOCHROMAT_COUNT],
    /// Net macroscopic magnetization vector $\vec{M}$ at `selected_voxel`.
    pub selected_net_m: Vector3<f64>,
    /// Selected voxel `[rho, t1, t2, delta_b0]` telemetry.
    pub selected_tissue: [f64; 4],
    /// Recent 3D spin trajectory of `selected_voxel`.
    pub selected_trajectory: Vec<Vector3<f64>>,
}

/// [`TheoryDescribable`] metadata for the 2D MRI k-Space & RF Studio.
#[derive(Default, Clone, Debug)]
pub struct MriKSpaceTheory;

impl TheoryDescribable for MriKSpaceTheory {
    fn theory_description(&self) -> String {
        "Couples 2D Fourier k-space encoding S(kx, ky) = ∑ ρ(x, y) exp(-i 2π(kx x + ky y)), \
         separable zero-allocation IDFT reconstruction, spatial Bloch T1/T2 relaxation, Hahn 180° \
         spin-echo refocusing of ΔB0 T2* dephasing, and Schnakenberg Turing reaction-diffusion phantoms."
            .to_string()
    }

    fn phonetic_description(&self) -> String {
        "Two dimensional magnetic resonance k-space Fourier reconstruction, spin echo refocusing, and Turing tissue phantoms"
            .to_string()
    }

    fn theory_citation(&self) -> String {
        "Lauterbur, P. C. (1973). Image Formation by Induced Local Interactions. Nature, 242, 190-191; \
         Hahn, E. L. (1950). Spin Echoes. Physical Review, 80(4), 580-594."
            .to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "k_space_encoding".to_string(),
            "2D spatial frequency plane k(t) = (γ / 2π) ∫ G(τ) dτ sampled before inverse FFT."
                .to_string(),
        );
        map.insert(
            "spin_echo_refocusing".to_string(),
            "A 180° RF pulse at t = τ conjugates static ΔB0 phase dispersion, restoring coherence at TE = 2τ."
                .to_string(),
        );
        map
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(
            r"S(k_x, k_y) = \iint M_{\perp}(x,y,t)\,e^{-i 2\pi (k_x x + k_y y)}\,dx\,dy,\quad \vec{k}(t) = \frac{\gamma}{2\pi}\int_0^t \vec{G}(\tau)\,d\tau"
                .to_string(),
        )
    }
}

/// Starter preset JSON envelope for the 2D MRI k-Space & RF Studio.
pub const PRESET_MRI_KSPACE_STUDIO: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "MRI k-Space & RF Studio - Radial Compressed Sensing on Turing Tissue",
    "domain": "mri",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "flip_angle_deg": 75.0,
    "tr": 0.95,
    "te": 0.025,
    "mask_cutoff": 0.35,
    "grad_gx_integral": 0.0,
    "grad_gy_integral": 0.0,
    "grid_size": 64,
    "radial_spokes": 20,
    "cartesian_stride": 2,
    "brush_radius": 4,
    "selected_voxel": [32, 32],
    "phantom": "turing_spots",
    "mask_preset": "radial",
    "brush_mode": "k_space_mask_erase",
    "evolve_turing": false,
    "camera": {
      "pitch": 0.45,
      "yaw": 0.55,
      "zoom": 1.0
    }
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "mri_kspace_turing_studio",
        title: "2D k-Space & Turing RF Studio",
        description: "S(kx,ky) = F{M_xy(u,v)}: Interactive k-space masks, herringbone RF spikes, and 180° spin-echo rescue on Turing tissues.",
        category: "Medical Physics",
        icon: "🧠",
        domain: "mri",
        tool_name: Some("MriKSpaceStudio"),
        preset_generator: || PRESET_MRI_KSPACE_STUDIO,
    }
}
